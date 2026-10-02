import logging
import os

import torch
from simple_lama_inpainting import SimpleLama
from ultralytics import YOLO

from src import config, translator
from src.font_model import FontClassifierModel

logger = logging.getLogger(__name__)


def _infer_num_style_classes(checkpoint):
    style_mapping = checkpoint.get("style_mapping")
    if style_mapping:
        return len(style_mapping)

    state_dict = checkpoint.get("model_state_dict", {})
    style_weight = state_dict.get("head_style.weight")
    if style_weight is not None:
        return int(style_weight.shape[0])

    raise KeyError("Unable to infer style class count from checkpoint.")


def _load_font_checkpoint(model_path, role):
    if not model_path:
        logger.info(f"Skipping font {role} checkpoint load: empty path.")
        return None
    if not os.path.exists(model_path):
        logger.info(f"Skipping font {role} checkpoint load: '{model_path}' does not exist.")
        return None

    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    style_mapping = checkpoint.get("style_mapping", {})
    num_classes = _infer_num_style_classes(checkpoint)
    backbone = checkpoint.get("backbone", "convnextv2_tiny.fcmae_ft_in1k")

    font_model = FontClassifierModel(
        num_classes,
        style_mapping,
        backbone_name=backbone,
    )
    font_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    font_model.checkpoint_path = model_path
    font_model.to(config.DEVICE)
    font_model.eval()

    state_dict = checkpoint.get("model_state_dict", {})
    logger.info(
        "Font model loaded. "
        f"role={role}, path={model_path}, backbone={backbone}, "
        f"style_head={'O' if 'head_style.weight' in state_dict else 'X'}"
    )
    return font_model


def load_translator_session():
    """번역 세션 — 설정에서 고른 번역 CLI(auto면 설치된 것 중 agy → Claude Code → Codex 순)와 그 CLI의 모델·생각 깊이.

    번역 창구가 요청마다 새로 연다. CLI를 못 찾으면 TranslationUnavailable — 실행을 멈추고 안내한다.
    """
    found = translator.resolve_backend()
    if not found:
        raise translator.TranslationUnavailable(translator.missing_backend_message())
    if not config.SYSTEM_PROMPT:
        raise ValueError("번역 지침 파일(prompt.txt)을 찾지 못했습니다. 프로그램 폴더에 prompt.txt가 있는지 확인해 주세요.")

    backend, executable = found
    session = translator.create_session(
        backend, executable, config.SYSTEM_PROMPT,
        model=str(config.TRANSLATION_MODELS.get(backend) or ""),
        effort=str(config.TRANSLATION_EFFORTS.get(backend) or ""),
        timeout_sec=int(config.AGY_TIMEOUT_SEC),
    )
    logger.debug("%s 번역 세션 준비: %s (model=%s, effort=%s)",
                 session.name, executable, session.model or "기본", session.effort or "기본")
    return session


def load_detection_model():
    """글자 탐지 모델 — 자체 YOLO(V2)가 아무것도 못 찾은 자리만 RT-DETR로 보완한다."""
    detection_model = YOLO(config.MODEL_PATH)
    detection_model.to(config.DEVICE)
    from src.detection import HybridDetector, RTDetrDetector
    logger.info("Detection model: YOLO + RT-DETR 보완")
    return HybridDetector(detection_model, RTDetrDetector(config.DEVICE))


def load_ocr_model():
    """글자 읽기 모델 — 겹쳐 읽기 (manga-ocr와 Hayai가 다르게 읽은 조각만 PaddleOCR-VL-For-Manga로 다시 읽는다)."""
    from src.ocr_ensemble import CascadeOcr
    logger.info("OCR: 겹쳐 읽기 (manga-ocr + Hayai, 엇갈리면 PaddleOCR-VL-For-Manga)")
    return CascadeOcr(config.DEVICE, config.OCR_BATCH_SIZE)


def load_detection_ocr_models():
    """Load object detection and OCR models."""
    detection_model = load_detection_model()
    ocr_model = load_ocr_model()
    logger.info("Detection and OCR models loaded.")
    return {
        "detection": detection_model,
        "ocr": ocr_model,
    }


def load_inpainting_model():
    """Load inpainting model.

    만화/애니메이션 파인튜닝 LaMa를 쓰고, 읽지 못하면 범용 big-lama (SimpleLama)로 대신한다.
    말풍선 밖 글자를 글자 모양대로 지울 마스크 모델도 함께 읽는다 (실패하면 박스로 지운다).
    """
    text_mask_model = _load_text_mask_model()
    models = {"text_mask": text_mask_model} if text_mask_model is not None else {}
    try:
        from src.lama_ffc import load_manga_lama
        manga_model = load_manga_lama(config.INPAINT_MANGA_MODEL_PATH, config.DEVICE)
        logger.info("Manga-specialized LaMa model loaded.")
        return {"inpainting": manga_model, **models}
    except Exception as e:
        logger.warning(f"만화 특화 인페인팅 모델 로드 실패, 범용 모델로 대체합니다 -> {e}")

    lama_model = SimpleLama(device=config.DEVICE)
    logger.info("Generic LaMa model loaded.")
    return {"inpainting": lama_model, **models}


def _load_text_mask_model():
    try:
        from src.text_mask import TextMaskModel
        model = TextMaskModel(config.DEVICE)
        logger.info("글자 모양 마스크 모델 로드 완료.")
        return model
    except Exception as e:
        logger.warning(f"글자 모양 마스크 모델 로드 실패, 말풍선 밖 글자도 박스로 지웁니다 -> {e}")
        return None
