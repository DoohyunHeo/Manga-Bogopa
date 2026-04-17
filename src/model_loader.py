import logging
import os

import google.generativeai as genai
import torch
from simple_lama_inpainting import SimpleLama
from ultralytics import YOLO

from src import config
from src.batch_manga_ocr import BatchMangaOcr
from src.models import FontClassifierModel

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
    checkpoint_cfg = checkpoint.get("config", {})
    size_target_transform = checkpoint_cfg.get("size_target_transform", "identity")
    size_target_reference = checkpoint_cfg.get("size_target_reference", "page_height")
    size_target_kind = checkpoint_cfg.get("size_target_kind", "legacy_ratio")

    font_model = FontClassifierModel(
        num_classes,
        style_mapping,
        backbone_name=backbone,
        size_target_transform=size_target_transform,
    )
    font_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    font_model.size_target_reference = size_target_reference
    font_model.size_target_kind = size_target_kind
    font_model.has_expressive_head = "head_expressive.weight" in checkpoint.get("model_state_dict", {})
    font_model.standard_style_index = next(
        (int(index) for index, name in style_mapping.items() if str(name) == "standard"),
        None,
    )
    font_model.checkpoint_path = model_path
    font_model.to(config.DEVICE)
    font_model.eval()

    state_dict = checkpoint.get("model_state_dict", {})
    logger.info(
        "Font model loaded. "
        f"role={role}, path={model_path}, backbone={backbone}, "
        f"style_head={'O' if 'head_style.weight' in state_dict else 'X'}, "
        f"expressive_head={'O' if 'head_expressive.weight' in state_dict else 'X'}, "
        f"angle_head={'O' if 'head_angle_vec.weight' in state_dict else 'X'}, "
        f"size_head={'O' if 'head_size.weight' in state_dict else 'X'}, "
        f"size_transform={size_target_transform}, size_ref={size_target_reference}"
    )
    return font_model


def _initialize_gemini():
    """Gemini API and chat session initialization."""
    api_key = config.get_api_key()
    if not api_key:
        raise ValueError(
            "Gemini API 키가 설정되지 않았습니다. "
            "웹 UI에서 설정하거나 GEMINI_API_KEY 환경변수를 설정하세요."
        )
    genai.configure(api_key=api_key)

    if not config.SYSTEM_PROMPT:
        raise ValueError("시스템 프롬프트가 설정되지 않았습니다. 설정에서 번역 프롬프트를 입력하세요.")

    translation_model = genai.GenerativeModel(config.GEMINI_MODEL)
    chat_session = translation_model.start_chat(history=[
        {"role": "user", "parts": [config.SYSTEM_PROMPT]},
        {"role": "model", "parts": ["네, 알겠습니다. 이제부터 지시에 따라 번역을 시작하겠습니다."]},
    ])
    logger.info("Gemini chat session initialized.")
    return chat_session


def load_translator_session():
    """Load Gemini translation session."""
    logger.info("Initializing Gemini chat session...")
    return _initialize_gemini()


def load_detection_ocr_models():
    """Load object detection and OCR models."""
    detection_model = YOLO(config.MODEL_PATH)
    detection_model.to(config.DEVICE)
    ocr_model = BatchMangaOcr(batch_size=config.OCR_BATCH_SIZE)
    logger.info("Detection and OCR models loaded.")
    return {
        "detection": detection_model,
        "ocr": ocr_model,
    }


def load_inpainting_model():
    """Load inpainting model."""
    lama_model = SimpleLama(device=config.DEVICE)
    logger.info("LaMa model loaded.")
    return {"inpainting": lama_model}


def load_font_model():
    """Load split font checkpoints first and use the legacy joint model as a fallback."""
    font_appearance_model = None
    font_size_model = None
    legacy_font_model = None

    try:
        font_appearance_model = _load_font_checkpoint(config.FONT_APPEARANCE_MODEL_PATH, "appearance")
    except Exception as e:
        logger.warning(f"Font appearance model load failed. -> {e}")

    try:
        font_size_model = _load_font_checkpoint(config.FONT_SIZE_MODEL_PATH, "size")
    except Exception as e:
        logger.warning(f"Font size model load failed. -> {e}")

    if font_appearance_model is None or font_size_model is None:
        try:
            legacy_font_model = _load_font_checkpoint(config.FONT_STYLE_MODEL_PATH, "legacy")
        except Exception as e:
            logger.warning(f"Legacy font classifier load failed. -> {e}")

        if legacy_font_model is not None:
            if font_appearance_model is None:
                logger.info("Appearance model unavailable; using legacy font classifier for style/angle inference.")
            if font_size_model is None:
                logger.info("Size model unavailable; using legacy font classifier for size inference.")

    if font_appearance_model is None and font_size_model is None and legacy_font_model is None:
        logger.warning("No font inference checkpoint could be loaded.")

    return {
        "font_classifier": legacy_font_model,
        "font_appearance_classifier": font_appearance_model,
        "font_size_regressor": font_size_model,
    }


def load_all_models():
    """Initialize and load all models plus the Gemini chat session."""
    logger.info("Initializing AI models and chat session...")

    translator_session = load_translator_session()
    detection_ocr_models = load_detection_ocr_models()
    inpainting_model = load_inpainting_model()
    font_models = load_font_model()

    logger.info("All models initialized.")

    return {
        **detection_ocr_models,
        **inpainting_model,
        **font_models,
        "translator": translator_session,
    }
