import logging
import os

import torch
import google.generativeai as genai
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

from src import config
from src.batch_manga_ocr import BatchMangaOcr
from src.models import FontClassifierModel

logger = logging.getLogger(__name__)


def _initialize_gemini():
    """Gemini API 및 챗 세션을 초기화합니다."""
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
        {'role': 'user', 'parts': [config.SYSTEM_PROMPT]},
        {'role': 'model', 'parts': ["네, 알겠습니다. 이제부터 지시에 따라 번역을 시작하겠습니다."]}
    ])
    logger.info("Gemini 챗 세션 초기화 완료.")
    return chat_session

def load_translator_session():
    """Gemini 번역 세션을 로드합니다."""
    logger.info("Gemini 챗 세션 초기화 중...")
    return _initialize_gemini()


def load_detection_ocr_models():
    """객체 탐지와 OCR 모델을 로드합니다."""
    detection_model = YOLO(config.MODEL_PATH)
    detection_model.to(config.DEVICE)
    ocr_model = BatchMangaOcr(batch_size=config.OCR_BATCH_SIZE)
    logger.info("Detection, OCR 모델 로딩 완료.")
    return {
        'detection': detection_model,
        'ocr': ocr_model,
    }


def load_inpainting_model():
    """Inpainting 모델을 로드합니다."""
    lama_model = SimpleLama(device=config.DEVICE)
    logger.info("LaMa 모델 로딩 완료.")
    return {'inpainting': lama_model}

def load_font_model():
    """폰트 스타일/각도/크기 통합 모델을 로드합니다."""
    font_model = None
    try:
        checkpoint = torch.load(config.FONT_STYLE_MODEL_PATH, map_location=config.DEVICE)
        style_mapping = checkpoint['style_mapping']
        num_classes = len(style_mapping)
        backbone = checkpoint.get('backbone', 'convnextv2_tiny.fcmae_ft_in1k')
        font_model = FontClassifierModel(num_classes, style_mapping, backbone_name=backbone)
        font_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        font_model.to(config.DEVICE)
        font_model.eval()

        has_size = 'head_size.weight' in checkpoint['model_state_dict']
        logger.info(f"FontClassifier 모델 로드 완료. backbone={backbone}, size_head={'O' if has_size else 'X'}")
    except Exception as e:
        logger.warning(f"FontClassifier 모델 로드 실패. -> {e}")

    return {'font_classifier': font_model}

def load_all_models():
    """모든 AI 모델과 구글 Gemini 챗 세션을 초기화하고 로드합니다."""
    logger.info("AI 모델 및 챗 세션을 초기화합니다...")

    translator_session = load_translator_session()
    detection_ocr_models = load_detection_ocr_models()
    inpainting_model = load_inpainting_model()
    font_models = load_font_model()

    logger.info("모든 모델 초기화 완료.")

    return {
        **detection_ocr_models,
        **inpainting_model,
        **font_models,
        'translator': translator_session
    }
