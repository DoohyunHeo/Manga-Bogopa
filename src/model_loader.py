import torch
import google.generativeai as genai
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

from src import config
from src.batch_manga_ocr import BatchMangaOcr
from src.models import FontClassifierModel, FontSizeModel


def _initialize_gemini():
    """Gemini API 및 챗 세션을 초기화합니다."""
    try:
        with open(config.API_KEY_FILE, 'r') as f:
            api_key = f.read().strip()
        genai.configure(api_key=api_key)
    except FileNotFoundError:
        raise FileNotFoundError(f"'{config.API_KEY_FILE}'을 찾을 수 없습니다.")
    
    with open(config.SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    translation_model = genai.GenerativeModel(config.GEMINI_MODEL)
    chat_session = translation_model.start_chat(history=[
        {'role': 'user', 'parts': [system_prompt]},
        {'role': 'model', 'parts': ["네, 알겠습니다. 이제부터 지시에 따라 번역을 시작하겠습니다."]}
    ])
    print("Gemini 챗 세션 초기화 완료.")
    return chat_session

def _load_vision_models():
    """객체 탐지, OCR, Inpainting 모델을 로드합니다."""
    detection_model = YOLO(config.MODEL_PATH)
    detection_model.to(config.DEVICE)
    ocr_model = BatchMangaOcr(batch_size=config.OCR_BATCH_SIZE)
    lama_model = SimpleLama(device=config.DEVICE)
    print("Detection, OCR, LaMa 모델 로딩 완료.")
    return {
        'detection': detection_model,
        'ocr': ocr_model,
        'inpainting': lama_model,
    }

def _load_font_models():
    """폰트 스타일 및 크기 예측 모델을 로드합니다."""
    font_classifier_model = None
    try:
        checkpoint = torch.load(config.FONT_STYLE_MODEL_PATH, map_location=config.DEVICE)
        style_mapping = checkpoint['style_mapping']
        num_classes = len(style_mapping)
        font_classifier_model = FontClassifierModel(num_classes, style_mapping)
        font_classifier_model.load_state_dict(checkpoint['model_state_dict'])
        font_classifier_model.to(config.DEVICE)
        font_classifier_model.eval()
        print("PyTorch FontClassifier(스타일/각도) 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"경고: FontClassifier 모델 로드 실패. -> {e}")

    font_size_model = None
    try:
        font_size_model = FontSizeModel()
        font_size_model.load_state_dict(torch.load(config.FONT_SIZE_MODEL_PATH, map_location=config.DEVICE))
        font_size_model.to(config.DEVICE)
        font_size_model.eval()
        print("PyTorch FontSize 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"경고: PyTorch FontSize 모델 로드 실패. -> {e}")
    
    return {
        'font_classifier': font_classifier_model,
        'font_size': font_size_model,
    }

def load_all_models():
    """모든 AI 모델과 구글 Gemini 챗 세션을 초기화하고 로드합니다."""
    print("AI 모델 및 챗 세션을 초기화합니다...")

    translator_session = _initialize_gemini()
    vision_models = _load_vision_models()
    font_models = _load_font_models()

    print("모든 모델 초기화 완료.")

    return {
        **vision_models,
        **font_models,
        'translator': translator_session
    }