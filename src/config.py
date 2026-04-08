import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch


@dataclass
class PipelineConfig:
    """파이프라인 설정을 관리하는 데이터 클래스"""

    # 모델 경로
    MODEL_PATH: str = 'data/models/MangaTextExtractor-V1.pt'
    FONT_STYLE_MODEL_PATH: str = 'data/models/font_style_analyzer.pth'
    FONT_SIZE_MODEL_PATH: str = 'data/models/font_size_predictor.pth'

    # 디렉토리
    INPUT_DIR: str = 'data/inputs/'
    DEBUG_CROPS_DIR: str = 'data/debug_crops'

    # API 설정
    SYSTEM_PROMPT_PATH: str = 'prompt.txt'
    API_KEY_FILE: str = 'api_key.txt'
    GEMINI_MODEL: str = 'gemini-2.5-flash'

    # 디바이스
    DEVICE: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

    # 탐지 설정
    TARGET_CLASSES: list = field(default_factory=lambda: ['bubble', 'text', 'free_text'])
    CLASSES_TO_ERASE: list = field(default_factory=lambda: ['text', 'free_text'])
    YOLO_CONF_THRESHOLD: float = 0.35
    TEXT_MERGE_OVERLAP_THRESHOLD: float = 0.2

    # 배치 크기
    TRANSLATION_BATCH_SIZE: int = 50
    OCR_BATCH_SIZE: int = 16
    FONT_MODEL_BATCH_SIZE: int = 16
    INPAINT_BATCH_SIZE: int = 8

    # 말풍선 레이아웃
    BUBBLE_PADDING: int = 0
    BUBBLE_EDGE_SAFE_MARGIN: int = 10
    BUBBLE_PADDING_RATIO: float = 0.15
    BUBBLE_ATTACHMENT_THRESHOLD: int = 5
    ATTACHED_BUBBLE_TEXT_MARGIN: int = 5

    # 인페인팅
    INPAINT_CONTEXT_PADDING: int = 50
    INPAINT_MASK_PADDING: int = 0

    # 텍스트 렌더링
    ENABLE_VERTICAL_TEXT: bool = True
    VERTICAL_TEXT_THRESHOLD: int = 4
    VERTICAL_TOLERANCE_RATIO: float = 0.05
    MIN_ROTATION_ANGLE: int = 2
    FONT_SHRINK_THRESHOLD_RATIO: float = 0.75

    # 자유 텍스트
    FREEFORM_PADDING_RATIO: float = 0.05
    FREEFORM_FONT_COLOR: Tuple[int, int, int] = (0, 0, 0)
    FREEFORM_STROKE_COLOR: Tuple[int, int, int] = (255, 255, 255)
    FREEFORM_STROKE_WIDTH: int = 2

    # 이미지 크기
    IMAGE_SIZE: Tuple[int, int] = (256, 256)

    # 폰트 맵
    FONT_MAP: Dict[str, str] = field(default_factory=lambda: {
        "pop": "data/fonts/SDSamliphopangcheTTFOutline.ttf",
        "angry": "data/fonts/a몬스터.ttf",
        "cute": "data/fonts/IM_Hyemin-Bold.ttf",
        "embarrassment": "data/fonts/JejuHallasan.ttf",
        "handwriting": "data/fonts/NanumPen.ttf",
        "narration": "data/fonts/NanumMyeongjo-Bold.ttf",
        "scared": "data/fonts/흔적체.ttf",
        "shouting": "data/fonts/Pretendard-ExtraBold.otf",
        "standard": "data/fonts/Pretendard-SemiBold.otf"
    })

    # 폰트 설정
    FONT_SCALE_FACTOR: int = 1
    FONT_LENGTH_ADJUSTMENT: bool = True
    MIN_FONT_SIZE: int = 5
    MAX_FONT_SIZE: int = 80
    DEFAULT_FONT_SIZE: int = 20
    FONT_UPSCALE_IF_TOO_SMALL: bool = False
    FONT_AREA_FILL_RATIO: float = 0.35

    # 디버그
    SAVE_DEBUG_CROPS: bool = True
    DRAW_DEBUG_BOXES: bool = False

    # OCR 업스케일링
    OCR_UPSCALE_ENABLED: bool = False
    OCR_UPSCALE_FACTOR: int = 1

    @property
    def DEFAULT_FONT_PATH(self) -> str:
        return self.FONT_MAP.get("standard", "")

    def __post_init__(self):
        if self.MIN_FONT_SIZE >= self.MAX_FONT_SIZE:
            raise ValueError(f"MIN_FONT_SIZE({self.MIN_FONT_SIZE}) must be < MAX_FONT_SIZE({self.MAX_FONT_SIZE})")


# 기본 설정 인스턴스 — 하위 호환성을 위해 config.CONSTANT_NAME 패턴 유지
_config = PipelineConfig()


def __getattr__(name):
    """모듈 레벨에서 config.CONSTANT_NAME 접근을 지원합니다."""
    try:
        return getattr(_config, name)
    except AttributeError:
        raise AttributeError(f"module 'src.config' has no attribute '{name}'")
