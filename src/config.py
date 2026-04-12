import json
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Tuple

import torch

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

# JSON 직렬화에서 제외할 필드 (런타임 전용)
_EXCLUDED_FIELDS = {"DEVICE"}


@dataclass
class PipelineConfig:
    """파이프라인 전체 설정을 관리하는 데이터 클래스"""

    # ── API ──
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    SYSTEM_PROMPT: str = ""

    # ── 디렉토리 ──
    INPUT_DIR: str = "data/inputs/"
    OUTPUT_DIR: str = "data/outputs"
    # ── 모델 경로 ──
    MODEL_PATH: str = "data/models/MangaTextExtractor-V2.pt"
    FONT_STYLE_MODEL_PATH: str = "data/models/font_style_analyzer.pth"

    # ── 디바이스 (런타임, JSON 제외) ──
    DEVICE: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # ── 탐지 설정 ──
    YOLO_CONF_THRESHOLD: float = 0.35
    TEXT_MERGE_OVERLAP_THRESHOLD: float = 0.2

    # ── 배치 크기 ──
    TRANSLATION_BATCH_SIZE: int = 50
    OCR_BATCH_SIZE: int = 16
    FONT_MODEL_BATCH_SIZE: int = 16
    INPAINT_BATCH_SIZE: int = 8

    # ── 말풍선 레이아웃 ──
    BUBBLE_EDGE_SAFE_MARGIN: int = 10
    BUBBLE_PADDING_RATIO: float = 0.15
    BUBBLE_ATTACHMENT_THRESHOLD: int = 5
    ATTACHED_BUBBLE_TEXT_MARGIN: int = 5

    # ── 인페인팅 ──
    INPAINT_CONTEXT_PADDING: int = 50
    INPAINT_MASK_PADDING: int = 0

    # ── 텍스트 렌더링 ──
    ENABLE_VERTICAL_TEXT: bool = True
    VERTICAL_TEXT_THRESHOLD: int = 4
    VERTICAL_TOLERANCE_RATIO: float = 0.05
    MIN_ROTATION_ANGLE: int = 2
    FONT_SHRINK_THRESHOLD_RATIO: float = 0.75

    # ── 말풍선 밖 텍스트 ──
    FREEFORM_PADDING_RATIO: float = 0.05
    FREEFORM_FONT_COLOR: Tuple[int, int, int] = (0, 0, 0)
    FREEFORM_STROKE_COLOR: Tuple[int, int, int] = (255, 255, 255)
    FREEFORM_STROKE_WIDTH: int = 2

    # ── 이미지 크기 ──
    IMAGE_SIZE: Tuple[int, int] = (224, 224)

    # ── 폰트 ──
    FONT_DIR: str = "data/fonts"
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

    # ── 폰트 설정 ──
    MIN_FONT_SIZE: int = 5
    MAX_FONT_SIZE: int = 80
    DEFAULT_FONT_SIZE: int = 20
    FONT_AREA_FILL_RATIO: float = 0.35

    # ── 디버그 ──
    DRAW_DEBUG_BOXES: bool = False


    @property
    def DEFAULT_FONT_PATH(self) -> str:
        return self.FONT_MAP.get("standard", "")

    def __post_init__(self):
        if self.MIN_FONT_SIZE >= self.MAX_FONT_SIZE:
            raise ValueError(f"MIN_FONT_SIZE({self.MIN_FONT_SIZE}) must be < MAX_FONT_SIZE({self.MAX_FONT_SIZE})")

    # ── JSON 저장/로드 ──

    def to_dict(self) -> dict:
        """JSON 직렬화 가능한 딕셔너리로 변환합니다. 런타임 전용 필드는 제외."""
        d = {}
        for f in fields(self):
            if f.name in _EXCLUDED_FIELDS:
                continue
            val = getattr(self, f.name)
            # tuple → list (JSON 호환)
            if isinstance(val, tuple):
                val = list(val)
            d[f.name] = val
        return d

    def save(self, path: str = None):
        """설정을 JSON 파일로 저장합니다."""
        path = path or CONFIG_PATH
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def update_from_dict(self, d: dict):
        """딕셔너리에서 값을 로드하여 현재 설정에 반영합니다."""
        for f in fields(self):
            if f.name in d and f.name not in _EXCLUDED_FIELDS:
                val = d[f.name]
                # list → tuple 복원
                current = getattr(self, f.name)
                if isinstance(current, tuple) and isinstance(val, list):
                    val = tuple(val)
                setattr(self, f.name, val)


def _load_config() -> PipelineConfig:
    """config.json이 있으면 로드, 없으면 기본값으로 생성합니다."""
    cfg = PipelineConfig()
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg.update_from_dict(data)
    else:
        # 첫 실행: prompt.txt가 있으면 내용을 SYSTEM_PROMPT로 마이그레이션
        prompt_path = os.path.join(os.path.dirname(CONFIG_PATH), "prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                cfg.SYSTEM_PROMPT = f.read()
        cfg.save()
    return cfg


# 싱글톤 인스턴스
_config = _load_config()


def is_configured() -> bool:
    """API 키가 설정되어 있는지 확인합니다."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        return True
    return bool(_config.GEMINI_API_KEY.strip())


def get_api_key() -> str:
    """API 키를 반환합니다. 환경변수 우선."""
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    return env_key if env_key else _config.GEMINI_API_KEY.strip()


def save():
    """현재 설정을 JSON 파일로 저장합니다."""
    _config.save()


def reload():
    """config.json에서 설정을 다시 로드합니다."""
    global _config
    _config = _load_config()


def __getattr__(name):
    """모듈 레벨에서 config.CONSTANT_NAME 접근을 지원합니다."""
    try:
        return getattr(_config, name)
    except AttributeError:
        raise AttributeError(f"module 'src.config' has no attribute '{name}'")
