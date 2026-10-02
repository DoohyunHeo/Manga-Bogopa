import json
import os
import re
import shutil
from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Tuple

import torch

from src.font_family import BoldFont, bold_partner

# 새 GPU(RTX 50 시리즈 등)는 처음 쓰는 커널을 드라이버가 번역해 디스크 캐시에 둔다. 기본 한도(1GB)가 다른
# 프로그램 몫으로 차 있으면 실행할 때마다 다시 번역해 첫 글자 찾기가 12초쯤 걸린다(한도를 늘리면 2.5초) —
# 한도를 최대(4GB)로 늘리되 환경 변수에 이미 정해 둔 값은 그대로 둔다. CUDA를 처음 쓰기(아래 DEVICE 판정) 전이어야 한다.
os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024 ** 3))

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
PROMPT_PATH = os.path.join(os.path.dirname(CONFIG_PATH), "prompt.txt")

# 화면에 있는 것만 설정이다 — config.json에 저장하고 다시 읽는 값은 웹 화면에서 바꿀 수 있는 이것뿐이다.
# 나머지 필드는 모두 코드 고정값이라 저장하지 않고, 예전 config.json에 남은 값도 읽지 않는다
# (저장해 두면 코드 기본값을 고쳐도 반영되지 않는다). 실험 스크립트는 실행 중에 바꿔 쓰면 된다.
USER_SETTINGS = {
    "AGY_PATH",              # 번역 CLI(agy) 위치 — 자동으로 못 찾을 때만 적는다
    "TRANSLATION_BACKEND",   # 번역에 쓸 CLI: auto / antigravity / claude / codex
    "TRANSLATION_MODELS",    # CLI마다 고른 모델 (빈 값 = CLI 기본)
    "TRANSLATION_EFFORTS",   # CLI마다 고른 생각 깊이 (예전 AGY_EFFORT '번역 품질'을 대신한다)
    "AGY_SESSIONS",          # 번역 동시 진행: 번역 요청을 동시에 몇 묶음까지 보낼지 (1~8)
    "FONT_MAP",              # 원문 글씨 모양별 글꼴
    "ENABLE_VERTICAL_TEXT",  # 좁고 긴 말풍선은 세로로 쓰기
    "LOW_VRAM_MODE",         # 그래픽카드 메모리 아끼기
    "NARRATION_AS_STANDARD", # 나레이션체(명조)도 평문체로 쓰기
    "ENABLE_FIT_TRANSLATION",  # 좁은 말풍선은 짧은 번역을 한 번 더 받기
    "SLANT_EXCLAMATIONS",    # 외침(느낌표 대사)은 기울여 쓰기
}
# INPUT_DIR·OUTPUT_DIR은 설정이 아니다 — 웹 화면은 책장(library.json)에서 책마다 폴더를 기억하고
# 실행할 때 프로세스 안에서만 바꾼다. 스크립트도 실행 중에 바꿔 쓴다.
AGY_SESSIONS_RANGE = (1, 8)
# 번역 CLI — auto면 설치된 것 중 이 순서로 고른다 (src/translator.resolve_backend)
TRANSLATION_BACKENDS = ("antigravity", "claude", "codex")
TRANSLATION_BACKEND_CHOICES = ("auto",) + TRANSLATION_BACKENDS
# CLI마다 받는 생각 깊이 — agy·claude는 --help에 적힌 값, Codex는 모델 목록(codex debug models)의 모델별 값을 합친 것.
# 모델마다 받는 값이 다를 수 있어 화면은 translator.backend_options로 고르게 한다
TRANSLATION_EFFORT_LEVELS = {
    "antigravity": ("low", "medium", "high", "max"),
    "claude": ("low", "medium", "high", "xhigh", "max"),
    "codex": ("low", "medium", "high", "xhigh", "max", "ultra"),
}
# medium = 번역 품질이 무너지지 않는 가장 낮은 단계 (agy에서 low는 뜻이 뒤집히는 오역·홍보 문구 누락이 나왔다).
# high는 번역 요청의 생각(출력 토큰)이 약 3배, 번역 시간이 2~3배라 기본으로 두지 않는다 — 화면에서 '꼼꼼하게'로 고를 수 있다
DEFAULT_TRANSLATION_EFFORT = "medium"
# CLI에 넘기는 모델 이름 모양 (gemini-3.8-flash-high, opus, sonnet[1m], gpt-5.5 …)
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\[\]-]{0,99}$")


def valid_model_name(name: str) -> bool:
    """CLI에 넘길 수 있는 모델 이름 모양인지 (빈 값 = CLI 기본은 따로 본다)."""
    return bool(_MODEL_NAME.match(str(name or "")))

# 메모리 아끼기를 켜면 절반으로 줄이는 묶음 크기 (끄면 이 값 그대로)
_BATCH_SIZES = {
    "DETECTION_BATCH_SIZE": 8,
    "OCR_BATCH_SIZE": 16,
    "FONT_MODEL_BATCH_SIZE": 16,
    "INPAINT_BATCH_SIZE": 8,
    "PASS2_MICROBATCH_SIZE": 4,
}


@dataclass
class PipelineConfig:
    """파이프라인 전체 설정을 관리하는 데이터 클래스"""

    # ── 번역 (Antigravity CLI · Claude Code · Codex CLI) ──
    # agy 실행 파일 경로 — 비우면 PATH와 기본 설치 위치(%LOCALAPPDATA%\agy\bin)에서 찾는다.
    # Claude Code·Codex는 PATH와 기본 설치 위치(홈 폴더의 .local\bin, %APPDATA%\npm)에서 찾는다
    AGY_PATH: str = ""
    # 번역에 쓸 CLI — auto면 설치된 것 중 agy → Claude Code → Codex 순 (TRANSLATION_BACKENDS)
    TRANSLATION_BACKEND: str = "auto"
    # CLI마다 고른 모델 — 빈 값이면 CLI 기본 모델. CLI를 바꿔도 각자 고른 값이 남는다
    TRANSLATION_MODELS: Dict[str, str] = field(default_factory=lambda: {b: "" for b in TRANSLATION_BACKENDS})
    # CLI마다 고른 생각 깊이 (TRANSLATION_EFFORT_LEVELS 가운데, 기본 medium). 짧은 번역·확인 같은 곁가지 요청은
    # 이 값과 상관없이 low다 (translator.SIDE_EFFORT)
    TRANSLATION_EFFORTS: Dict[str, str] = field(
        default_factory=lambda: {b: DEFAULT_TRANSLATION_EFFORT for b in TRANSLATION_BACKENDS})
    # 번역 요청을 동시에 몇 개까지 돌릴지 — 쪽을 읽는 동안 뒤에서 번역하고, 번역이 밀리면 함께 돌린다.
    # 요청마다 새 대화를 연다 (대화를 이어 쓰면 앞 내용이 입력으로 다시 들어가 토큰이 약 2배였다)
    AGY_SESSIONS: int = 4
    # 식자를 한 번 맞춰 보고 좁은 말풍선(원문보다 한참 작아지거나 어절이 끊기는 곳)만 번역기에 짧은 번역을 한 번 더
    # 받는다 (src/fit_translation.py). 기본은 끔 — 번역 요청이 한 번 더 들어 시간이 늘어나는 데 비해 나아지는 말풍선이 적다
    ENABLE_FIT_TRANSLATION: bool = False
    # 번역 요청 한 번을 기다리는 최대 시간 (초) — 생각을 오래 하는 모델은 한 번에 몇 분씩 걸리기도 한다
    AGY_TIMEOUT_SEC: int = 1800
    # 번역 지침 — 늘 저장소의 prompt.txt를 읽는다 (_load_config). 화면에서 고치지 않는다
    SYSTEM_PROMPT: str = ""

    # ── 디렉토리 ──
    INPUT_DIR: str = "data/inputs/"
    OUTPUT_DIR: str = "data/outputs"
    # ── 모델 경로 ──
    MODEL_PATH: str = "data/models/MangaTextExtractor-V2.pt"
    # 원문 글씨를 생김새로 나누는 모양 6분류 모델 — 식자 직전에 책 단위로 글씨체를 정한다 (src/font_relative.py)
    FONT_STYLE6_MODEL_PATH: str = "data/models/font_style6_analyzer.pth"

    # ── 디바이스 (런타임, JSON 제외) ──
    DEVICE: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # ── 탐지 설정 ──
    # 0.40보다 0.35가 낫다 — 0.35에서만 잡히는 박스에도 옮겨야 할 글자(손글씨 혼잣말·작은 설명 글 등)가 있다
    YOLO_CONF_THRESHOLD: float = 0.35
    TEXT_MERGE_OVERLAP_THRESHOLD: float = 0.2
    # 탐지 모델이 학습된 해상도(1344)로 추론해야 작은 글자까지 제대로 잡힌다.
    DETECTION_IMGSZ: int = 1344
    DETECTION_BATCH_SIZE: int = 8

    # ── 배치 크기 ──
    # 그래픽카드 메모리 아끼기 — 켜면 _BATCH_SIZES의 묶음 크기를 절반으로 (조금 느려짐)
    LOW_VRAM_MODE: bool = False
    # 쪽을 이만큼씩 읽고, 읽은 묶음은 번역에 넘긴 뒤 바로 다음 묶음을 읽는다 (번역은 뒤에서 돈다).
    # 작을수록 첫 번역이 일찍 시작하고, 다 읽은 뒤 번역만 기다리는 시간이 짧다
    TRANSLATION_BATCH_SIZE: int = 12
    OCR_BATCH_SIZE: int = 16
    OCR_NUM_BEAMS: int = 2
    FONT_MODEL_BATCH_SIZE: int = 16
    INPAINT_BATCH_SIZE: int = 8
    PASS1_IMAGE_LOAD_WORKERS: int = 4
    PASS2_MICROBATCH_SIZE: int = 4
    PASS2_IMAGE_LOAD_WORKERS: int = 4

    # ── 말풍선 레이아웃 ──
    BUBBLE_EDGE_SAFE_MARGIN: int = 10
    # 0.15는 양쪽 30%를 깎고 시작해 한국어 번역문이 과하게 줄어드는 주원인이었음
    BUBBLE_PADDING_RATIO: float = 0.10
    ATTACHED_BUBBLE_TEXT_MARGIN: int = 5

    # ── 패널 테두리 붙음 감지 ──
    BUBBLE_ATTACHMENT_EDGE_RATIO: float = 0.10
    BUBBLE_ATTACHMENT_MIN_LENGTH_RATIO: float = 0.8
    FREEFORM_ATTACHMENT_SEARCH_PX: int = 100
    FREEFORM_ATTACHMENT_MIN_LENGTH_RATIO: float = 0.7

    # ── 후리가나 컬럼 제거 (세로 일본어 전용) ──
    # 메인 한자 옆 좁은 후리가나(읽기용 작은 글자) 컬럼을 굵기 측정 크롭에서만 잘라낸다
    # (src/font_analysis.strip_furigana_column). 글자 읽기와 크기 측정은 원래 크롭을 쓴다.
    VERTICAL_FURIGANA_MIN_GAP_RATIO: float = 0.08

    # ── 인페인팅 ──
    # 만화/애니메이션 특화 LaMa — 읽지 못하면 범용 big-lama로 지운다 (model_loader.load_inpainting_model)
    INPAINT_MANGA_MODEL_PATH: str = "data/models/anime_lama/lama_large_512px.ckpt"
    # 지우기 값 3개(주변 범위 20·말풍선 밖 여유 2·말풍선 안 여유 3) — 더 넓은 값(50·3·8)은 말풍선 테두리를 흐리거나
    # 얼룩·줄을 남긴다
    INPAINT_CONTEXT_PADDING: int = 20
    # 말풍선 밖 글자 지울 영역 확장 (px)
    INPAINT_MASK_PADDING: int = 2
    # 말풍선 안 글자 지울 영역 확장 (px) — 말풍선 테두리는 침범하지 않도록 자동 보호된다.
    # 8로 넉넉하게 잡으면 테두리가 흐려지는 일이 더 많아 3을 쓴다 (위 지우기 값 설명 참고)
    INPAINT_BUBBLE_MASK_PADDING: int = 3

    # ── 텍스트 렌더링 ──
    ENABLE_VERTICAL_TEXT: bool = True
    # 원문이 명조(나레이션체)여도 평문체 글꼴로 쓴다 (굵기 판정도 평문체와 같이 한다)
    NARRATION_AS_STANDARD: bool = False
    # 느낌표가 붙은 말풍선 대사를 가로쓰기일 때 10도 기울여 쓴다 (page_drawer._EXCLAMATION_SLANT). 기본은 끔 —
    # 출간된 한국어판도 기울이는 일이 드물고 출판사마다 다르다
    SLANT_EXCLAMATIONS: bool = False
    VERTICAL_FORCE_ASPECT_RATIO: float = 6.0
    MIN_ROTATION_ANGLE: int = 2
    # 가로로 맞춘 크기가 원문 크기의 이 배보다 작으면 세로쓰기로 바꿀지 본다
    FONT_SHRINK_THRESHOLD_RATIO: float = 0.9

    # ── 원문 크기 따라가기 ──
    # 글씨 크기는 잉크 기하 측정(glyph_metrics)으로 산출한다. 측정 불가 시 휴리스틱(크롭 높이 70%) 폴백.
    # 번역문 크기는 원문 크기의 FLOOR_RATIO~CEILING_RATIO배 안에서 맞춘다 (상한은 넘지 않고, 하한 아래는 벌점 —
    # src/text_fitting.py)
    MODEL_FONT_SIZE_FLOOR_RATIO: float = 0.8
    MODEL_FONT_SIZE_CEILING_RATIO: float = 1.2
    MIN_READABLE_TEXT_SIZE: int = 16

    # ── 말풍선 밖 텍스트 ──
    FREEFORM_PADDING_RATIO: float = 0.05
    # 가로쓰기 프리텍스트가 탐지 박스보다 이 비율만큼 더 넓게/높게 퍼지는 것을 허용
    # (탐지 박스가 빠듯해 한국어 번역문이 들어가며 글씨가 너무 작아지는 것 방지)
    FREEFORM_BOX_OVERFLOW_RATIO: float = 0.2
    FREEFORM_FONT_COLOR: Tuple[int, int, int] = (0, 0, 0)
    FREEFORM_STROKE_COLOR: Tuple[int, int, int] = (255, 255, 255)
    FREEFORM_STROKE_WIDTH: int = 2

    # ── 폰트 ──
    FONT_DIR: str = "data/fonts"
    FONT_MAP: Dict[str, str] = field(default_factory=lambda: {
        "pop": "data/fonts/SDSamliphopangcheTTFOutline.ttf",
        "angry": "data/fonts/a몬스터.ttf",  # 각진 글씨
        "handwriting": "data/fonts/NanumPen.ttf",
        "narration": "data/fonts/GowunBatang-Regular.ttf",  # 말풍선 밖 평문·가는명조 원문 (src/font_relative.py)
        "scared": "data/fonts/흔적체.ttf",  # 호러 글씨
        "shouting": "data/fonts/Pretendard-ExtraBold.otf",  # 고르지 않는다 — apply_font_modes가 standard에서 정한다
        "standard": "data/fonts/Pretendard-SemiBold.otf"
    })
    # 굵은 대사 글꼴(FONT_MAP["shouting"])은 평범한 대사 글꼴의 같은 가족 굵은 글꼴이다 — 설정을 읽거나 바꿀 때마다
    # apply_font_modes가 다시 정하고(src/font_family.py) 저장하지 않는다. 같은 가족 굵은 파일이 없으면 평범한 대사
    # 글꼴 그대로 두고 굵은 대사 글씨 모양의 합성 굵게로 그린다 (synthetic=True)
    BOLD_FONT: Optional[BoldFont] = None

    # ── 폰트 설정 ──
    MIN_FONT_SIZE: int = 5
    # 원문이 100px 넘는 붓글씨·큰 제목도 따라가게 넉넉히 둔다.
    # 크기의 실제 상한은 원문 크기 × MODEL_FONT_SIZE_CEILING_RATIO와 상자 크기다
    MAX_FONT_SIZE: int = 160
    # 글자가 상자 넓이를 채울 최소 비율 — 목표 채움의 하한이고, 이보다 덜 채운 후보는 맞춤 점수를 깎는다
    FONT_AREA_FILL_RATIO: float = 0.15


    @property
    def DEFAULT_FONT_PATH(self) -> str:
        return self.FONT_MAP.get("standard", "")

    def __post_init__(self):
        if self.MIN_FONT_SIZE >= self.MAX_FONT_SIZE:
            raise ValueError(f"MIN_FONT_SIZE({self.MIN_FONT_SIZE}) must be < MAX_FONT_SIZE({self.MAX_FONT_SIZE})")
        self.apply_font_modes()

    def apply_font_modes(self):
        """사용자 설정에서 정해지는 값을 다시 맞춘다 — 번역 CLI 값 정리, 메모리 아끼기 묶음 크기, 굵은 대사 글꼴."""
        # 번역 CLI·모델·생각 깊이 — CLI가 받지 않는 값은 남겨 두지 않는다 (모델은 이름 모양만 본다)
        if self.TRANSLATION_BACKEND not in TRANSLATION_BACKEND_CHOICES:
            self.TRANSLATION_BACKEND = "auto"
        models = self.TRANSLATION_MODELS if isinstance(self.TRANSLATION_MODELS, dict) else {}
        efforts = self.TRANSLATION_EFFORTS if isinstance(self.TRANSLATION_EFFORTS, dict) else {}
        self.TRANSLATION_MODELS, self.TRANSLATION_EFFORTS = {}, {}
        for backend in TRANSLATION_BACKENDS:
            model = str(models.get(backend) or "").strip()
            effort = str(efforts.get(backend) or "").strip().lower()
            self.TRANSLATION_MODELS[backend] = model if valid_model_name(model) else ""
            self.TRANSLATION_EFFORTS[backend] = (effort if effort in TRANSLATION_EFFORT_LEVELS[backend]
                                                 else DEFAULT_TRANSLATION_EFFORT)
        try:
            sessions = int(self.AGY_SESSIONS)
        except (TypeError, ValueError):
            sessions = PipelineConfig.AGY_SESSIONS
        self.AGY_SESSIONS = min(max(sessions, AGY_SESSIONS_RANGE[0]), AGY_SESSIONS_RANGE[1])
        for name, size in _BATCH_SIZES.items():
            setattr(self, name, max(1, size // 2) if self.LOW_VRAM_MODE else size)

        # 굵은 대사 글꼴 = 평범한 대사 글꼴의 같은 가족 굵은 글꼴 (저장된 shouting 값은 쓰지 않는다)
        standard = self.FONT_MAP.get("standard")
        if standard:
            self.BOLD_FONT = bold_partner(standard, self.FONT_DIR)
            self.FONT_MAP["shouting"] = self.BOLD_FONT.path

    # ── JSON 저장/로드 ──

    def to_dict(self) -> dict:
        """JSON에 저장할 사용자 설정(USER_SETTINGS)만 딕셔너리로."""
        d = {}
        for f in fields(self):
            if f.name in _EXCLUDED_FIELDS:
                continue
            val = getattr(self, f.name)
            if f.name == "FONT_MAP":  # 굵은 대사 글꼴은 화면에서 고르지 않는다 — 늘 평범한 대사 글꼴에서 정한다
                val = {style: path for style, path in val.items() if style != "shouting"}
            d[f.name] = val
        return d

    def save(self, path: str = None):
        """설정을 JSON 파일로 저장합니다. 예전처럼 모든 값을 담은 파일은 처음 한 번 .bak으로 남긴다."""
        path = path or CONFIG_PATH
        _backup_full_config(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def update_from_dict(self, d: dict):
        """딕셔너리에서 사용자 설정만 읽어 현재 설정에 반영합니다 (그 밖의 값은 코드 고정값)."""
        for f in fields(self):
            if f.name in d and f.name not in _EXCLUDED_FIELDS:
                setattr(self, f.name, d[f.name])
        # 예전 설정의 번역 품질(AGY_EFFORT)은 Antigravity의 생각 깊이로 옮긴다 (다음 저장부터는 새 이름으로)
        if "TRANSLATION_EFFORTS" not in d and d.get("AGY_EFFORT"):
            self.TRANSLATION_EFFORTS = {**self.TRANSLATION_EFFORTS, "antigravity": str(d["AGY_EFFORT"])}
        # 글꼴은 기본 FONT_MAP에 있는 칸만 쓴다 — 저장된 설정에 그 밖의 칸이 있으면 버린다
        self.FONT_MAP = {style: path for style, path in self.FONT_MAP.items() if style in _FONT_SLOTS}
        # 굵은 대사 글꼴·묶음 크기 같은 정해지는 값은 읽은 설정으로 다시 맞춘다
        self.apply_font_modes()


# 사용자 설정이 아닌 필드 전부 — 저장하지도 읽지도 않는다
_EXCLUDED_FIELDS = {f.name for f in fields(PipelineConfig)} - USER_SETTINGS
# 글꼴 칸 — 기본 FONT_MAP의 키 (shouting은 apply_font_modes가 standard에서 정한다)
_FONT_SLOTS = frozenset(PipelineConfig.__dataclass_fields__["FONT_MAP"].default_factory())


def _backup_full_config(path: str):
    """사용자 설정 밖의 값이 든 예전 config.json을 처음 한 번 config.json.bak으로 남긴다."""
    backup = path + ".bak"
    if not os.path.exists(path) or os.path.exists(backup):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return
    if isinstance(stored, dict) and set(stored) - USER_SETTINGS:
        shutil.copy2(path, backup)


def _load_config() -> PipelineConfig:
    """config.json이 있으면 사용자 설정을 읽고, 없으면 기본값으로 만든다. 번역 지침은 늘 prompt.txt."""
    cfg = PipelineConfig()
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg.update_from_dict(data)
    else:
        cfg.save()
    if os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
            cfg.SYSTEM_PROMPT = f.read()
    return cfg


# 싱글톤 인스턴스
_config = _load_config()


def save():
    """현재 설정을 JSON 파일로 저장합니다."""
    _config.save()


def __getattr__(name):
    """모듈 레벨에서 config.CONSTANT_NAME 접근을 지원합니다."""
    try:
        return getattr(_config, name)
    except AttributeError:
        raise AttributeError(f"module 'src.config' has no attribute '{name}'")
