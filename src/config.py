import torch

MODEL_PATH = 'data/models/v27.pt'
INPUT_DIR = 'data/inputs/'
SYSTEM_PROMPT_PATH = 'prompt.txt'
API_KEY_FILE = 'api_key.txt'
DEBUG_CROPS_DIR = 'data/debug_crops'
TRANSLATION_BATCH_SIZE = 50

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_CLASSES = ['bubble', 'text', 'free_text']
CLASSES_TO_ERASE = ['text', 'free_text']
GEMINI_MODEL = 'gemini-2.5-flash'
YOLO_CONF_THRESHOLD = 0.35

TEXT_MERGE_OVERLAP_THRESHOLD = 0.2

BUBBLE_PADDING = 0
BUBBLE_EDGE_SAFE_MARGIN = 10
INPAINT_CONTEXT_PADDING = 50
INPAINT_MASK_PADDING = 0
ATTACHED_BUBBLE_TEXT_MARGIN = 5
BUBBLE_PADDING_RATIO = 0.15
VERTICAL_TOLERANCE_RATIO = 0.05
BUBBLE_ATTACHMENT_THRESHOLD = 5

ENABLE_VERTICAL_TEXT = True
VERTICAL_TEXT_THRESHOLD = 4
MIN_ROTATION_ANGLE = 2
FONT_SHRINK_THRESHOLD_RATIO = 0.75

FREEFORM_PADDING_RATIO = 0.05
FREEFORM_FONT_COLOR = (0, 0, 0)
FREEFORM_STROKE_COLOR = (255, 255, 255)
FREEFORM_STROKE_WIDTH = 2

FONT_STYLE_MODEL_PATH = 'data/models/font_style_analyzer_2-v1.pth'
FONT_SIZE_MODEL_PATH = 'data/models/font_size_predictor.pth'

IMAGE_SIZE = (256, 256)

FONT_MAP = {
    "pop": "data/fonts/SDSamliphopangcheTTFOutline.ttf",
    "angry": "data/fonts/a몬스터.ttf",
    "cute": "data/fonts/IM_Hyemin-Bold.ttf",
    "embarrassment": "data/fonts/JejuHallasan.ttf",
    "handwriting": "data/fonts/NanumPen.ttf",
    "narration": "data/fonts/NanumMyeongjo-Bold.ttf",
    "scared": "data/fonts/흔적체.ttf",
    "shouting": "data/fonts/Pretendard-ExtraBold.otf",
    "standard": "data/fonts/Pretendard-SemiBold.otf"
}

DEFAULT_FONT_PATH = FONT_MAP["standard"]
FONT_SCALE_FACTOR = 1
FONT_LENGTH_ADJUSTMENT = True
MIN_FONT_SIZE = 5
MAX_FONT_SIZE = 80
DEFAULT_FONT_SIZE = 20
FONT_UPSCALE_IF_TOO_SMALL = False
FONT_AREA_FILL_RATIO = 0.35

OCR_BATCH_SIZE=16
FONT_MODEL_BATCH_SIZE=16


SAVE_DEBUG_CROPS=True

# True로 설정하면 최종 결과물에 버블, 텍스트, 자유 텍스트 영역을 박스로 그립니다.
DRAW_DEBUG_BOXES=False

# OCR 정확도 향상을 위한 저해상도 텍스트 이미지 업스케일링
OCR_UPSCALE_ENABLED = False

OCR_UPSCALE_FACTOR = 1