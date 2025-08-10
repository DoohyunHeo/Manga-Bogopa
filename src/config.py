import torch

MODEL_PATH = 'data/models/v22.pt'
INPUT_DIR = 'data/inputs4/'
SYSTEM_PROMPT_PATH = 'prompt.txt'
API_KEY_FILE = 'api_key.txt'
DEBUG_CROPS_DIR = 'data/debug_crops'
TRANSLATION_BATCH_SIZE = 50

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_CLASSES = ['bubble', 'text', 'free_text']
CLASSES_TO_ERASE = ['text', 'free_text']
GEMINI_MODEL = 'gemini-2.5-flash'
YOLO_CONF_THRESHOLD = 0.4

TEXT_MERGE_OVERLAP_THRESHOLD = 0.2

BUBBLE_PADDING = 0
BUBBLE_EDGE_SAFE_MARGIN = 10
INPAINT_CONTEXT_PADDING = 50
INPAINT_MASK_PADDING = 0
ATTACHED_BUBBLE_TEXT_MARGIN = 5
BUBBLE_PADDING_RATIO = 0.15
VERTICAL_TOLERANCE_RATIO = 0.05
MIN_AREA_FILL_RATIO = 0.4
BUBBLE_ATTACHMENT_THRESHOLD = 5

ENABLE_VERTICAL_TEXT = True
VERTICAL_TEXT_THRESHOLD = 4
MIN_ROTATION_ANGLE = 4
FONT_SHRINK_THRESHOLD_RATIO = 0.75

FREEFORM_PADDING_RATIO = 0.05
FREEFORM_FONT_COLOR = (0, 0, 0)
FREEFORM_STROKE_COLOR = (255, 255, 255)
FREEFORM_STROKE_WIDTH = 2

MTL_MODEL_PATH = 'data/models/font_style_analyzer.pth'
FONT_SIZE_MODEL_PATH = 'data/models/font_size_predictor.pth'
LABEL_ENCODER_PATH = 'data/models/style_label_encoder.json'
IMAGE_SIZE = (256, 256)

FONT_MAP = {
    "angry": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/JejuHallasan.ttf",
    "cute": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/IM_Hyemin-Bold.ttf",
    "embarrassment": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/인천교육힘찬.ttf",
    "handwriting": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/NanumPen.ttf",
    "narration": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/NanumMyeongjoBold.ttf",
    "scared": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/흔적체.ttf",
    "shouting": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/Pretendard-ExtraBold.otf",
    "standard": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/Pretendard-SemiBold.otf"
}

DEFAULT_FONT_PATH = FONT_MAP["standard"]
FONT_SCALE_FACTOR = 1
FONT_LENGTH_ADJUSTMENT = True
MIN_FONT_SIZE = 15
MAX_FONT_SIZE = 60
DEFAULT_FONT_SIZE = 20
FONT_UPSCALE_IF_TOO_SMALL = False
FONT_AREA_FILL_RATIO = 0.35

OCR_BATCH_SIZE=16
FONT_MODEL_BATCH_SIZE=16


SAVE_DEBUG_CROPS=True