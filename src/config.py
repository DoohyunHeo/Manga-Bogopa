import torch

MODEL_PATH = 'data/models/v21.pt'
INPUT_DIR = 'data/inputs3/'
SYSTEM_PROMPT_PATH = 'prompt.txt'
API_KEY_FILE = 'api_key.txt'
DEBUG_CROPS_DIR = 'data/debug_crops'
TRANSLATION_BATCH_SIZE = 100

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_CLASSES = ['bubble', 'text', 'free_text']
CLASSES_TO_ERASE = ['text', 'free_text']
GEMINI_MODEL = 'gemini-2.0-flash'

BUBBLE_PADDING = 0
BUBBLE_EDGE_SAFE_MARGIN = 10
INPAINT_CONTEXT_PADDING = 50
INPAINT_MASK_PADDING = 0
ATTACHED_BUBBLE_TEXT_MARGIN = 5
BUBBLE_PADDING_RATIO = 0.15
VERTICAL_TOLERANCE_RATIO = 0.05
MAX_AREA_FILL_RATIO = 0.7
BUBBLE_ATTACHMENT_THRESHOLD = 5

ENABLE_VERTICAL_TEXT = True
VERTICAL_TEXT_THRESHOLD = 4

FREEFORM_PADDING_RATIO = 0.05
FREEFORM_FONT_COLOR = (0, 0, 0)
FREEFORM_STROKE_COLOR = (255, 255, 255)
FREEFORM_STROKE_WIDTH = 2

MTL_MODEL_PATH = 'data/models/mtl_text_analyzer_pytorch.pth'
LABEL_ENCODER_PATH = 'data/models/style_label_encoder.json'
IMAGE_SIZE = (224, 224)

FONT_MAP = {
    "standard": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/NanumBarunGothic.ttf",
    "shouting": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/87MMILSANG-Oblique.ttf",
    "bold": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/Pretendard-Black.ttf",
    "confused": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/IM_Hyemin-Regular.ttf",
    "sad": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/JejuHallasan.ttf",
    "handwriting": "C:/Users/Admin/AppData/Local/Microsoft/Windows/Fonts/Pretendard-Black.ttf"
}

FONT_CLASSIFIER_PATH = 'data/models/font_classifier.keras'
DEFAULT_FONT_PATH = FONT_MAP["standard"]
FONT_CLASS_NAMES = ['bold', 'confused', 'handwriting', 'narration', 'sad', 'shouting', 'standard']
FONT_SCALE_FACTOR = 1
FONT_LENGTH_ADJUSTMENT = True
MIN_FONT_SIZE = 15
MAX_FONT_SIZE = 50
DEFAULT_FONT_SIZE = 20
FONT_UPSCALE_IF_TOO_SMALL = True
FONT_AREA_FILL_RATIO = 0.35

OCR_BATCH_SIZE=64


SAVE_DEBUG_CROPS=True
