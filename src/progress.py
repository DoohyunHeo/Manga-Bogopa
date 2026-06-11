from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Dict, Optional

import numpy as np


class PipelinePhase(StrEnum):
    LOADING_MODELS = "loading_models"
    DETECTION = "detection"
    OCR = "ocr"
    FONT_ANALYSIS = "font_analysis"
    TRANSLATION = "translation"
    PASS1_BATCH = "pass1_batch"
    SAVING_JSON = "saving_json"
    PASS2_PAGE = "pass2_page"
    COMPLETE = "complete"


class EventLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ProgressEvent:
    phase: PipelinePhase
    current: int
    total: int
    message: str
    page_name: Optional[str] = None
    image_rgb: Optional[np.ndarray] = None
    elapsed_sec: Optional[float] = None
    level: str = EventLevel.INFO
    extras: Dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]


def noop_callback(event: ProgressEvent) -> None:
    pass
