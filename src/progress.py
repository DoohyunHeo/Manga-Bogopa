from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Optional

import numpy as np


class PipelinePhase(StrEnum):
    LOADING_MODELS = "loading_models"
    PASS1_BATCH = "pass1_batch"
    SAVING_JSON = "saving_json"
    PASS2_PAGE = "pass2_page"
    COMPLETE = "complete"


@dataclass
class ProgressEvent:
    phase: PipelinePhase
    current: int
    total: int
    message: str
    page_name: Optional[str] = None
    image_rgb: Optional[np.ndarray] = None


ProgressCallback = Callable[[ProgressEvent], None]


def noop_callback(event: ProgressEvent) -> None:
    pass
