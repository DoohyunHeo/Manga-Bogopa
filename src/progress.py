import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Dict, Optional


class PipelinePhase(StrEnum):
    LOADING_MODELS = "loading_models"
    DETECTION = "detection"
    OCR = "ocr"
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
    elapsed_sec: Optional[float] = None
    level: str = EventLevel.INFO
    extras: Dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]


def noop_callback(event: ProgressEvent) -> None:
    pass


class StageClock:
    """한 실행의 단계별 누적 시간 — 끝날 때 어디에 시간이 들었는지 한 줄로 알린다."""

    # 의심 대사 확인은 짧은 번역과 함께 돌고, 짧은 번역 뒤에 더 기다린 시간만 센다
    ORDER = ("모델 준비", "글자 찾기", "글자 읽기", "글자 크기 재기", "번역", "짧은 번역", "의심 대사 확인", "지우기", "식자", "저장")

    def __init__(self):
        self.started_at = time.perf_counter()
        self.seconds: Dict[str, float] = {}

    def add(self, stage: str, seconds: float) -> None:
        self.seconds[stage] = self.seconds.get(stage, 0.0) + seconds

    @contextmanager
    def measure(self, stage: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - started)

    def summary(self, pages: int = 0) -> str:
        total = time.perf_counter() - self.started_at
        parts = [f"{stage} {self.seconds[stage]:.1f}초" for stage in self.ORDER if stage in self.seconds]
        text = "단계별 시간 — " + " · ".join(parts) + f" / 전체 {total:.0f}초"
        if pages:
            text += f" (쪽당 {total / pages:.1f}초)"
        try:
            import torch
            if torch.cuda.is_available():
                text += f" · GPU 메모리 최대 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GB"
        except Exception:  # noqa: BLE001 — 메모리 표시는 곁들이는 정보
            pass
        return text
