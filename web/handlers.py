"""파이프라인 실행 및 이벤트 핸들링 모듈"""
import logging
import os
import queue
import tempfile
import threading
from typing import Optional

from src import config
from src.progress import ProgressEvent, PipelinePhase
from web.state import app_state

logger = logging.getLogger(__name__)


def run_pipeline_with_events(
    input_folder: str,
    output_folder: Optional[str],
    yolo_threshold: float,
    batch_size: int,
    draw_debug: bool,
    enable_checkpoint: bool,
):
    """파이프라인을 실행하고 ProgressEvent를 yield하는 제너레이터입니다."""
    if not app_state.is_ready:
        yield ProgressEvent(PipelinePhase.COMPLETE, 0, 0, "모델이 로드되지 않았습니다. 설정을 확인하세요.")
        return

    if not app_state.acquire():
        yield ProgressEvent(PipelinePhase.COMPLETE, 0, 0, "이미 실행 중입니다. 완료 후 다시 시도하세요.")
        return

    try:
        config._config.INPUT_DIR = input_folder
        config._config.YOLO_CONF_THRESHOLD = yolo_threshold
        config._config.TRANSLATION_BATCH_SIZE = batch_size
        config._config.DRAW_DEBUG_BOXES = draw_debug

        out_dir = output_folder or "data/outputs"
        os.makedirs(out_dir, exist_ok=True)

        correction_mode = str(getattr(config, "FONT_SIZE_CORRECTION_MODE", "strong")).lower()
        if correction_mode == "off":
            yield ProgressEvent(
                PipelinePhase.FONT_ANALYSIS,
                0,
                1,
                "폰트 크기 보정: off (모델 예측 크기 그대로 사용)",
            )
        else:
            sub_modes = []
            if getattr(config, "FONT_CHAR_FIT_ENABLED", False):
                sub_modes.append("char_fit")
            if getattr(config, "FONT_STROKE_FIT_ENABLED", False):
                sub_modes.append("stroke_fit")
            sub_label = "+".join(sub_modes) if sub_modes else "cap-only"
            yield ProgressEvent(
                PipelinePhase.FONT_ANALYSIS,
                0,
                1,
                f"폰트 크기 보정: {correction_mode} ({sub_label}, "
                f"floor {config.MODEL_FONT_SIZE_FLOOR_RATIO:.2f}x, "
                f"ceiling {config.MODEL_FONT_SIZE_CEILING_RATIO:.2f}x)",
            )

        if getattr(config, "FONT_STYLE_FALLBACK_ENABLED", True):
            yield ProgressEvent(
                PipelinePhase.FONT_ANALYSIS,
                0,
                1,
                "font style fallback: enabled "
                f"(low-conf < {config.FONT_STYLE_LOW_CONFIDENCE_THRESHOLD:.2f}, "
                f"margin < {config.FONT_STYLE_LOW_MARGIN_THRESHOLD:.2f} -> standard)",
            )

        event_queue: queue.Queue[ProgressEvent] = queue.Queue()

        def on_progress(event: ProgressEvent):
            event_queue.put(event)

        pipeline = app_state.pipeline
        pipeline.callback = on_progress
        pipeline.enable_checkpoint = enable_checkpoint
        pipeline.output_dir = out_dir

        error_holder = [None]

        def _run():
            try:
                pipeline.run()
            except Exception as e:
                logger.error(f"파이프라인 오류: {e}", exc_info=True)
                error_holder[0] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while thread.is_alive() or not event_queue.empty():
            try:
                event = event_queue.get(timeout=0.3)
                yield event
            except queue.Empty:
                continue

        thread.join()

        if error_holder[0]:
            yield ProgressEvent(PipelinePhase.COMPLETE, 0, 0, f"오류 발생: {error_holder[0]}")
        else:
            yield ProgressEvent(PipelinePhase.COMPLETE, 1, 1, "완료", page_name=out_dir)

    finally:
        app_state.release()
