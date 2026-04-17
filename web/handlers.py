"""파이프라인 실행 및 이벤트 핸들링 모듈.

handlers는 UI가 저장해 둔 config를 읽어서 pipeline을 실행하고 ProgressEvent를
yield할 뿐, config를 쓰지 않는다. "config 저장"은 UI 계층(`_apply_run_settings`)
단일 진입점에서만 일어난다.
"""
import logging
import os
import queue
import threading
from typing import Optional

from src import config
from src.progress import EventLevel, PipelinePhase, ProgressEvent
from web.state import app_state

logger = logging.getLogger(__name__)


def _emit_font_mode_banner():
    """Yield 초기 설정 요약 이벤트들."""
    correction_mode = str(getattr(config, "FONT_SIZE_CORRECTION_MODE", "strong")).lower()
    if correction_mode == "off":
        yield ProgressEvent(
            PipelinePhase.FONT_ANALYSIS, 0, 1,
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
            PipelinePhase.FONT_ANALYSIS, 0, 1,
            f"폰트 크기 보정: {correction_mode} ({sub_label}, "
            f"floor {config.MODEL_FONT_SIZE_FLOOR_RATIO:.2f}x, "
            f"ceiling {config.MODEL_FONT_SIZE_CEILING_RATIO:.2f}x)",
        )

    if getattr(config, "FONT_STYLE_FALLBACK_ENABLED", True):
        yield ProgressEvent(
            PipelinePhase.FONT_ANALYSIS, 0, 1,
            f"글씨체 판정 폴백: 활성 "
            f"(신뢰도 < {config.FONT_STYLE_LOW_CONFIDENCE_THRESHOLD:.2f} "
            f"또는 격차 < {config.FONT_STYLE_LOW_MARGIN_THRESHOLD:.2f} → standard로 되돌림)",
        )


def run_pipeline_with_events(
    output_folder: Optional[str],
    enable_checkpoint: bool,
):
    """파이프라인을 실행하고 ProgressEvent를 yield하는 제너레이터.

    config 쓰기 없음. 호출 전에 UI가 `_apply_run_settings`로 config를 갱신·저장한
    상태여야 한다.
    """
    if not app_state.is_ready:
        yield ProgressEvent(
            PipelinePhase.COMPLETE, 0, 0,
            "모델이 로드되지 않았습니다. 설정을 확인하세요.",
            level=EventLevel.ERROR,
        )
        return

    if not app_state.acquire():
        yield ProgressEvent(
            PipelinePhase.COMPLETE, 0, 0,
            "이미 실행 중입니다. 완료 후 다시 시도하세요.",
            level=EventLevel.ERROR,
        )
        return

    try:
        out_dir = output_folder or config.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        yield from _emit_font_mode_banner()

        event_queue: queue.Queue[ProgressEvent] = queue.Queue()

        def on_progress(event: ProgressEvent):
            event_queue.put(event)

        pipeline = app_state.pipeline
        pipeline.callback = on_progress
        pipeline.enable_checkpoint = enable_checkpoint
        pipeline.output_dir = out_dir

        def _run():
            try:
                pipeline.run()
            except Exception as e:
                logger.error(f"파이프라인 오류: {e}", exc_info=True)
                event_queue.put(ProgressEvent(
                    PipelinePhase.COMPLETE, 0, 0, f"오류 발생: {e}",
                    level=EventLevel.ERROR,
                ))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        saw_error = False
        while thread.is_alive() or not event_queue.empty():
            try:
                event = event_queue.get(timeout=0.3)
                if event.level == EventLevel.ERROR:
                    saw_error = True
                yield event
            except queue.Empty:
                continue

        thread.join()

        if not saw_error:
            yield ProgressEvent(
                PipelinePhase.COMPLETE, 1, 1, "완료", page_name=out_dir,
            )

    finally:
        app_state.release()
