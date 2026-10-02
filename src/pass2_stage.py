import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, Tuple

import cv2

from src import config, inpainter, page_drawer
from src.data_models import PageData
from src.progress import EventLevel, PipelinePhase, ProgressCallback, ProgressEvent, StageClock, noop_callback
from src.utils import read_image_bgr, write_image_bgr

logger = logging.getLogger(__name__)


class Pass2Stage:
    """Pass 2 execution helper for microbatched inpainting and rendering."""

    def __init__(
        self,
        models,
        output_dir: str,
        progress_callback: Optional[ProgressCallback] = None,
        ensure_inpainting_model: Optional[Callable[[], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        clock: Optional[StageClock] = None,
    ):
        self.models = models
        self.output_dir = output_dir
        self.callback = progress_callback or noop_callback
        self.ensure_inpainting_model = ensure_inpainting_model
        self.should_stop = should_stop or (lambda: False)
        self.clock = clock or StageClock()

    def run(self, pages_to_process: Sequence[PageData], image_paths: Sequence[str], ckpt=None,
            done_before: int = 0, total: Optional[int] = None) -> List[str]:
        """지우기·식자를 하고 결과를 저장한다. 만들지 못한 페이지 이름(원본 없음·원본 읽기 실패·저장 실패)을 돌려준다
        — 완료로 치지 않는다.

        두 번에 나눠 부를 때(의심 대사 확인을 기다리는 동안 먼저 쓴 쪽이 있을 때) 진행 번호는 앞서 끝낸 done_before쪽
        뒤로, 전체 total쪽 가운데로 센다."""
        failed: List[str] = []
        if not pages_to_process:
            return failed

        path_map = {os.path.basename(path): path for path in image_paths}
        processable_pages = [page for page in pages_to_process if page.source_page in path_map]
        failed += [page.source_page for page in pages_to_process if page.source_page not in path_map]
        if not processable_pages:
            logger.warning("Pass 2에서 처리 가능한 페이지가 없습니다.")
            return failed

        if self.ensure_inpainting_model:
            with self.clock.measure("모델 준비"):
                self.ensure_inpainting_model()

        microbatch_size = max(1, int(config.PASS2_MICROBATCH_SIZE))
        batch_pages_total = len(processable_pages)
        total_batches = (batch_pages_total + microbatch_size - 1) // microbatch_size
        total_pages = max(int(total or 0), done_before + batch_pages_total)  # 진행 표시의 전체 쪽 수
        completed_pages = done_before

        logger.info(
            "Pass 2 microbatch mode: %d pages, batch size %d",
            batch_pages_total,
            microbatch_size,
        )

        for batch_index, start in enumerate(range(0, batch_pages_total, microbatch_size), start=1):
            if self.should_stop():
                logger.info("중단 요청: Pass 2를 배치 경계에서 멈춥니다 (%d/%d 페이지 완료).",
                            completed_pages, total_pages)
                self.callback(ProgressEvent(
                    PipelinePhase.PASS2_PAGE, completed_pages, total_pages,
                    "중단 요청 — 완성된 페이지까지는 저장되었습니다",
                    level=EventLevel.WARNING,
                ))
                break
            batch_pages = processable_pages[start:start + microbatch_size]
            logger.info("Pass 2 batch %d/%d: %d pages", batch_index, total_batches, len(batch_pages))
            loaded_pages, load_failed = self._load_page_batch(batch_pages, path_map)
            failed += load_failed
            if not loaded_pages:
                continue

            inpaint_started_at = time.perf_counter()
            inpainted_images = inpainter.inpaint_pages_in_batch(self.models, loaded_pages)
            inpaint_elapsed = time.perf_counter() - inpaint_started_at
            self.clock.add("지우기", inpaint_elapsed)
            inpaint_share = inpaint_elapsed / len(loaded_pages)  # 쪽마다 걸린 시간에 지우기 몫도 넣는다

            for page_data, inpainted_image in zip(loaded_pages, inpainted_images):
                page_started_at = time.perf_counter()
                with self.clock.measure("식자"):
                    final_image_rgb = page_drawer.draw_text_on_image(inpainted_image, page_data)

                output_path = os.path.join(self.output_dir, page_data.source_page)
                final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
                with self.clock.measure("저장"):
                    saved = write_image_bgr(output_path, final_image_bgr)
                if not saved:
                    failed.append(page_data.source_page)
                    logger.error("결과 이미지 저장 실패: %s", output_path)
                    self.callback(ProgressEvent(
                        PipelinePhase.PASS2_PAGE, completed_pages, total_pages,
                        f"{page_data.source_page} 저장 실패 — 다음 실행에서 다시 만듭니다",
                        level=EventLevel.WARNING,
                    ))
                    page_data.image_rgb = None
                    continue  # 완료 알림·진행 수·완료 기록은 저장된 쪽만

                completed_pages += 1
                page_elapsed = time.perf_counter() - page_started_at + inpaint_share
                self.callback(
                    ProgressEvent(
                        PipelinePhase.PASS2_PAGE,
                        completed_pages,
                        total_pages,
                        f"{page_data.source_page} 완료 ({page_elapsed:.1f}초)",
                        page_name=page_data.source_page,
                        elapsed_sec=page_elapsed,
                    )
                )

                if ckpt:
                    ckpt.mark_pass2_page_complete(page_data.source_page)

                page_data.image_rgb = None

            self._release_batch(loaded_pages)

        return failed

    def _load_page_batch(self, batch_pages: Sequence[PageData], path_map) -> Tuple[List[PageData], List[str]]:
        """원본을 읽어 붙인 페이지와 읽지 못한 페이지 이름."""
        batch_entries = []
        missing = []
        for page_data in batch_pages:
            original_path = path_map.get(page_data.source_page)
            if not original_path:
                logger.warning("'%s'의 원본 경로를 찾을 수 없습니다.", page_data.source_page)
                missing.append(page_data.source_page)
                continue
            batch_entries.append((page_data, original_path))

        if not batch_entries:
            return [], missing

        worker_count = min(max(1, int(config.PASS2_IMAGE_LOAD_WORKERS)), len(batch_entries))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            images_bgr = list(executor.map(read_image_bgr, [path for _, path in batch_entries]))

        loaded_pages = []
        failed_names = []
        for (page_data, original_path), image_bgr in zip(batch_entries, images_bgr):
            if image_bgr is None:
                failed_names.append(page_data.source_page)
                logger.warning("'%s' 로딩 실패", original_path)
                continue
            page_data.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            loaded_pages.append(page_data)

        if failed_names:
            logger.warning("%d개의 페이지 로딩에 실패했습니다: %s", len(failed_names), failed_names)
            self.callback(ProgressEvent(
                PipelinePhase.PASS2_PAGE, 0, 0,
                f"식자 단계 이미지 로딩 실패: {', '.join(failed_names[:3])}"
                f"{'…' if len(failed_names) > 3 else ''}",
                level=EventLevel.WARNING,
                extras={"failed_count": len(failed_names), "failed_pages": failed_names},
            ))

        return loaded_pages, missing + failed_names

    @staticmethod
    def _release_batch(loaded_pages: Sequence[PageData]) -> None:
        for page_data in loaded_pages:
            page_data.image_rgb = None
