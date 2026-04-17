import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np
import torch

from src import config, inpainter, page_drawer
from src.data_models import PageData
from src.progress import EventLevel, PipelinePhase, ProgressCallback, ProgressEvent, noop_callback

logger = logging.getLogger(__name__)


class Pass2Stage:
    """Pass 2 execution helper for microbatched inpainting and rendering."""

    def __init__(
        self,
        models,
        output_dir: str,
        progress_callback: Optional[ProgressCallback] = None,
        debug_draw_boxes: bool = False,
        debug_box_drawer: Optional[Callable[[np.ndarray, PageData], None]] = None,
        ensure_inpainting_model: Optional[Callable[[], None]] = None,
    ):
        self.models = models
        self.output_dir = output_dir
        self.callback = progress_callback or noop_callback
        self.debug_draw_boxes = debug_draw_boxes
        self.debug_box_drawer = debug_box_drawer
        self.ensure_inpainting_model = ensure_inpainting_model

    def run(self, pages_to_process: Sequence[PageData], image_paths: Sequence[str], ckpt=None) -> None:
        if not pages_to_process:
            return

        path_map = {os.path.basename(path): path for path in image_paths}
        processable_pages = [page for page in pages_to_process if page.source_page in path_map]
        if not processable_pages:
            logger.warning("Pass 2에서 처리 가능한 페이지가 없습니다.")
            return

        if self.ensure_inpainting_model:
            self.ensure_inpainting_model()

        microbatch_size = max(1, int(config.PASS2_MICROBATCH_SIZE))
        total_pages = len(processable_pages)
        total_batches = (total_pages + microbatch_size - 1) // microbatch_size
        completed_pages = 0

        logger.info(
            "Pass 2 microbatch mode: %d pages, batch size %d",
            total_pages,
            microbatch_size,
        )

        for batch_index, start in enumerate(range(0, total_pages, microbatch_size), start=1):
            batch_pages = processable_pages[start:start + microbatch_size]
            logger.info("Pass 2 batch %d/%d: %d pages", batch_index, total_batches, len(batch_pages))
            loaded_pages = self._load_page_batch(batch_pages, path_map)
            if not loaded_pages:
                self._maybe_empty_cache(batch_index)
                continue

            inpainted_images = inpainter.inpaint_pages_in_batch(self.models, loaded_pages)

            for page_data, inpainted_image in zip(loaded_pages, inpainted_images):
                page_started_at = time.perf_counter()
                final_image_rgb = page_drawer.draw_text_on_image(inpainted_image, page_data)

                if self.debug_draw_boxes and self.debug_box_drawer:
                    self.debug_box_drawer(final_image_rgb, page_data)

                output_path = os.path.join(self.output_dir, page_data.source_page)
                final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, final_image_bgr)

                completed_pages += 1
                page_elapsed = time.perf_counter() - page_started_at
                self.callback(
                    ProgressEvent(
                        PipelinePhase.PASS2_PAGE,
                        completed_pages,
                        total_pages,
                        f"{page_data.source_page} 완료 ({page_elapsed:.1f}초)",
                        page_name=page_data.source_page,
                        image_rgb=final_image_rgb,
                        elapsed_sec=page_elapsed,
                        extras={
                            "bubbles": len(page_data.speech_bubbles),
                            "freeform": len(page_data.freeform_texts),
                        },
                    )
                )

                if ckpt:
                    ckpt.mark_pass2_page_complete(page_data.source_page)

                page_data.image_rgb = None

            self._maybe_empty_cache(batch_index)
            self._release_batch(loaded_pages)

    def _load_page_batch(self, batch_pages: Sequence[PageData], path_map) -> List[PageData]:
        batch_entries = []
        for page_data in batch_pages:
            original_path = path_map.get(page_data.source_page)
            if not original_path:
                logger.warning("'%s'의 원본 경로를 찾을 수 없습니다.", page_data.source_page)
                continue
            batch_entries.append((page_data, original_path))

        if not batch_entries:
            return []

        worker_count = min(max(1, int(config.PASS2_IMAGE_LOAD_WORKERS)), len(batch_entries))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            images_bgr = list(executor.map(cv2.imread, [path for _, path in batch_entries]))

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

        return loaded_pages

    def _maybe_empty_cache(self, batch_index: int) -> None:
        interval = int(getattr(config, "PASS2_EMPTY_CACHE_EVERY_N_BATCHES", 0))
        if interval <= 0:
            return
        if batch_index % interval != 0:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _release_batch(loaded_pages: Sequence[PageData]) -> None:
        for page_data in loaded_pages:
            page_data.image_rgb = None
