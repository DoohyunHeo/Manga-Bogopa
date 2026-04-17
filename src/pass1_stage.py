import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

import cv2
import torch

from src import config, extractor, model_loader, translator
from src.data_models import PageData
from src.progress import EventLevel, PipelinePhase, ProgressCallback, ProgressEvent, noop_callback
from src.serialization import save_page_data_json

logger = logging.getLogger(__name__)


class Pass1Stage:
    """Pass 1 orchestration for load, detect, OCR, translate, and checkpoint reuse."""

    def __init__(self, models, output_dir: str, progress_callback: Optional[ProgressCallback] = None):
        self.models = models
        self.output_dir = output_dir
        self.callback = progress_callback or noop_callback

    def run(self, image_paths: Sequence[str], ckpt=None, force_fresh: bool = False) -> List[PageData]:
        all_page_data = self._prepare_pass1_data(image_paths, ckpt=ckpt, force_fresh=force_fresh)
        if not all_page_data:
            return []

        json_path = os.path.join(self.output_dir, "translation_data.json")
        if ckpt:
            ckpt.replace_pass1_data(all_page_data, complete=True)
        else:
            save_page_data_json(all_page_data, json_path)
        self.callback(ProgressEvent(PipelinePhase.SAVING_JSON, 1, 1, "JSON 저장 완료"))
        logger.info(f"번역 데이터를 '{json_path}' 파일로 저장했습니다.")
        return all_page_data

    def has_full_output_image_set(self, image_paths: Sequence[str]) -> bool:
        """Check whether output images already exist for every input image."""
        if not os.path.isdir(self.output_dir):
            return False

        input_names = {os.path.basename(path) for path in image_paths}
        if not input_names:
            return False

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        output_names = {
            name for name in os.listdir(self.output_dir)
            if os.path.splitext(name)[1].lower() in image_exts
        }
        return input_names.issubset(output_names)

    def _has_font_models(self) -> bool:
        return any([
            self.models.get("font_appearance_classifier"),
            self.models.get("font_size_regressor"),
            self.models.get("font_classifier"),
        ])

    def _reload_font_models(self):
        """Always refresh font models so newly trained checkpoints are picked up without a full app restart."""
        for key in ("font_appearance_classifier", "font_size_regressor", "font_classifier"):
            self.models.pop(key, None)
        self.models.update(model_loader.load_font_model())

    def _ensure_pass1_models(self):
        """Load only the models required for Pass 1."""
        needs_detection_ocr = "detection" not in self.models or "ocr" not in self.models
        needs_font = True
        needs_translator = "translator" not in self.models
        if not needs_detection_ocr and not needs_font and not needs_translator:
            return

        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 0, 1, "Pass 1 모델 로딩 중..."))
        if needs_detection_ocr:
            self.models.update(model_loader.load_detection_ocr_models())
        if needs_font:
            self._reload_font_models()
        if needs_translator:
            self.models["translator"] = model_loader.load_translator_session()
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 1, 1, "Pass 1 모델 로딩 완료"))

    def _prepare_pass1_data(self, image_paths, ckpt=None, force_fresh=False):
        """Reuse checkpoints when possible and run Pass 1 only for missing pages."""
        if force_fresh:
            self._ensure_pass1_models()
            return self._extract_and_translate_data(image_paths, ckpt)

        if not ckpt:
            self._ensure_pass1_models()
            return self._extract_and_translate_data(image_paths, None)

        existing_page_data = ckpt.load_pass1_data()
        if existing_page_data:
            existing_page_data = self._sort_page_data_by_input_order(existing_page_data, image_paths)
            completed_pages, incomplete_pages = self._split_translated_pages(existing_page_data)
            expected_pages = {os.path.basename(path) for path in image_paths}
            completed_names = {page.source_page for page in completed_pages}

            if completed_names == expected_pages and not incomplete_pages:
                logger.info(f"완전한 체크포인트 JSON을 재사용합니다: {len(completed_pages)}페이지")
                ckpt.replace_pass1_data(completed_pages, complete=True)
                return completed_pages

            if completed_pages or incomplete_pages:
                logger.info(
                    f"Pass 1 체크포인트 재개: 완료 {len(completed_pages)}페이지, 재처리 {len(incomplete_pages)}페이지"
                )
                ckpt.replace_pass1_data(completed_pages, complete=False)
        else:
            completed_pages = []

        completed_names = {page.source_page for page in completed_pages}
        remaining_paths = [
            path for path in image_paths
            if os.path.basename(path) not in completed_names
        ]

        if not remaining_paths:
            return completed_pages

        self._ensure_pass1_models()
        new_page_data = self._extract_and_translate_data(remaining_paths, ckpt)
        return self._sort_page_data_by_input_order(completed_pages + new_page_data, image_paths)

    @staticmethod
    def _is_text_translated(text: str) -> bool:
        if text is None:
            return False
        normalized = text.strip()
        return bool(normalized) and normalized != "번역 불가"

    def _is_page_translated(self, page_data: PageData) -> bool:
        for bubble in page_data.speech_bubbles:
            if not self._is_text_translated(bubble.text_element.translated_text):
                return False
        for freeform_text in page_data.freeform_texts:
            if not self._is_text_translated(freeform_text.translated_text):
                return False
        return True

    def _split_translated_pages(self, page_data_list):
        completed = []
        incomplete = []
        for page_data in page_data_list:
            if self._is_page_translated(page_data):
                completed.append(page_data)
            else:
                incomplete.append(page_data)
        return completed, incomplete

    @staticmethod
    def _sort_page_data_by_input_order(page_data_list, image_paths):
        ordered = {page_data.source_page: page_data for page_data in page_data_list}
        return [
            ordered[os.path.basename(path)]
            for path in image_paths
            if os.path.basename(path) in ordered
        ]

    def _extract_and_translate_data(self, image_paths, ckpt=None):
        """Extract data from images and translate it."""
        all_page_data = []
        batch_size = config.TRANSLATION_BATCH_SIZE
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        for i in range(0, len(image_paths), batch_size):
            batch_idx = i // batch_size + 1
            logger.info(f"--- Processing Batch {batch_idx}/{total_batches} ---")
            batch_paths = image_paths[i:i + batch_size]
            batch_started_at = time.perf_counter()

            worker_count = min(max(1, int(config.PASS1_IMAGE_LOAD_WORKERS)), len(batch_paths))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                batch_images_bgr = list(executor.map(cv2.imread, batch_paths))

            failed_count = sum(1 for img in batch_images_bgr if img is None)
            if failed_count > 0:
                failed_names = [
                    os.path.basename(p) for p, img in zip(batch_paths, batch_images_bgr) if img is None
                ]
                logger.warning(f"{failed_count}개 이미지 로딩 실패: {failed_names}")
                self.callback(ProgressEvent(
                    PipelinePhase.PASS1_BATCH, batch_idx, total_batches,
                    f"배치 {batch_idx}: {failed_count}개 이미지 로딩 실패 ({', '.join(failed_names[:3])}"
                    f"{'…' if failed_count > 3 else ''})",
                    level=EventLevel.WARNING,
                    extras={"failed_count": failed_count, "failed_pages": failed_names},
                ))

            batch_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch_images_bgr if img is not None]
            valid_paths = [p for p, img in zip(batch_paths, batch_images_bgr) if img is not None]
            if not batch_images_rgb:
                continue

            num_pages = len(batch_images_rgb)
            detection_started_at = time.perf_counter()
            self.callback(ProgressEvent(
                PipelinePhase.DETECTION, batch_idx, total_batches,
                f"배치 {batch_idx}/{total_batches}: {num_pages}페이지 탐지 중...",
                extras={"pages": num_pages},
            ))
            all_text_items, all_bubbles_by_page = extractor.detect_objects(self.models["detection"], batch_images_rgb)
            raw_box_count = len(all_text_items)
            bubble_count = sum(len(page_bubbles) for page_bubbles in all_bubbles_by_page)
            merged_text_items = extractor.merge_text_boxes(all_text_items)
            merged_box_count = len(merged_text_items)
            detection_elapsed = time.perf_counter() - detection_started_at
            self.callback(ProgressEvent(
                PipelinePhase.DETECTION, batch_idx, total_batches,
                f"배치 {batch_idx}: 말풍선 {bubble_count} / 글자 박스 {raw_box_count}→{merged_box_count} 병합",
                elapsed_sec=detection_elapsed,
                extras={
                    "pages": num_pages,
                    "bubbles": bubble_count,
                    "raw_boxes": raw_box_count,
                    "merged_boxes": merged_box_count,
                },
            ))

            ocr_started_at = time.perf_counter()
            self.callback(ProgressEvent(
                PipelinePhase.OCR, batch_idx, total_batches,
                f"배치 {batch_idx}: {merged_box_count}개 글자 조각 인식 + 폰트 분석 중...",
                extras={"target": merged_box_count},
            ))
            processed_text_elements = extractor.extract_text_properties(
                self.models,
                batch_images_rgb,
                merged_text_items,
                valid_paths,
            )
            kept_count = len(processed_text_elements)
            filtered_count = max(0, merged_box_count - kept_count)
            ocr_elapsed = time.perf_counter() - ocr_started_at
            self.callback(ProgressEvent(
                PipelinePhase.OCR, batch_idx, total_batches,
                f"배치 {batch_idx}: 인식 {kept_count}개 / 품질 필터 {filtered_count}개 제외",
                elapsed_sec=ocr_elapsed,
                level=EventLevel.WARNING if filtered_count > 0 and kept_count == 0 else EventLevel.INFO,
                extras={"kept": kept_count, "filtered": filtered_count, "target": merged_box_count},
            ))

            untranslated_page_data = extractor.structure_page_data(
                valid_paths,
                batch_images_rgb,
                all_bubbles_by_page,
                processed_text_elements,
            )

            total_texts = sum(len(p.speech_bubbles) + len(p.freeform_texts) for p in untranslated_page_data)
            translation_started_at = time.perf_counter()
            self.callback(ProgressEvent(
                PipelinePhase.TRANSLATION, batch_idx, total_batches,
                f"배치 {batch_idx}: {total_texts}개 대사 번역 요청 중...",
                extras={"texts": total_texts},
            ))
            translated_page_data = translator.translate_pages_in_batch(
                self.models["translator"],
                untranslated_page_data,
                callback=self.callback,
            )
            translation_elapsed = time.perf_counter() - translation_started_at

            for page_data in translated_page_data:
                page_data.image_rgb = None

            if ckpt:
                ckpt.mark_pass1_batch_complete(translated_page_data)
            all_page_data.extend(translated_page_data)

            batch_elapsed = time.perf_counter() - batch_started_at
            self.callback(ProgressEvent(
                PipelinePhase.PASS1_BATCH, batch_idx, total_batches,
                f"배치 {batch_idx}/{total_batches} 완료 ({len(valid_paths)}페이지, {batch_elapsed:.1f}초)",
                elapsed_sec=batch_elapsed,
                extras={
                    "pages": len(valid_paths),
                    "detection_sec": round(detection_elapsed, 2),
                    "ocr_sec": round(ocr_elapsed, 2),
                    "translation_sec": round(translation_elapsed, 2),
                },
            ))

            self._maybe_empty_cache_after_pass1_batch(batch_idx)

        return all_page_data

    def _maybe_empty_cache_after_pass1_batch(self, batch_idx: int) -> None:
        """Conditionally clear CUDA cache after a Pass 1 batch."""
        interval = int(getattr(config, "PASS1_EMPTY_CACHE_EVERY_N_BATCHES", 0))
        if interval <= 0:
            return
        if batch_idx % interval != 0:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
