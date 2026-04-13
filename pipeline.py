import logging
import os
import glob

import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src import config, model_loader, extractor, translator, inpainter, drawer
from src.extractor import detect_objects, merge_text_boxes, extract_text_properties, structure_page_data
from src.data_models import PageData
from src.progress import ProgressEvent, PipelinePhase, ProgressCallback, noop_callback
from src.serialization import save_page_data_json, load_page_data_json
from src.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class MangaTranslationPipeline:
    def __init__(self, progress_callback: ProgressCallback = None, enable_checkpoint: bool = True):
        """파이프라인을 초기화합니다. 모델은 실행 시점에 지연 로드합니다."""
        self.callback = progress_callback or noop_callback
        self.enable_checkpoint = enable_checkpoint
        self.output_dir = config.OUTPUT_DIR
        self.models = {}

    def run(self):
        """전체 만화 번역 및 식자 프로세스를 실행합니다."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
        logger.info(f"Using device: {config.DEVICE}")
        os.makedirs(self.output_dir, exist_ok=True)

        image_paths = sorted(glob.glob(os.path.join(config.INPUT_DIR, "*")))
        if not image_paths:
            logger.info(f"'{config.INPUT_DIR}' 폴더에 이미지가 없습니다.")
            return

        ckpt = None
        force_fresh_pass1 = self._has_full_output_image_set(image_paths)
        if self.enable_checkpoint:
            ckpt = CheckpointManager(self.output_dir, config.INPUT_DIR)
            ckpt.load_or_create(len(image_paths))
            if force_fresh_pass1:
                logger.info(
                    "출력 폴더에 입력 전체와 동일한 결과 이미지가 이미 있어 "
                    "체크포인트를 재사용하지 않고 새 번역을 시작합니다."
                )
                ckpt.reset_for_new_run(clear_json=True)

        # --- Pass 1: 탐지 + OCR + 번역 ---
        all_page_data = self._prepare_pass1_data(
            image_paths,
            ckpt,
            force_fresh=force_fresh_pass1,
        )

        if not all_page_data:
            logger.info("처리할 데이터가 없어 파이프라인을 종료합니다.")
            return

        json_path = os.path.join(self.output_dir, "translation_data.json")
        if ckpt:
            ckpt.replace_pass1_data(all_page_data, complete=True)
        else:
            save_page_data_json(all_page_data, json_path)
        self.callback(ProgressEvent(PipelinePhase.SAVING_JSON, 1, 1, "JSON 저장 완료"))
        logger.info(f"번역 데이터를 '{json_path}' 파일로 저장했습니다.")

        # --- Pass 2: 인페인팅 + 렌더링 ---
        pages_to_process = ckpt.get_pass2_remaining_pages(all_page_data) if ckpt else all_page_data
        if pages_to_process:
            self._ensure_inpainting_model()
            self._inpaint_and_draw_streaming(pages_to_process, image_paths, ckpt)
        else:
            logger.info("Pass 2 체크포인트가 이미 완료되어 렌더링을 건너뜁니다.")

        if ckpt:
            ckpt.mark_complete()

        self.callback(ProgressEvent(PipelinePhase.COMPLETE, 1, 1, "모든 프로세스 완료"))
        logger.info("모든 프로세스 완료.")

    def _ensure_pass1_models(self):
        """탐지, OCR, 폰트 분석, 번역에 필요한 모델만 로드합니다."""
        missing = {
            "detection",
            "ocr",
            "font_classifier",
            "translator",
        } - set(self.models.keys())
        if not missing:
            return

        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 0, 1, "Pass 1 모델 로딩 중..."))
        if "detection" in missing or "ocr" in missing:
            self.models.update(model_loader.load_detection_ocr_models())
        if "font_classifier" in missing:
            self.models.update(model_loader.load_font_model())
        if "translator" in missing:
            self.models["translator"] = model_loader.load_translator_session()
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 1, 1, "Pass 1 모델 로딩 완료"))

    def _ensure_inpainting_model(self):
        """Pass 2 렌더링에 필요한 Inpainting 모델만 로드합니다."""
        if "inpainting" in self.models:
            return
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 0, 1, "Inpainting 모델 로딩 중..."))
        self.models.update(model_loader.load_inpainting_model())
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 1, 1, "Inpainting 모델 로딩 완료"))

    def _prepare_pass1_data(self, image_paths, ckpt=None, force_fresh=False):
        """체크포인트를 재사용하거나 필요한 페이지만 Pass 1을 수행합니다."""
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
                logger.info(
                    f"완전한 체크포인트 JSON을 재사용합니다: {len(completed_pages)}페이지"
                )
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

    def _has_full_output_image_set(self, image_paths):
        """출력 폴더에 입력 전체와 동일한 이름의 결과 이미지가 이미 있는지 확인합니다."""
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

    def _extract_and_translate_data(self, image_paths, ckpt=None):
        """이미지에서 데이터를 추출하고 번역합니다."""
        all_page_data = []
        batch_size = config.TRANSLATION_BATCH_SIZE
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        for i in range(0, len(image_paths), batch_size):
            batch_idx = i // batch_size + 1
            logger.info(f"--- Processing Batch {batch_idx}/{total_batches} ---")
            batch_paths = image_paths[i:i + batch_size]

            with ThreadPoolExecutor(max_workers=4) as executor:
                batch_images_bgr = list(executor.map(cv2.imread, batch_paths))

            failed_count = sum(1 for img in batch_images_bgr if img is None)
            if failed_count > 0:
                logger.warning(f"{failed_count}개 이미지 로딩 실패")

            batch_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch_images_bgr if img is not None]
            valid_paths = [p for p, img in zip(batch_paths, batch_images_bgr) if img is not None]
            if not batch_images_rgb:
                continue

            # 1. 탐지
            num_pages = len(batch_images_rgb)
            self.callback(ProgressEvent(
                PipelinePhase.DETECTION, batch_idx, total_batches,
                f"{num_pages}페이지 말풍선·텍스트 탐지 중..."
            ))
            all_text_items, all_bubbles_by_page = detect_objects(self.models['detection'], batch_images_rgb)
            merged_text_items = merge_text_boxes(all_text_items)

            # 2. OCR + 폰트분석
            self.callback(ProgressEvent(
                PipelinePhase.OCR, batch_idx, total_batches,
                f"{len(merged_text_items)}개 텍스트 OCR + 폰트 분석 중..."
            ))
            processed_text_elements = extract_text_properties(self.models, batch_images_rgb, merged_text_items, valid_paths)

            # 3. 데이터 구조화
            untranslated_page_data = structure_page_data(valid_paths, batch_images_rgb, all_bubbles_by_page, processed_text_elements)

            # 4. 번역
            total_texts = sum(len(p.speech_bubbles) + len(p.freeform_texts) for p in untranslated_page_data)
            self.callback(ProgressEvent(
                PipelinePhase.TRANSLATION, batch_idx, total_batches,
                f"{total_texts}개 대사 Gemini에 전송 중..."
            ))
            translated_page_data = translator.translate_pages_in_batch(
                self.models['translator'], untranslated_page_data, callback=self.callback
            )

            # 이미지 데이터 해제
            for page_data in translated_page_data:
                page_data.image_rgb = None

            # 체크포인트: 배치별 증분 저장
            if ckpt:
                ckpt.mark_pass1_batch_complete(translated_page_data)
            all_page_data.extend(translated_page_data)

            self.callback(ProgressEvent(
                PipelinePhase.PASS1_BATCH, batch_idx, total_batches,
                f"배치 {batch_idx}/{total_batches} 완료 ({len(valid_paths)}페이지)"
            ))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_page_data

    def _inpaint_and_draw_streaming(self, pages_to_process, image_paths, ckpt=None):
        """페이지별로 이미지를 재로드하여 인페인팅 + 렌더링 + 저장을 스트리밍합니다."""
        path_map = {os.path.basename(p): p for p in image_paths}
        total_pages = len(pages_to_process)

        logger.info(f"스트리밍 모드로 {total_pages}페이지 Inpainting + 식자 작업을 시작합니다...")
        for idx, page_data in enumerate(tqdm(pages_to_process, desc="Inpaint & Draw")):
            original_path = path_map.get(page_data.source_page)
            if not original_path:
                logger.warning(f"'{page_data.source_page}'의 원본 경로를 찾을 수 없습니다.")
                continue

            image_bgr = cv2.imread(original_path)
            if image_bgr is None:
                logger.warning(f"'{original_path}' 로딩 실패")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            page_data.image_rgb = image_rgb

            inpainted_images = inpainter.inpaint_pages_in_batch(self.models, [page_data])
            final_image_rgb = drawer.draw_text_on_image(inpainted_images[0], page_data)

            if config.DRAW_DEBUG_BOXES:
                self._draw_debug_boxes(final_image_rgb, page_data)

            output_path = os.path.join(self.output_dir, page_data.source_page)
            final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, final_image_bgr)

            # 진행 콜백 (완성된 이미지 포함)
            self.callback(ProgressEvent(
                PipelinePhase.PASS2_PAGE, idx + 1, total_pages,
                f"{page_data.source_page} 완료",
                page_name=page_data.source_page,
                image_rgb=final_image_rgb
            ))

            if ckpt:
                ckpt.mark_pass2_page_complete(page_data.source_page)

            # 메모리 해제
            page_data.image_rgb = None
            del image_bgr, image_rgb, inpainted_images, final_image_rgb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _draw_debug_boxes(self, image_rgb, page_data):
        """디버깅 목적으로 탐지된 모든 박스를 이미지에 그립니다."""
        COLOR_BUBBLE = (255, 0, 0)
        COLOR_TEXT = (0, 255, 0)
        COLOR_FREE_TEXT = (0, 0, 255)

        for bubble in page_data.speech_bubbles:
            x1, y1, x2, y2 = map(int, bubble.bubble_box)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), COLOR_BUBBLE, 2)
            tx1, ty1, tx2, ty2 = map(int, bubble.text_element.text_box)
            cv2.rectangle(image_rgb, (tx1, ty1), (tx2, ty2), COLOR_TEXT, 2)

        for ff_text in page_data.freeform_texts:
            tx1, ty1, tx2, ty2 = map(int, ff_text.text_box)
            cv2.rectangle(image_rgb, (tx1, ty1), (tx2, ty2), COLOR_FREE_TEXT, 2)
