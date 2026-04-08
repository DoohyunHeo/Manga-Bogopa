import logging
import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from tqdm import tqdm
from src import config, model_loader, extractor, translator, inpainter, drawer
from src.data_models import PageData, TextElement, SpeechBubble

logger = logging.getLogger(__name__)


class MangaTranslationPipeline:
    def __init__(self):
        """파이프라인을 초기화하고 모든 모델을 로드합니다."""
        self.models = model_loader.load_all_models()
        self.output_dir = "data/outputs"

    def run(self):
        """전체 만화 번역 및 식자 프로세스를 실행합니다."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
        logger.info(f"Using device: {config.DEVICE}")
        os.makedirs(config.DEBUG_CROPS_DIR, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        image_paths = sorted(glob.glob(os.path.join(config.INPUT_DIR, "*")))
        if not image_paths:
            logger.info(f"'{config.INPUT_DIR}' 폴더에 이미지가 없습니다.")
            return

        # Pass 1: 탐지 + OCR + 번역 (텍스트 데이터 수집)
        all_page_data = self._extract_and_translate_data(image_paths)

        if not all_page_data:
            logger.info("처리할 데이터가 없어 파이프라인을 종료합니다.")
            return

        self._save_data_to_json(all_page_data)

        # Pass 2: 이미지 재로드 → 인페인팅 → 렌더링 → 저장 (페이지별 스트리밍)
        self._inpaint_and_draw_streaming(all_page_data, image_paths)

        logger.info("모든 프로세스 완료.")

    def _extract_and_translate_data(self, image_paths):
        """이미지에서 데이터를 추출하고 번역합니다."""
        all_page_data = []
        batch_size = config.TRANSLATION_BATCH_SIZE
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        for i in range(0, len(image_paths), batch_size):
            logger.info(f"--- Processing Batch {i // batch_size + 1}/{total_batches} ---")
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

            untranslated_page_data = extractor.process_image_batch(self.models, batch_images_rgb, valid_paths)
            translated_page_data = translator.translate_pages_in_batch(self.models['translator'], untranslated_page_data)

            # Pass 1 완료 후 이미지 데이터 해제 — 텍스트 메타데이터만 유지
            for page_data in translated_page_data:
                page_data.image_rgb = None
            all_page_data.extend(translated_page_data)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_page_data

    def _save_data_to_json(self, all_page_data):
        """추출된 데이터를 JSON 파일로 저장합니다."""
        class DataclassEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, PageData):
                    d = o.__dict__.copy()
                    d.pop('image_rgb', None)
                    return d
                if isinstance(o, (SpeechBubble, TextElement)):
                    return o.__dict__
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)

        with open('translation_data.json', 'w', encoding='utf-8') as f:
            json.dump(all_page_data, f, ensure_ascii=False, indent=4, cls=DataclassEncoder)
        logger.info("모든 페이지의 최종 데이터 구조를 'translation_data.json' 파일로 저장했습니다.")

    def _inpaint_and_draw_streaming(self, all_page_data, image_paths):
        """페이지별로 이미지를 재로드하여 인페인팅 + 렌더링 + 저장을 스트리밍합니다."""
        # source_page 이름으로 원본 경로를 매핑
        path_map = {os.path.basename(p): p for p in image_paths}

        logger.info("스트리밍 모드로 Inpainting + 식자 작업을 시작합니다...")
        for page_data in tqdm(all_page_data, desc="Inpaint & Draw"):
            original_path = path_map.get(page_data.source_page)
            if not original_path:
                logger.warning(f"'{page_data.source_page}'의 원본 경로를 찾을 수 없습니다.")
                continue

            # 이미지 재로드
            image_bgr = cv2.imread(original_path)
            if image_bgr is None:
                logger.warning(f"'{original_path}' 로딩 실패")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            page_data.image_rgb = image_rgb

            # 인페인팅
            inpainted_images = inpainter.inpaint_pages_in_batch(self.models, [page_data])

            # 텍스트 렌더링
            final_image_rgb = drawer.draw_text_on_image(inpainted_images[0], page_data)

            if config.DRAW_DEBUG_BOXES:
                self._draw_debug_boxes(final_image_rgb, page_data)

            # 저장
            output_path = os.path.join(self.output_dir, page_data.source_page)
            final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, final_image_bgr)

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
