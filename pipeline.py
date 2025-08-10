import os
import glob
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from src import config, model_loader, extractor, translator, inpainter, drawer
from src.data_models import PageData, TextElement, SpeechBubble # Import dataclasses

class MangaTranslationPipeline:
    def __init__(self):
        """파이프라인을 초기화하고 모든 모델을 로드합니다."""
        self.models = model_loader.load_all_models()

    

    def run(self):
        """전체 만화 번역 및 식자 프로세스를 실행합니다."""
        print(f"Using device: {config.DEVICE}")
        os.makedirs(config.DEBUG_CROPS_DIR, exist_ok=True)
        output_dir = "data/outputs"
        os.makedirs(output_dir, exist_ok=True)

        image_paths = sorted(glob.glob(os.path.join(config.INPUT_DIR, "*")))
        if not image_paths:
            print(f"'{config.INPUT_DIR}' 폴더에 이미지가 없습니다.")
            return

        all_page_data = []
        batch_size = config.TRANSLATION_BATCH_SIZE

        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        for i in range(0, len(image_paths), batch_size):
            tqdm.write(f"\n--- Processing Batch {i // batch_size + 1}/{total_batches} ---")
            batch_paths = image_paths[i:i + batch_size]
            
            batch_images_bgr = [cv2.imread(p) for p in batch_paths]
            batch_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch_images_bgr if img is not None]
            valid_paths = [p for p, img in zip(batch_paths, batch_images_bgr) if img is not None]
            if not batch_images_rgb: continue

            # 1. 정보 추출
            untranslated_page_data = extractor.process_image_batch(self.models, batch_images_rgb, valid_paths)

            # 2. 번역
            translated_page_data = translator.translate_pages_in_batch(self.models['translator'], untranslated_page_data)
            all_page_data.extend(translated_page_data)

            # 배치 처리 후 GPU 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 3. 최종 데이터 저장 (JSON)
        if all_page_data:
            class DataclassEncoder(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o, PageData):
                        d = o.__dict__.copy()
                        del d['image_rgb']
                        return d
                    if isinstance(o, (SpeechBubble, TextElement)):
                        return o.__dict__
                    if isinstance(o, np.ndarray):
                        return o.tolist()
                    return super().default(o)

            with open('translation_data.json', 'w', encoding='utf-8') as f:
                json.dump(all_page_data, f, ensure_ascii=False, indent=4, cls=DataclassEncoder)
            tqdm.write("\n모든 페이지의 최종 데이터 구조를 'translation_data.json' 파일로 저장했습니다.")

        # 4. 식자 작업 (Inpainting & Drawing)
        if all_page_data:
            tqdm.write("\n모든 페이지에 대한 Inpainting 작업을 시작합니다...")
            # 4a. 모든 페이지의 텍스트를 일괄적으로 제거 (Batch Inpainting)
            inpainted_images = inpainter.inpaint_pages_in_batch(self.models, all_page_data)

            tqdm.write("\n모든 페이지에 대한 식자 작업을 시작합니다...")
            for i, page_data in enumerate(tqdm(all_page_data, desc="Drawing Texts")):
                # 4b. 번역된 텍스트 그리기
                final_image_rgb = drawer.draw_text_on_image(inpainted_images[i], page_data)
                
                # 4c. 결과 저장
                output_path = os.path.join(output_dir, page_data.source_page)
                final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, final_image_bgr)

        print("\n모든 프로세스 완료.")
