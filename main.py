import cv2
import os
import shutil
import json
import glob
import numpy as np
from src import config, extractor, drawmanager


def main():
    # --- 초기화 ---
    print(f"Using device: {config.DEVICE}")
    if os.path.exists(config.DEBUG_CROPS_DIR):
        shutil.rmtree(config.DEBUG_CROPS_DIR)
    os.makedirs(config.DEBUG_CROPS_DIR, exist_ok=True)

    # --- 1. AI 모델 및 세션 준비 ---
    try:
        models = extractor.initialize_models_and_session()
        detection_model, ocr_model, chat_session, font_classifier_model, lama_model = models
    except Exception as e:
        print(f"초기화 실패: {e}")
        return

    # --- 2. 이미지 파일 목록 로드 및 배치 생성 ---
    image_paths = sorted(glob.glob(f"{config.INPUT_DIR}/*"))
    if not image_paths:
        print(f"'{config.INPUT_DIR}' 폴더에 이미지가 없습니다.")
        return

    all_final_data = []
    batch_size = config.TRANSLATION_BATCH_SIZE

    extraction_models = (detection_model, ocr_model, chat_session, font_classifier_model)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"\n--- 처리 시작: 배치 {i // batch_size + 1} ({len(batch_paths)} 페이지) ---")

        batch_images_rgb = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in batch_paths if
                            cv2.imread(p) is not None]
        if not batch_images_rgb: continue

        # --- 3. [변경] 추출 및 번역 일괄 처리 ---
        # 복잡한 로직을 extractor의 새 함수에 위임
        final_batch_data = extractor.process_image_batch(extraction_models, batch_images_rgb, batch_paths)
        all_final_data.extend(final_batch_data)

    # --- 5. 최종 데이터 저장 ---
    if all_final_data:
        with open('translation_data.json', 'w', encoding='utf-8') as f:
            json.dump(all_final_data, f, ensure_ascii=False, indent=4)
        print("\n모든 페이지의 최종 데이터 구조를 'translation_data.json' 파일로 저장했습니다.")

        # --- 6. 식자 작업 실행 ---
        if all_final_data:
            print("\n모든 페이지에 대한 식자 작업을 시작합니다...")

            output_dir = "data/outputs"
            os.makedirs(output_dir, exist_ok=True)

            # 1단계: Inpainting 할 데이터와 원본 고화질 이미지 수집
            inpainting_targets = []
            original_images_rgb = []  # <-- 원본 고화질 이미지를 담을 리스트
            masks = []  # <-- 생성된 마스크를 담을 리스트

            for page_data in all_final_data:
                source_page_name = page_data['source_page']
                source_page_path = os.path.join(config.INPUT_DIR, source_page_name)

                image_bgr = cv2.imread(source_page_path)
                if image_bgr is None: continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                original_images_rgb.append(image_rgb)  # 원본 이미지 저장

                detected_boxes = detection_model(image_rgb)[0].boxes
                text_mask = drawmanager.create_mask(image_rgb, detected_boxes, config.CLASSES_TO_ERASE,
                                                    detection_model.names, padding=config.INPAINT_MASK_PADDING)
                masks.append(text_mask)  # 마스크 저장

                inpainting_targets.append((image_rgb, text_mask))

            # 2단계: 배치 Inpainting 실행
            # 이 함수가 반환하는 inpainted_images_low_res는 저화질이지만, Inpainting된 부분의 정보만 사용할 것입니다.
            inpainted_images_low_res = drawmanager.erase_text_in_batch(lama_model, inpainting_targets)

            # [수정] 3단계: 원본 고화질 이미지에 Inpainting 결과 합성
            final_base_images = []
            for original_image, inpainted_image, mask in zip(original_images_rgb, inpainted_images_low_res, masks):
                # 마스크를 3채널로 확장 (RGB 이미지와 합성을 위해)
                mask_3d = np.stack([mask] * 3, axis=-1) > 0

                # np.where을 사용하여 마스크 영역은 inpainted_image에서, 나머지는 original_image에서 가져옴
                final_base_image = np.where(mask_3d, inpainted_image, original_image)
                final_base_images.append(final_base_image)

            # [수정] 4단계: 합성된 고화질 이미지에 식자 작업
            for page_data, final_base_image in zip(all_final_data, final_base_images):
                source_page_name = page_data['source_page']
                print(f"[{source_page_name}] 식자 작업 중...")

                # 고화질 베이스 이미지에 번역 텍스트를 그립니다.
                final_image_rgb = drawmanager.draw_translations(final_base_image, page_data)

                output_path = os.path.join(output_dir, source_page_name)
                final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, final_image_bgr)
                print(f" -> '{output_path}'에 최종 결과물 저장 완료.")

        print("\n모든 프로세스 완료.")


if __name__ == '__main__':
    main()