import cv2
import os
import shutil
import json
import glob
from src import config, extractor, drawmanager


def main():
    # --- 초기화 ---
    print(f"Using device: {config.DEVICE}")
    if os.path.exists(config.DEBUG_CROPS_DIR):
        shutil.rmtree(config.DEBUG_CROPS_DIR)
    os.makedirs(config.DEBUG_CROPS_DIR, exist_ok=True)

    # --- 1. AI 모델 및 세션 준비 ---
    try:
        # models 튜플은 이제 4개의 요소를 가짐
        models = extractor.initialize_models_and_session()
        # [변경] 4개의 반환값을 모두 받도록 수정
        detection_model, ocr_model, chat_session, font_classifier_model = models
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

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"\n--- 처리 시작: 배치 {i // batch_size + 1} ({len(batch_paths)} 페이지) ---")

        batch_images_rgb = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in batch_paths if
                            cv2.imread(p) is not None]
        if not batch_images_rgb: continue

        print(f"  -> {len(batch_images_rgb)}개 페이지 일괄 탐지 중...")
        batch_results = detection_model(batch_images_rgb)

        # --- 3. 페이지별 데이터 구조화 ---
        untranslated_batch_data = []
        for (path, image_rgb, results) in zip(batch_paths, batch_images_rgb, batch_results):
            page_identifier = os.path.basename(path)
            page_data = extractor.structure_page_data(models, image_rgb, page_identifier, results)
            untranslated_batch_data.append(page_data)

        if not untranslated_batch_data: continue

        # --- 4. 배치 단위 번역 실행 ---
        final_batch_data = extractor.translate_image_batch(chat_session, untranslated_batch_data)
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

        for page_data in all_final_data:
            source_page_name = page_data['source_page']
            source_page_path = os.path.join(config.INPUT_DIR, source_page_name)

            print(f"[{source_page_name}] 식자 작업 중...")

            image_bgr = cv2.imread(source_page_path)
            if image_bgr is None: continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            detected_boxes = detection_model(image_rgb)[0].boxes
            text_mask = drawmanager.create_mask(image_rgb, detected_boxes, config.CLASSES_TO_ERASE,
                                                detection_model.names, padding=config.INPAINT_MASK_PADDING)
            inpainted_image = drawmanager.erase_text_with_lama(image_rgb, text_mask)

            final_image_rgb = drawmanager.draw_translations(inpainted_image, page_data)

            output_path = os.path.join(output_dir, source_page_name)
            final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, final_image_bgr)
            print(f" -> '{output_path}'에 최종 결과물 저장 완료.")

    print("\n모든 프로세스 완료.")


if __name__ == '__main__':
    main()