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
        (detection_model, ocr_model, chat_session, lama_model,
         mtl_model, style_mapping) = models
    except Exception as e:
        print(f"초기화 실패: {e}")
        return

    # --- 2. 이미지 파일 목록 로드 및 배치 생성 ---
    image_paths = sorted(glob.glob(os.path.join(config.INPUT_DIR, "*")))
    if not image_paths:
        print(f"'{config.INPUT_DIR}' 폴더에 이미지가 없습니다.")
        return

    all_final_data = []
    batch_size = config.TRANSLATION_BATCH_SIZE

    extraction_models = (detection_model, ocr_model, chat_session, mtl_model, style_mapping)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"\n--- 처리 시작: 배치 {i // batch_size + 1} ({len(batch_paths)} 페이지) ---")

        batch_images_rgb = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in batch_paths if
                            cv2.imread(p) is not None]
        if not batch_images_rgb: continue

        # --- 3. 추출 및 번역 일괄 처리 ---
        final_batch_data = extractor.process_image_batch(extraction_models, batch_images_rgb, batch_paths)
        all_final_data.extend(final_batch_data)

    # --- 5. 최종 데이터 저장 ---
    if all_final_data:
        with open('translation_data.json', 'w', encoding='utf-8') as f:
            json.dump(all_final_data, f, ensure_ascii=False, indent=4)
        print("\n모든 페이지의 최종 데이터 구조를 'translation_data.json' 파일로 저장했습니다.")

    # --- 6. 식자 작업 실행 (문맥 인식 Crop & Patch 방식) ---
    if all_final_data:
        print("\n모든 페이지에 대한 식자 작업을 시작합니다...")
        output_dir = "data/outputs"
        os.makedirs(output_dir, exist_ok=True)

        patch_mask_list = []
        patch_metadata = []
        original_pages_rgb = [None] * len(all_final_data)

        for i, page_data in enumerate(all_final_data):
            source_page_path = os.path.join(config.INPUT_DIR, page_data['source_page'])
            image_bgr = cv2.imread(source_page_path)
            if image_bgr is None: continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            original_pages_rgb[i] = image_rgb
            img_h, img_w = image_rgb.shape[:2]

            # [핵심 수정 1] 페이지 전체에 대한 마스크를 먼저 생성
            all_boxes_coords = [b['text_box'] for b in page_data['speech_bubbles'] + page_data['freeform_texts']]
            full_page_mask = drawmanager.create_mask_from_coords(image_rgb, all_boxes_coords,
                                                                 padding=config.INPAINT_MASK_PADDING)

            for coords in all_boxes_coords:
                x1, y1, x2, y2 = map(int, coords)

                # 문맥을 포함하는 더 큰 영역(context_box) 계산
                pad = config.INPAINT_CONTEXT_PADDING
                ctx_x1, ctx_y1 = max(0, x1 - pad), max(0, y1 - pad)
                ctx_x2, ctx_y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)

                # 고해상도 문맥 조각 잘라내기
                context_patch = image_rgb[ctx_y1:ctx_y2, ctx_x1:ctx_x2]

                # [핵심 수정 2] 전체 마스크에서 동일한 위치를 잘라내어 patch_mask로 사용
                patch_mask = full_page_mask[ctx_y1:ctx_y2, ctx_x1:ctx_x2]

                patch_mask_list.append((context_patch, patch_mask))
                patch_metadata.append({'page_index': i, 'coords': (ctx_x1, ctx_y1, ctx_x2, ctx_y2)})

        # --- 2, 3, 4단계는 변경 없음 ---
        inpainted_patches = drawmanager.erase_patches_in_batch(lama_model, patch_mask_list)
        final_base_images = [img.copy() for img in original_pages_rgb if img is not None]

        for i, patch_meta in enumerate(patch_metadata):
            page_idx = patch_meta['page_index']
            if final_base_images[page_idx] is None: continue

            ctx_x1, ctx_y1, ctx_x2, ctx_y2 = patch_meta['coords']
            inpainted_patch = inpainted_patches[i]

            h, w = ctx_y2 - ctx_y1, ctx_x2 - ctx_x1
            if inpainted_patch.shape[0] != h or inpainted_patch.shape[1] != w:
                inpainted_patch = cv2.resize(inpainted_patch, (w, h), interpolation=cv2.INTER_LANCZOS4)

            final_base_images[page_idx][ctx_y1:ctx_y2, ctx_x1:ctx_x2] = inpainted_patch

        for page_data, final_base_image in zip(all_final_data, final_base_images):
            if final_base_image is None: continue

            source_page_name = page_data['source_page']
            print(f"[{source_page_name}] 식자 작업 중...")
            final_image_rgb = drawmanager.draw_translations(final_base_image, page_data)
            output_path = os.path.join(output_dir, source_page_name)
            final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, final_image_bgr)
            print(f" -> '{output_path}'에 최종 결과물 저장 완료.")

    print("\n모든 프로세스 완료.")

if __name__ == '__main__':
    main()