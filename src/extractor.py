import os
import re
import cv2
import pytesseract
import google.generativeai as genai
import numpy as np
import tensorflow as tf
from simple_lama_inpainting import SimpleLama
from ultralytics import YOLO
from PIL import Image
from src import config
from src.batch_manga_ocr import BatchMangaOcr
import concurrent.futures

FONT_CLASS_NAMES = ['bold', 'confused', 'handwriting', 'narration', 'sad', 'shouting', 'standard']


def initialize_models_and_session():
    """AI 모델들을 로드하고 제미나이 챗 세션을 초기화합니다."""
    print("AI 모델 및 챗 세션을 초기화합니다...")
    try:
        with open(config.API_KEY_FILE, 'r') as f:
            api_key = f.read().strip()
        genai.configure(api_key=api_key)
    except FileNotFoundError:
        raise FileNotFoundError(f"'{config.API_KEY_FILE}'을 찾을 수 없습니다.")

    with open(config.SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    detection_model = YOLO(config.MODEL_PATH)
    detection_model.to(config.DEVICE)
    ocr_model = BatchMangaOcr()

    try:
        font_classifier_model = tf.keras.models.load_model(config.FONT_CLASSIFIER_PATH)
        print("폰트 분류 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"경고: 폰트 분류 모델 로드 실패. 기본 스타일만 사용됩니다. -> {e}")
        font_classifier_model = None

    # LaMa 모델을 이곳에서 한 번만 초기화합니다.
    lama_model = SimpleLama(device=config.DEVICE)
    print("SimpleLama Inpainting 모델을 성공적으로 로드했습니다.")

    translation_model = genai.GenerativeModel(config.GEMINI_MODEL)
    chat_session = translation_model.start_chat(history=[
        {'role': 'user', 'parts': [system_prompt]},
        {'role': 'model', 'parts': ["네, 알겠습니다. 이제부터 지시에 따라 번역을 시작하겠습니다."]},
    ])

    print("초기화 완료.")
    # 초기화된 lama_model을 함께 반환합니다.
    return detection_model, ocr_model, chat_session, font_classifier_model, lama_model


def _is_kana(char):
    """주어진 문자가 히라가나 또는 가타카나인지 확인합니다."""
    if len(char) != 1:
        return False
    return 0x3040 <= ord(char) <= 0x309F or 0x30A0 <= ord(char) <= 0x30FF


def _calculate_font_size(cropped_pil_image):
    """Tesseract로 측정한 문자 픽셀 높이를 기반으로 폰트 크기를 직접 계산합니다."""
    try:
        img = np.array(cropped_pil_image)
        h, w = img.shape[:2]

        lang, psm = ('jpn', '7') if w > h else ('jpn_vert', '5')
        config_str = f'--psm {psm}'

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        scale = 2 if w < 100 or h < 100 else 1
        if scale > 1:
            img_gray = cv2.resize(img_gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        boxes_data = pytesseract.image_to_boxes(img_gray, lang=lang, config=config_str)

        lines = boxes_data.strip().splitlines()
        if not lines: return 0

        char_heights = [abs(int(line.split()[4]) - int(line.split()[2])) for line in lines if
                        len(line.split()) == 6 and _is_kana(line.split()[0])]
        if not char_heights: return 0

        avg_box_height_px = (sum(char_heights) / len(char_heights)) / scale
        return avg_box_height_px * config.FONT_SCALE_FACTOR

    except Exception as e:
        print(f"    -> Tesseract 오류: {e}")
        return 0


def check_bubble_attachment(cropped_bubble_image_rgb):
    """
    말풍선이 컷에 붙어있는지 확인합니다.
    """
    try:
        img = cropped_bubble_image_rgb
        img_h, img_w = img.shape[:2]

        scan_width = int(img_w * 0.10)
        if scan_width == 0: return 'none'

        left_crop = img[:, :scan_width]
        right_crop = img[:, -scan_width:]

        threshold = config.BUBBLE_ATTACHMENT_THRESHOLD
        has_left_vertical_line = False
        has_right_vertical_line = False

        # --- 왼쪽 영역 검사 ---
        gray_left = cv2.cvtColor(left_crop, cv2.COLOR_RGB2GRAY)
        edges_left = cv2.Canny(gray_left, 50, 150)
        lines_left = cv2.HoughLinesP(edges_left, 1, np.pi / 180, threshold=10,
                                     minLineLength=img_h * 0.8, maxLineGap=10)

        if lines_left is not None:
            for line in lines_left:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < threshold:
                    if x1 < threshold:
                        has_left_vertical_line = True
                        break

        # --- 오른쪽 영역 검사 ---
        gray_right = cv2.cvtColor(right_crop, cv2.COLOR_RGB2GRAY)
        edges_right = cv2.Canny(gray_right, 50, 150)
        lines_right = cv2.HoughLinesP(edges_right, 1, np.pi / 180, threshold=10,
                                      minLineLength=img_h * 0.8, maxLineGap=10)

        if lines_right is not None:
            for line in lines_right:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < threshold:
                    if x1 > scan_width - threshold:
                        has_right_vertical_line = True
                        break

        # --- 최종 판단 ---
        if has_right_vertical_line and not has_left_vertical_line:
            return 'right'
        if has_left_vertical_line and not has_right_vertical_line:
            return 'left'

        return 'none'
    except Exception:
        return 'none'


def process_image_batch(models, batch_images_rgb, batch_paths):
    """
    이미지 배치 전체를 받아 탐지, OCR, 구조화, 번역까지 모두 처리하는 통합 함수.
    [개선] 순차 처리 부분을 배치 및 병렬 처리로 최적화
    """
    detection_model, ocr_model, chat_session, font_classifier = models

    # 1. YOLO 일괄 탐지
    print(f"-> {len(batch_images_rgb)}개 페이지 일괄 탐지 중...")
    batch_results = detection_model(batch_images_rgb)

    # 2. 모든 페이지의 텍스트 조각(crop)과 원본 정보를 하나의 리스트로 통합
    all_items_to_process = []
    all_bubbles_by_page = [[] for _ in batch_images_rgb]
    for page_idx, (image_rgb, results) in enumerate(zip(batch_images_rgb, batch_results)):
        page_identifier = os.path.basename(batch_paths[page_idx])

        safe_subdir_name = page_identifier.replace('.', '_')
        page_debug_dir = os.path.join(config.DEBUG_CROPS_DIR, safe_subdir_name)
        if config.SAVE_DEBUG_CROPS:
            os.makedirs(page_debug_dir, exist_ok=True)

        for i, box in enumerate(results.boxes):
            class_name = results.names[int(box.cls[0])]
            coords = box.xyxy[0].cpu().numpy().astype(int)
            if class_name == 'bubble':
                all_bubbles_by_page[page_idx].append(coords)
            elif class_name in config.TARGET_CLASSES:
                cropped_pil = Image.fromarray(image_rgb[coords[1]:coords[3], coords[0]:coords[2]])
                if config.SAVE_DEBUG_CROPS:
                    crop_filename = f"crop_{i}_{class_name}.png"
                    save_path = os.path.join(page_debug_dir, crop_filename)
                    cropped_pil.save(save_path)
                all_items_to_process.append({
                    'page_idx': page_idx, 'crop': cropped_pil, 'box': coords, 'class_name': class_name
                })

    if not all_items_to_process:
        untranslated_data = [{"source_page": os.path.basename(p), "speech_bubbles": [], "freeform_texts": []} for p in
                             batch_paths]
        return translate_image_batch(chat_session, untranslated_data)

    # 3. MangaOCR 일괄 처리
    print(f"-> {len(all_items_to_process)}개의 텍스트 조각을 Batch OCR (GPU) 처리 중...")
    all_ocr_results = ocr_model([item['crop'] for item in all_items_to_process])

    # 4. [개선] 폰트 스타일 일괄 분류 (배치 처리)
    font_styles = ['standard'] * len(all_items_to_process)
    if font_classifier:
        print(f"-> {len(all_items_to_process)}개의 텍스트 조각을 폰트 분류 (배치) 처리 중...")
        # [수정] 크기가 다른 이미지를 바로 배열로 만들 수 없으므로, 먼저 리스트에 저장합니다.
        font_images_to_classify = [np.array(item['crop']) for item in all_items_to_process]

        # [수정] 각 이미지를 개별적으로 리사이즈한 후, tf.stack으로 하나의 배치 텐서로 합칩니다.
        resized_images_list = [tf.image.resize(img, (96, 96)) for img in font_images_to_classify]
        image_batch = tf.stack(resized_images_list)

        # 모델 예측
        predictions = font_classifier.predict(image_batch, batch_size=config.OCR_BATCH_SIZE, verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)
        font_styles = [config.FONT_CLASS_NAMES[i] for i in predicted_indices]

    # 5. [개선] 폰트 크기 병렬 계산 (병렬 처리)
    print(f"-> {len(all_items_to_process)}개의 텍스트 조각을 폰트 크기 (병렬) 계산 중...")
    crops_for_font_size = [item['crop'] for item in all_items_to_process]
    font_sizes = [0] * len(all_items_to_process)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        font_sizes = list(executor.map(_calculate_font_size, crops_for_font_size))

    # 6. [개선] 사전 계산된 결과 통합 (매우 빠름)
    processed_text_items = []
    print("-> 사전 계산된 결과 통합 중...")
    for i, item in enumerate(all_items_to_process):
        if not all_ocr_results[i]: continue

        font_size = font_sizes[i]
        font_style = font_styles[i]

        base_font_size = font_size if font_size > 0 else (item['box'][3] - item['box'][1])
        final_font_size = int(max(config.MIN_FONT_SIZE, min(base_font_size, config.MAX_FONT_SIZE)))

        processed_text_items.append({
            'page_idx': item['page_idx'], 'box': item['box'], 'class_name': item['class_name'],
            'original_text': all_ocr_results[i], 'font_size': final_font_size, 'font_style': font_style
        })

    # 7. 페이지별로 최종 데이터 구조 조립
    untranslated_batch_data = []
    for page_idx, path in enumerate(batch_paths):
        page_identifier = os.path.basename(path)
        image_rgb = batch_images_rgb[page_idx]

        page_bubbles = all_bubbles_by_page[page_idx]
        page_texts = [item for item in processed_text_items if
                      item['page_idx'] == page_idx and item['class_name'] == 'text']
        page_free_texts = [item for item in processed_text_items if
                           item['page_idx'] == page_idx and item['class_name'] == 'free_text']

        structured_bubbles = []
        unmatched_text_indices = set(range(len(page_texts)))
        for bubble_box in page_bubbles:
            bubble_center_x, bubble_center_y = (bubble_box[0] + bubble_box[2]) / 2, (bubble_box[1] + bubble_box[3]) / 2
            closest_text_idx, min_distance = -1, float('inf')

            for j, text_info in enumerate(page_texts):
                if j in unmatched_text_indices:
                    text_box = text_info['box']
                    text_center_x, text_center_y = (text_box[0] + text_box[2]) / 2, (text_box[1] + text_box[3]) / 2
                    distance = np.sqrt((bubble_center_x - text_center_x) ** 2 + (bubble_center_y - text_center_y) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_text_idx = j

            if closest_text_idx != -1:
                matched_text_info = page_texts[closest_text_idx]
                b_x1, b_y1, b_x2, b_y2 = bubble_box
                cropped_bubble = image_rgb[b_y1:b_y2, b_x1:b_x2]

                current_bubble = {
                    "bubble_box": bubble_box, "text_box": matched_text_info['box'],
                    "original_text": matched_text_info['original_text'], "font_size": matched_text_info['font_size'],
                    "font_style": matched_text_info['font_style'], "attachment": check_bubble_attachment(cropped_bubble)
                }
                structured_bubbles.append(current_bubble)
                unmatched_text_indices.remove(closest_text_idx)

        for bubble in structured_bubbles:
            bubble['bubble_box'] = bubble['bubble_box'].tolist()
            bubble['text_box'] = bubble['text_box'].tolist()

        for ff_text in page_free_texts:
            if 'box' in ff_text:
                ff_text['text_box'] = ff_text['box'].tolist()
                del ff_text['box']

        untranslated_batch_data.append({
            "source_page": page_identifier,
            "speech_bubbles": structured_bubbles,
            "freeform_texts": page_free_texts
        })

    # 8. 배치 단위 번역
    final_batch_data = translate_image_batch(chat_session, untranslated_batch_data)

    return final_batch_data



def translate_image_batch(chat_session, batch_page_data):
    """여러 페이지의 데이터를 받아, 모든 텍스트를 모아 단 한 번의 API 호출로 번역합니다."""
    print(f"{len(batch_page_data)} 페이지의 데이터 전체 번역을 시작합니다...")

    all_texts_to_translate = []
    for page_idx, page_data in enumerate(batch_page_data):
        for item in page_data['speech_bubbles']:
            if item.get('original_text'):
                all_texts_to_translate.append({
                    "page_idx": page_idx,
                    "source_item": item,
                    "text": item.get('original_text')
                })
        for item in page_data['freeform_texts']:
            if item.get('original_text'):
                all_texts_to_translate.append({
                    "page_idx": page_idx,
                    "source_item": item,
                    "text": item.get('original_text')
                })

    if not all_texts_to_translate:
        print("번역할 텍스트가 없습니다.")
        return batch_page_data

    formatted_request_text = ""
    for i, item in enumerate(all_texts_to_translate):
        formatted_request_text += f"{item['page_idx'] + 1}.{i + 1}.({item['text']})\n"

    try:
        prompt = formatted_request_text.strip()
        response = chat_session.send_message(prompt)

        print("\n--- 통합 번역 결과 ---")
        print(f">> 요청:\n{prompt}")
        print(f"\n>> 응답:\n{response.text.strip()}")

        translated_lines = response.text.strip().split('\n')

        for i, item_to_translate in enumerate(all_texts_to_translate):
            if i < len(translated_lines):
                line = translated_lines[i]
                match = re.search(r'\((.*)\)', line)
                if match:
                    cleaned_line = match.group(1)
                else:
                    parts = line.split('.', 2)
                    cleaned_line = parts[2].strip() if len(parts) > 2 else line

                final_text = cleaned_line.replace("...", "···")

                item_to_translate['source_item']['translated_text'] = final_text
            else:
                item_to_translate['source_item']['translated_text'] = ""

    except Exception as e:
        print(f"통합 번역 중 오류 발생: {e}")
        for item_to_translate in all_texts_to_translate:
            item_to_translate['source_item']['translated_text'] = ""

    print("\n번역 완료.")
    return batch_page_data