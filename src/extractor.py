import os
import re
import cv2
import pytesseract
import google.generativeai as genai
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from manga_ocr import MangaOcr
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from src import config

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
    ocr_model = MangaOcr()

    try:
        font_classifier_model = tf.keras.models.load_model(config.FONT_CLASSIFIER_PATH)
        print("폰트 분류 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"경고: 폰트 분류 모델 로드 실패. 기본 스타일만 사용됩니다. -> {e}")
        font_classifier_model = None

    translation_model = genai.GenerativeModel(config.GEMINI_MODEL)
    chat_session = translation_model.start_chat(history=[
        {'role': 'user', 'parts': [system_prompt]},
        {'role': 'model', 'parts': ["네, 알겠습니다. 이제부터 지시에 따라 번역을 시작하겠습니다."]},
    ])

    print("초기화 완료.")
    return detection_model, ocr_model, chat_session, font_classifier_model


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


def structure_page_data(models, image_rgb, page_identifier, detection_results):
    _, ocr_model, _, font_classifier = models
    print(f"  -> [{page_identifier}] 탐지 결과 구조화 시작...")

    boxes = detection_results.boxes
    class_names = detection_results.names

    safe_subdir_name = page_identifier.replace('.', '_')
    page_debug_dir = os.path.join(config.DEBUG_CROPS_DIR, safe_subdir_name)
    os.makedirs(page_debug_dir, exist_ok=True)

    bubble_boxes = []
    text_boxes_with_info = []
    freeform_texts = []

    # 1. OCR, 폰트 계산/분류 병렬 처리를 위한 작업 목록 생성
    items_to_process = []
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        coords = box.xyxy[0].cpu().numpy().astype(int)
        if class_name == 'bubble':
            bubble_boxes.append(coords)
        elif class_name in config.TARGET_CLASSES:
            cropped_pil = Image.fromarray(image_rgb[coords[1]:coords[3], coords[0]:coords[2]])
            items_to_process.append({'crop': cropped_pil, 'box': coords, 'class_name': class_name, 'crop_idx': i})

    def _process_item(item):
        ocr_text = ocr_model(item['crop'])
        font_size = _calculate_font_size(item['crop'])
        font_style = 'standard'
        if font_classifier and ocr_text:
            img_array = tf.image.resize(np.array(item['crop']), (96, 96))
            img_array = tf.expand_dims(img_array, 0)
            predictions = font_classifier.predict(img_array, verbose=0)
            font_style = config.FONT_CLASS_NAMES[np.argmax(predictions[0])]
        return ocr_text, font_size, font_style, item

    # 2. 병렬 처리 실행 및 결과 수집
    print(f"  -> {len(items_to_process)}개의 텍스트 조각을 병렬 처리 중...")
    with ThreadPoolExecutor() as executor:
        results = executor.map(_process_item, items_to_process)

    for ocr_text, font_size, font_style, source_item in results:
        if ocr_text:
            coords, class_name = source_item['box'], source_item['class_name']
            if config.SAVE_DEBUG_CROPS:
                crop_filename = f"crop_{source_item['crop_idx']}_{class_name}.png"
                source_item['crop'].save(os.path.join(page_debug_dir, crop_filename))

            base_font_size = font_size if font_size > 0 else (coords[3] - coords[1])
            final_font_size = int(max(config.MIN_FONT_SIZE, min(base_font_size, config.MAX_FONT_SIZE)))
            item_data = {"original_text": ocr_text, "font_size": final_font_size, "font_style": font_style}

            if class_name == 'free_text':
                item_data["text_box"] = coords.tolist()
                freeform_texts.append(item_data)
            else:
                item_data.update({'box': coords})
                text_boxes_with_info.append(item_data)

    # 3. 말풍선과 텍스트 1:1 매칭
    structured_bubbles = []
    unmatched_text_indices = set(range(len(text_boxes_with_info)))
    for i, bubble_box in enumerate(bubble_boxes):
        bubble_center_x, bubble_center_y = (bubble_box[0] + bubble_box[2]) / 2, (bubble_box[1] + bubble_box[3]) / 2
        closest_text_idx, min_distance = -1, float('inf')

        for j, text_info in enumerate(text_boxes_with_info):
            if j in unmatched_text_indices:
                text_box = text_info['box']
                text_center_x, text_center_y = (text_box[0] + text_box[2]) / 2, (text_box[1] + text_box[3]) / 2
                distance = np.sqrt((bubble_center_x - text_center_x) ** 2 + (bubble_center_y - text_center_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_text_idx = j

        if closest_text_idx != -1:
            matched_text_info = text_boxes_with_info[closest_text_idx]
            cropped_bubble = image_rgb[bubble_box[1]:bubble_box[3], bubble_box[0]:bubble_box[2]]

            current_bubble = {
                "bubble_box": bubble_box.tolist(),
                "text_box": matched_text_info['box'].tolist(),
                "original_text": matched_text_info['original_text'],
                "font_size": matched_text_info['font_size'],
                "font_style": matched_text_info['font_style'],
                "attachment": check_bubble_attachment(cropped_bubble)
            }
            structured_bubbles.append(current_bubble)
            unmatched_text_indices.remove(closest_text_idx)

    print(f"  -> 총 {len(structured_bubbles)}개의 말풍선과 {len(freeform_texts)}개의 자유 텍스트 구조화 완료.")
    return {"source_page": page_identifier, "speech_bubbles": structured_bubbles, "freeform_texts": freeform_texts}


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