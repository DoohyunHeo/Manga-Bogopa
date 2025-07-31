import os
import re
import cv2
import json
import numpy as np
import google.generativeai as genai
from PIL import Image
from ultralytics import YOLO

# PyTorch 관련 라이브러리 추가
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from src import config
from src.batch_manga_ocr import BatchMangaOcr
from simple_lama_inpainting import SimpleLama


# 훈련 스크립트에서 정의했던 MTLModel 클래스
class MTLModel(nn.Module):
    def __init__(self, num_classes):
        super(MTLModel, self).__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 512),
            nn.ReLU()
        )
        self.size_head = nn.Linear(512, 1)
        self.angle_head = nn.Linear(512, 1)
        self.style_head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        size_output = self.size_head(x)
        angle_output = self.angle_head(x)
        style_output = self.style_head(x)
        return {
            "font_size": size_output.squeeze(-1),  # squeeze() -> squeeze(-1) for batch safety
            "angle": angle_output.squeeze(-1),
            "style": style_output
        }


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

    # 공통 모델 로딩
    detection_model = YOLO(config.MODEL_PATH)
    detection_model.to(config.DEVICE)
    ocr_model = BatchMangaOcr(batch_size=config.OCR_BATCH_SIZE)
    lama_model = SimpleLama(device=config.DEVICE)
    print("Detection, OCR, LaMa 모델 로딩 완료.")

    # PyTorch MTL 모델 로딩
    try:
        with open(config.LABEL_ENCODER_PATH, 'r', encoding='utf-8') as f:
            style_mapping = json.load(f)
        num_classes = len(style_mapping)

        mtl_model = MTLModel(num_classes)
        mtl_model.load_state_dict(torch.load(config.MTL_MODEL_PATH, map_location=config.DEVICE))
        mtl_model.to(config.DEVICE)
        mtl_model.eval()
        print("PyTorch MTL 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"경고: PyTorch MTL 모델 로드 실패. -> {e}")
        mtl_model = None
        style_mapping = None

    # Gemini 챗 세션 초기화
    translation_model = genai.GenerativeModel(config.GEMINI_MODEL)
    chat_session = translation_model.start_chat(history=[
        {'role': 'user', 'parts': [system_prompt]},
        {'role': 'model', 'parts': ["네, 알겠습니다. 이제부터 지시에 따라 번역을 시작하겠습니다."]},
    ])

    print("초기화 완료.")
    return (detection_model, ocr_model, chat_session, lama_model,
            mtl_model, style_mapping)


def check_bubble_attachment(cropped_bubble_image_rgb):
    """말풍선이 컷에 붙어있는지 확인합니다."""
    try:
        img = cropped_bubble_image_rgb
        img_h, img_w = img.shape[:2]
        scan_width = int(img_w * 0.10)
        if scan_width == 0: return 'none'
        left_crop, right_crop = img[:, :scan_width], img[:, -scan_width:]
        threshold = config.BUBBLE_ATTACHMENT_THRESHOLD

        gray_left = cv2.cvtColor(left_crop, cv2.COLOR_RGB2GRAY)
        edges_left = cv2.Canny(gray_left, 50, 150)
        lines_left = cv2.HoughLinesP(edges_left, 1, np.pi / 180, threshold=10, minLineLength=img_h * 0.8, maxLineGap=10)
        has_left_vertical_line = any(abs(l[0] - l[2]) < threshold and l[0] < threshold for l in
                                     lines_left[0]) if lines_left is not None else False

        gray_right = cv2.cvtColor(right_crop, cv2.COLOR_RGB2GRAY)
        edges_right = cv2.Canny(gray_right, 50, 150)
        lines_right = cv2.HoughLinesP(edges_right, 1, np.pi / 180, threshold=10, minLineLength=img_h * 0.8,
                                      maxLineGap=10)
        has_right_vertical_line = any(abs(l[0] - l[2]) < threshold and l[0] > scan_width - threshold for l in
                                      lines_right[0]) if lines_right is not None else False

        if has_right_vertical_line and not has_left_vertical_line: return 'right'
        if has_left_vertical_line and not has_right_vertical_line: return 'left'
        return 'none'
    except Exception:
        return 'none'


def process_image_batch(models, batch_images_rgb, batch_paths):
    """이미지 배치 전체를 받아 탐지, OCR, 속성 분석, 번역까지 모두 처리합니다."""
    detection_model, ocr_model, chat_session, mtl_model, style_mapping = models

    # 1. YOLO 일괄 탐지
    print(f"-> {len(batch_images_rgb)}개 페이지 일괄 탐지 중...")
    batch_results = detection_model(batch_images_rgb)

    # 2. 모든 페이지의 텍스트 조각(crop)과 원본 정보를 하나의 리스트로 통합
    all_items_to_process = []
    all_bubbles_by_page = [[] for _ in batch_images_rgb]
    for page_idx, (image_rgb, results) in enumerate(zip(batch_images_rgb, batch_results)):
        for i, box in enumerate(results.boxes):
            class_name = results.names[int(box.cls[0])]
            coords = box.xyxy[0].cpu().numpy().astype(int)
            if class_name == 'bubble':
                all_bubbles_by_page[page_idx].append(coords)
            elif class_name in config.TARGET_CLASSES:
                cropped_pil = Image.fromarray(image_rgb[coords[1]:coords[3], coords[0]:coords[2]])
                all_items_to_process.append({
                    'page_idx': page_idx, 'crop': cropped_pil, 'box': coords, 'class_name': class_name
                })
    if not all_items_to_process:
        return [{"source_page": os.path.basename(p), "speech_bubbles": [], "freeform_texts": []} for p in batch_paths]

    # 3. MangaOCR 일괄 처리
    print(f"-> {len(all_items_to_process)}개의 텍스트 조각을 Batch OCR 처리 중...")
    all_ocr_results = ocr_model([item['crop'] for item in all_items_to_process])

    # 4. MTL 모델 일괄 예측
    all_props = []
    if mtl_model and all_items_to_process:
        print(f"-> {len(all_items_to_process)}개의 텍스트 조각을 MTL 모델로 일괄 분석 중...")
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        crops_for_mtl = [item['crop'].convert("RGB") for item in all_items_to_process]
        image_tensors = [transform(crop) for crop in crops_for_mtl]
        image_batch = torch.stack(image_tensors).to(config.DEVICE)

        with torch.no_grad():
            outputs = mtl_model(image_batch)

        pred_sizes = outputs['font_size'].cpu().numpy()
        pred_angles = outputs['angle'].cpu().numpy()
        pred_style_indices = torch.argmax(outputs['style'], dim=1).cpu().numpy()

        for i in range(len(pred_sizes)):
            style_idx_str = str(pred_style_indices[i])
            all_props.append({
                'font_size': int(round(pred_sizes[i])),
                'angle': float(pred_angles[i]),
                'font_style': style_mapping.get(style_idx_str, 'standard')
            })
    else:
        all_props = [{'font_size': 20, 'angle': 0.0, 'font_style': 'standard'}] * len(all_items_to_process)

    # 5. 사전 계산된 결과 통합
    processed_text_items = []
    for i, item in enumerate(all_items_to_process):
        if not all_ocr_results[i]: continue
        props = all_props[i]
        processed_text_items.append({
            'page_idx': item['page_idx'], 'box': item['box'], 'class_name': item['class_name'],
            'original_text': all_ocr_results[i], 'font_size': props['font_size'],
            'angle': props['angle'], 'font_style': props['font_style'],
        })

    # 6. 페이지별로 최종 데이터 구조 조립
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
                        min_distance, closest_text_idx = distance, j

            if closest_text_idx != -1:
                matched_text_info = page_texts[closest_text_idx]
                b_x1, b_y1, b_x2, b_y2 = bubble_box
                cropped_bubble = image_rgb[b_y1:b_y2, b_x1:b_x2]
                current_bubble = {
                    "bubble_box": bubble_box.tolist(), "text_box": matched_text_info['box'].tolist(),
                    "original_text": matched_text_info['original_text'], "font_size": matched_text_info['font_size'],
                    "font_style": matched_text_info['font_style'], "angle": matched_text_info['angle'],
                    "attachment": check_bubble_attachment(cropped_bubble)
                }
                structured_bubbles.append(current_bubble)
                unmatched_text_indices.remove(closest_text_idx)

        for ff_text in page_free_texts:
            ff_text['text_box'] = ff_text['box'].tolist()
            del ff_text['box']

        untranslated_batch_data.append({
            "source_page": page_identifier,
            "speech_bubbles": structured_bubbles,
            "freeform_texts": page_free_texts
        })

    # 7. 배치 단위 번역
    final_batch_data = translate_image_batch(chat_session, untranslated_batch_data)
    return final_batch_data


def translate_image_batch(chat_session, batch_page_data):
    """여러 페이지의 데이터를 받아, 모든 텍스트를 모아 단 한 번의 API 호출로 번역합니다."""
    print(f"-> {len(batch_page_data)} 페이지 데이터 전체 번역 요청...")
    all_texts_to_translate = []
    for page_idx, page_data in enumerate(batch_page_data):
        for item in page_data['speech_bubbles'] + page_data['freeform_texts']:
            if item.get('original_text'):
                all_texts_to_translate.append({"source_item": item, "text": item.get('original_text')})

    if not all_texts_to_translate:
        print("-> 번역할 텍스트가 없습니다.")
        return batch_page_data

    formatted_request_text = "\n".join([f"{i + 1}.({item['text']})" for i, item in enumerate(all_texts_to_translate)])

    try:
        response = chat_session.send_message(formatted_request_text)
        translated_lines = response.text.strip().split('\n')

        for i, item_to_translate in enumerate(all_texts_to_translate):
            if i < len(translated_lines):
                line = translated_lines[i]
                match = re.search(r'\(.*\)', line)
                cleaned_line = match.group(0)[1:-1] if match else line.split('.', 1)[-1].strip()
                item_to_translate['source_item']['translated_text'] = cleaned_line.replace("...", "···")
            else:
                item_to_translate['source_item']['translated_text'] = ""
    except Exception as e:
        print(f"-> 통합 번역 중 오류 발생: {e}")
        for item_to_translate in all_texts_to_translate:
            item_to_translate['source_item']['translated_text'] = "[번역 실패]"

    print("-> 번역 완료.")
    return batch_page_data