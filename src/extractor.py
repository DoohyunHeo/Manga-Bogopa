import os
import re
import cv2
import json
import numpy as np
import google.generativeai as genai
import timm
from PIL import Image
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from src import config
from src.batch_manga_ocr import BatchMangaOcr
from simple_lama_inpainting import SimpleLama


class Letterbox:
    def __init__(self, new_shape=(256, 256), color=(128, 128, 128)):
        self.new_shape = new_shape
        self.color = color

    def __call__(self, img):
        shape = img.size
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        r = min(r, 1.0)
        new_unpad = (int(round(shape[0] * r)), int(round(shape[1] * r)))
        dw, dh = (self.new_shape[0] - new_unpad[0]) // 2, (self.new_shape[1] - new_unpad[1]) // 2
        if shape != new_unpad:
            img = img.resize(new_unpad, Image.Resampling.LANCZOS)
        new_image = Image.new("RGB", self.new_shape, self.color)
        new_image.paste(img, (dw, dh))
        return new_image


class FontClassifierModel(nn.Module):
    def __init__(self, num_classes, style_mapping=None):
        super(FontClassifierModel, self).__init__()
        # [수정] timm을 사용하여 ConvNeXT V2 백본 로드
        self.backbone = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in1k',
            pretrained=True,
            features_only=True
        )
        last_channel = 768 # ConvNeXT V2 Tiny의 출력 채널
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(last_channel, 512), nn.ReLU())
        self.angle_head = nn.Linear(512, 1)
        self.style_head = nn.Linear(512, num_classes)
        self.style_mapping = style_mapping

    def forward(self, x):
        # timm의 features_only는 각 단계별 특징 맵의 리스트를 반환하므로, 마지막 것을 사용
        features = self.backbone(x)
        x = features[-1]
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        angle_output = self.angle_head(x).squeeze(-1)
        style_output = self.style_head(x)
        return {"angle": angle_output, "style": style_output}


class FontSizeModel(nn.Module):
    def __init__(self):
        super(FontSizeModel, self).__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 512), nn.ReLU())
        self.size_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.backbone(x);
        x = self.pool(x);
        x = torch.flatten(x, 1);
        x = self.fc(x)
        return self.size_head(x).squeeze(-1)


def initialize_models_and_session():
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
    ocr_model = BatchMangaOcr(batch_size=config.OCR_BATCH_SIZE)
    lama_model = SimpleLama(device=config.DEVICE)
    print("Detection, OCR, LaMa 모델 로딩 완료.")

    try:
        checkpoint = torch.load(config.MTL_MODEL_PATH, map_location=config.DEVICE)
        style_mapping = checkpoint['style_mapping']
        num_classes = len(style_mapping)
        font_classifier_model = FontClassifierModel(num_classes, style_mapping)
        font_classifier_model.load_state_dict(checkpoint['model_state_dict'])
        font_classifier_model.to(config.DEVICE)
        font_classifier_model.eval()
        print("PyTorch FontClassifier(스타일/각도) 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"경고: FontClassifier 모델 로드 실패. -> {e}")
        font_classifier_model = None

    try:
        font_size_model = FontSizeModel()
        font_size_model.load_state_dict(torch.load(config.FONT_SIZE_MODEL_PATH, map_location=config.DEVICE))
        font_size_model.to(config.DEVICE)
        font_size_model.eval()
        print("PyTorch FontSize 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"경고: PyTorch FontSize 모델 로드 실패. -> {e}")
        font_size_model = None

    translation_model = genai.GenerativeModel(config.GEMINI_MODEL)
    chat_session = translation_model.start_chat(history=[{'role': 'user', 'parts': [system_prompt]}, {'role': 'model',
                                                                                                      'parts': [
                                                                                                          "네, 알겠습니다. 이제부터 지시에 따라 번역을 시작하겠습니다."]}])
    print("초기화 완료.")
    return (detection_model, ocr_model, chat_session, lama_model, font_classifier_model, font_size_model)


def check_bubble_attachment(cropped_bubble_image_rgb):
    try:
        img = cropped_bubble_image_rgb;
        img_h, img_w = img.shape[:2]
        scan_width = int(img_w * 0.10)
        if scan_width == 0: return 'none'
        left_crop, right_crop = img[:, :scan_width], img[:, -scan_width:]
        threshold = config.BUBBLE_ATTACHMENT_THRESHOLD
        gray_left = cv2.cvtColor(left_crop, cv2.COLOR_RGB2GRAY);
        edges_left = cv2.Canny(gray_left, 50, 150)
        lines_left = cv2.HoughLinesP(edges_left, 1, np.pi / 180, threshold=10, minLineLength=img_h * 0.8, maxLineGap=10)
        has_left_vertical_line = any(abs(l[0] - l[2]) < threshold and l[0] < threshold for l in
                                     lines_left[0]) if lines_left is not None else False
        gray_right = cv2.cvtColor(right_crop, cv2.COLOR_RGB2GRAY);
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
    detection_model, ocr_model, chat_session, font_classifier_model, font_size_model = models

    print(f"-> {len(batch_images_rgb)}개 페이지 일괄 탐지 중...")
    batch_results = detection_model(batch_images_rgb, conf=0.5)
    all_items_to_process, all_bubbles_by_page = [], [[] for _ in batch_images_rgb]
    for page_idx, (image_rgb, results) in enumerate(zip(batch_images_rgb, batch_results)):
        page_identifier = os.path.basename(batch_paths[page_idx]);
        safe_subdir_name = os.path.splitext(page_identifier)[0]
        page_debug_dir = os.path.join(config.DEBUG_CROPS_DIR, safe_subdir_name)
        if config.SAVE_DEBUG_CROPS: os.makedirs(page_debug_dir, exist_ok=True)
        for i, box in enumerate(results.boxes):
            class_name = results.names[int(box.cls[0])];
            coords = box.xyxy[0].cpu().numpy().astype(int)
            if class_name == 'bubble':
                all_bubbles_by_page[page_idx].append(coords)
            elif class_name in config.TARGET_CLASSES:
                cropped_pil = Image.fromarray(image_rgb[coords[1]:coords[3], coords[0]:coords[2]])
                if config.SAVE_DEBUG_CROPS and page_debug_dir:
                    cropped_pil.save(os.path.join(page_debug_dir, f"{i:03d}_{class_name}.png"))
                all_items_to_process.append(
                    {'page_idx': page_idx, 'crop': cropped_pil, 'box': coords, 'class_name': class_name})
    if not all_items_to_process: return [
        {"source_page": os.path.basename(p), "speech_bubbles": [], "freeform_texts": []} for p in batch_paths]

    print(f"-> {len(all_items_to_process)}개의 텍스트 조각을 Batch OCR 처리 중...")
    all_ocr_results = ocr_model([item['crop'] for item in all_items_to_process])

    all_props = []
    if font_classifier_model and font_size_model:
        print(f"-> {len(all_items_to_process)}개의 텍스트 조각을 폰트 모델로 일괄 분석 중...")
        transform = transforms.Compose([Letterbox(config.IMAGE_SIZE), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image_tensors = [transform(item['crop'].convert("RGB")) for item in all_items_to_process]
        image_batch = torch.stack(image_tensors).to(config.DEVICE)

        with torch.no_grad():
            outputs_classifier = font_classifier_model(image_batch)
            pred_sizes = font_size_model(image_batch).cpu().numpy()

        pred_angles = outputs_classifier['angle'].cpu().numpy()
        pred_style_indices = torch.argmax(outputs_classifier['style'], dim=1).cpu().numpy()

        for i in range(len(pred_sizes)):
            style_name = font_classifier_model.style_mapping.get(pred_style_indices[i], 'standard')
            all_props.append(
                {'font_size': int(round(pred_sizes[i])) + 6, 'angle': float(pred_angles[i]), 'font_style': style_name})
    else:
        all_props = [{'font_size': 20, 'angle': 0.0, 'font_style': 'standard'}] * len(all_items_to_process)

    processed_text_items = []
    for i, item in enumerate(all_items_to_process):
        if all_ocr_results[i]: processed_text_items.append(
            {'page_idx': item['page_idx'], 'box': item['box'], 'class_name': item['class_name'],
             'original_text': all_ocr_results[i], **all_props[i]})

    untranslated_batch_data = []
    for page_idx, path in enumerate(batch_paths):
        page_identifier, image_rgb, page_bubbles = os.path.basename(path), batch_images_rgb[page_idx], \
        all_bubbles_by_page[page_idx]
        page_texts = [item for item in processed_text_items if
                      item['page_idx'] == page_idx and item['class_name'] == 'text']
        page_free_texts = [item for item in processed_text_items if
                           item['page_idx'] == page_idx and item['class_name'] == 'free_text']
        structured_bubbles, unmatched_text_indices = [], set(range(len(page_texts)))
        for bubble_box in page_bubbles:
            bubble_center = ((bubble_box[0] + bubble_box[2]) / 2, (bubble_box[1] + bubble_box[3]) / 2)
            closest_text_idx, min_distance = -1, float('inf')
            for j, text_info in enumerate(page_texts):
                if j in unmatched_text_indices:
                    text_center = (
                    (text_info['box'][0] + text_info['box'][2]) / 2, (text_info['box'][1] + text_info['box'][3]) / 2)
                    distance = np.sqrt(
                        (bubble_center[0] - text_center[0]) ** 2 + (bubble_center[1] - text_center[1]) ** 2)
                    if distance < min_distance: min_distance, closest_text_idx = distance, j
            if closest_text_idx != -1:
                matched = page_texts[closest_text_idx]
                b = bubble_box;
                cropped_bubble = image_rgb[b[1]:b[3], b[0]:b[2]]
                current_bubble = {"bubble_box": b.tolist(), "text_box": matched['box'].tolist(),
                                  **{k: v for k, v in matched.items() if k not in ['page_idx', 'box', 'class_name']},
                                  "attachment": check_bubble_attachment(cropped_bubble)}
                structured_bubbles.append(current_bubble)
                unmatched_text_indices.remove(closest_text_idx)
        for ff_text in page_free_texts: ff_text['text_box'] = ff_text.pop('box', None).tolist()
        untranslated_batch_data.append(
            {"source_page": page_identifier, "speech_bubbles": structured_bubbles, "freeform_texts": page_free_texts})

    return translate_image_batch(chat_session, untranslated_batch_data)


def translate_image_batch(chat_session, batch_page_data):
    print(f"-> {len(batch_page_data)} 페이지 데이터 전체 번역 요청...")
    texts_to_translate = [{"source_item": item, "text": item.get('original_text')} for p in batch_page_data for item in
                          p['speech_bubbles'] + p['freeform_texts'] if item.get('original_text')]
    if not texts_to_translate: return batch_page_data
    request_text = "\n".join([f"{i + 1}.({item['text']})" for i, item in enumerate(texts_to_translate)])
    try:
        response = chat_session.send_message(request_text)
        lines = response.text.strip().split('\n')
        for i, item in enumerate(texts_to_translate):
            if i < len(lines):
                line = lines[i]
                match = re.search(r'\(.*\)', line)
                cleaned = match.group(0)[1:-1] if match else line.split('.', 1)[-1].strip()
                item['source_item']['translated_text'] = cleaned.replace("...", "···")
            else:
                item['source_item']['translated_text'] = ""
    except Exception as e:
        print(f"-> 통합 번역 중 오류 발생: {e}")
        for item in texts_to_translate: item['source_item']['translated_text'] = "[번역 실패]"
    print("-> 번역 완료.")
    return batch_page_data