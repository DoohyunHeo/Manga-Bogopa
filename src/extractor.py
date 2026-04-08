import logging
import os
from collections import defaultdict, deque
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

from src import config
from src.utils import Letterbox, calculate_iou, merge_boxes, is_box_inside
from src.data_models import Attachment, PageData, TextElement, SpeechBubble


def check_bubble_attachment(cropped_bubble_image_rgb):
    """잘라낸 말풍선 이미지를 분석하여 말꼬리의 방향을 판단합니다."""
    try:
        img = cropped_bubble_image_rgb
        img_h, img_w = img.shape[:2]
        scan_width = int(img_w * 0.10)
        if scan_width == 0: return Attachment.NONE

        left_crop = img[:, :scan_width]
        right_crop = img[:, -scan_width:]
        threshold = config.BUBBLE_ATTACHMENT_THRESHOLD

        gray_left = cv2.cvtColor(left_crop, cv2.COLOR_RGB2GRAY)
        edges_left = cv2.Canny(gray_left, 50, 150)
        lines_left = cv2.HoughLinesP(edges_left, 1, np.pi / 180, threshold=10, minLineLength=img_h * 0.8, maxLineGap=10)
        has_left_vertical_line = any(abs(l[0] - l[2]) < threshold and l[0] < threshold for l in lines_left[0]) if lines_left is not None else False

        gray_right = cv2.cvtColor(right_crop, cv2.COLOR_RGB2GRAY)
        edges_right = cv2.Canny(gray_right, 50, 150)
        lines_right = cv2.HoughLinesP(edges_right, 1, np.pi / 180, threshold=10, minLineLength=img_h * 0.8, maxLineGap=10)
        has_right_vertical_line = any(abs(l[0] - l[2]) < threshold and l[0] > scan_width - threshold for l in lines_right[0]) if lines_right is not None else False

        if has_right_vertical_line and not has_left_vertical_line: return Attachment.RIGHT
        if has_left_vertical_line and not has_right_vertical_line: return Attachment.LEFT
        return Attachment.NONE
    except Exception as e:
        logger.warning(f"말풍선 방향 감지 실패: {e}")
        return Attachment.NONE


def detect_objects(detection_model, batch_images_rgb):
    """YOLO 모델을 사용하여 이미지 내의 모든 객체를 탐지합니다."""
    logger.info(f"{len(batch_images_rgb)}개 페이지 일괄 탐지 중...")
    # 모델에 저장된 imgsz를 사용하고, 없을 경우 기본값으로 (800, 560)을 사용합니다.
    batch_results = detection_model(batch_images_rgb, conf=config.YOLO_CONF_THRESHOLD, verbose=False)

    all_text_items = []
    all_bubbles_by_page = [[] for _ in batch_images_rgb]
    for page_idx, results in enumerate(tqdm(batch_results, desc="Detection")):
        for box in results.boxes:
            class_name = results.names[int(box.cls[0])]
            coords = box.xyxy[0].cpu().numpy()
            if class_name == 'bubble':
                all_bubbles_by_page[page_idx].append(coords.astype(int))
            elif class_name in ['text', 'free_text']:
                all_text_items.append({
                    'page_idx': page_idx, 'box': coords, 'class_name': class_name
                })
    return all_text_items, all_bubbles_by_page


def merge_text_boxes(text_items):
    """탐지된 텍스트 박스 중 겹치는 것들을 병합합니다."""
    logger.info(f"{len(text_items)}개의 탐지된 객체에 대해 중복 박스 병합 처리 중...")
    grouped_items = defaultdict(list)
    for item in text_items:
        grouped_items[(item['page_idx'], item['class_name'])].append(item)

    final_items = []
    for (page_idx, class_name), items in grouped_items.items():
        if len(items) < 2:
            final_items.extend(items)
            continue

        num_items = len(items)
        adj_matrix = np.zeros((num_items, num_items))
        for i in range(num_items):
            for j in range(i + 1, num_items):
                iou = calculate_iou(items[i]['box'], items[j]['box'])
                if iou > config.TEXT_MERGE_OVERLAP_THRESHOLD:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1

        visited = [False] * num_items
        for i in range(num_items):
            if not visited[i]:
                component = []
                q = deque([i])
                visited[i] = True
                while q:
                    u = q.popleft()
                    component.append(u)
                    for v in range(num_items):
                        if adj_matrix[u, v] and not visited[v]:
                            visited[v] = True
                            q.append(v)
                
                if len(component) > 1:
                    cluster_items = [items[k] for k in component]
                    merged_box = merge_boxes([item['box'] for item in cluster_items])
                    final_items.append({'page_idx': page_idx, 'box': merged_box, 'class_name': class_name})
                else:
                    final_items.append(items[component[0]])
    
    logger.info(f"병합 후 {len(final_items)}개의 객체로 정리되었습니다.")
    return final_items


def _prepare_crops(text_items, batch_images_rgb, batch_paths):
    """텍스트 아이템에서 크롭 이미지를 준비하고 OCR용 이미지 리스트를 반환합니다."""
    crops_for_ocr = []
    for item in text_items:
        image_rgb = batch_images_rgb[item['page_idx']]
        coords = item['box'].astype(int)
        original_crop_pil = Image.fromarray(image_rgb[coords[1]:coords[3], coords[0]:coords[2]])
        item['crop'] = original_crop_pil

        crops_for_ocr.append(original_crop_pil)

        if config.SAVE_DEBUG_CROPS:
            try:
                page_name = os.path.splitext(os.path.basename(batch_paths[item['page_idx']]))[0]
                x1, y1, x2, y2 = coords
                crop_filename = f"{page_name}_{item['class_name']}_{x1}_{y1}_{x2}_{y2}.png"
                crop_path = os.path.join(config.DEBUG_CROPS_DIR, crop_filename)
                ocr_crop_pil.save(crop_path)
            except Exception as e:
                logger.warning(f"디버그 크롭 저장 실패: {e}")

    return crops_for_ocr


def _predict_font_properties(font_classifier_model, font_size_model, text_items):
    """폰트 스타일, 크기, 각도를 예측합니다."""
    if not (font_classifier_model and font_size_model):
        return [{'font_size': 20, 'angle': 0, 'font_style': 'standard'}] * len(text_items)

    logger.info(f"{len(text_items)}개의 텍스트 조각을 폰트 모델로 분석 중...")
    transform = transforms.Compose([
        Letterbox(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_props = []
    font_batch_size = config.FONT_MODEL_BATCH_SIZE
    for i in tqdm(range(0, len(text_items), font_batch_size), desc="Font Model"):
        batch_items = text_items[i:i+font_batch_size]
        image_tensors = torch.stack([transform(item['crop'].convert("RGB")) for item in batch_items]).to(config.DEVICE)

        with torch.no_grad():
            outputs_classifier = font_classifier_model(image_tensors)
            pred_sizes = font_size_model(image_tensors)
            pred_angles = outputs_classifier['angle'].cpu().numpy()
            pred_style_indices = torch.argmax(outputs_classifier['style'], dim=1).cpu().numpy()
            pred_sizes_np = pred_sizes.cpu().numpy()

        for j in range(len(batch_items)):
            if 0 in font_classifier_model.style_mapping:
                style_idx = int(pred_style_indices[j])
            else:
                style_idx = str(pred_style_indices[j])

            style_name = font_classifier_model.style_mapping.get(style_idx, 'standard')
            all_props.append({
                'font_size': int(round(pred_sizes_np[j])),
                'angle': int(pred_angles[j]),
                'font_style': style_name
            })

    return all_props


def extract_text_properties(models, batch_images_rgb, text_items, batch_paths):
    """텍스트 박스에 대해 OCR과 폰트 분석을 수행하고 TextElement 리스트를 반환합니다."""
    if not text_items:
        return []

    # 1. 크롭 이미지 준비
    crops_for_ocr = _prepare_crops(text_items, batch_images_rgb, batch_paths)

    # 2. OCR 실행
    logger.info(f"{len(crops_for_ocr)}개의 텍스트 조각을 Batch OCR 처리 중...")
    all_ocr_results = models['ocr'](crops_for_ocr)

    # 3. 폰트 속성 예측
    all_props = _predict_font_properties(models['font_classifier'], models['font_size'], text_items)

    # 4. TextElement 객체 생성
    processed_text_elements = []
    for i, item in enumerate(text_items):
        if all_ocr_results[i]:
            element = TextElement(
                text_box=item['box'].tolist(),
                original_text=all_ocr_results[i],
                **all_props[i]
            )
            processed_text_elements.append({'element': element, 'page_idx': item['page_idx'], 'class_name': item['class_name']})

    return processed_text_elements


def structure_page_data(batch_paths, batch_images_rgb, all_bubbles_by_page, processed_text_elements):
    """탐지된 정보들을 기반으로 최종 PageData 객체 리스트를 구성합니다."""
    batch_page_data = []
    for page_idx, path in enumerate(batch_paths):
        page_data = PageData(source_page=os.path.basename(path), image_rgb=batch_images_rgb[page_idx])
        page_bubbles = all_bubbles_by_page[page_idx]

        page_text_elements = [item for item in processed_text_elements if item['page_idx'] == page_idx and item['class_name'] == 'text']
        page_free_texts = [item['element'] for item in processed_text_elements if item['page_idx'] == page_idx and item['class_name'] == 'free_text']
        
        unmatched_text_indices = set(range(len(page_text_elements)))
        unmatched_free_text_indices = set(range(len(page_free_texts)))

        for bubble_box in page_bubbles:
            bubble_center = ((bubble_box[0] + bubble_box[2]) / 2, (bubble_box[1] + bubble_box[3]) / 2)
            
            closest_text_idx, min_distance = -1, float('inf')
            for j, text_info in enumerate(page_text_elements):
                if j in unmatched_text_indices:
                    text_center = ((text_info['element'].text_box[0] + text_info['element'].text_box[2]) / 2, (text_info['element'].text_box[1] + text_info['element'].text_box[3]) / 2)
                    distance = np.sqrt((bubble_center[0] - text_center[0]) ** 2 + (bubble_center[1] - text_center[1]) ** 2)
                    if distance < min_distance:
                        min_distance, closest_text_idx = distance, j
            
            matched_element = None
            if closest_text_idx != -1:
                matched_element = page_text_elements[closest_text_idx]['element']
                unmatched_text_indices.remove(closest_text_idx)
            else:
                found_free_text_idx = -1
                for j in unmatched_free_text_indices:
                    free_text_element = page_free_texts[j]
                    if is_box_inside(free_text_element.text_box, bubble_box):
                        found_free_text_idx = j
                        break
                if found_free_text_idx != -1:
                    matched_element = page_free_texts[found_free_text_idx]
                    unmatched_free_text_indices.remove(found_free_text_idx)

            if matched_element:
                b = bubble_box
                cropped_bubble_rgb = page_data.image_rgb[b[1]:b[3], b[0]:b[2]]
                attachment = check_bubble_attachment(cropped_bubble_rgb)
                
                speech_bubble = SpeechBubble(
                    bubble_box=b.tolist(),
                    text_element=matched_element,
                    attachment=attachment
                )
                page_data.speech_bubbles.append(speech_bubble)

        page_data.freeform_texts = [page_free_texts[i] for i in sorted(list(unmatched_free_text_indices))]
        batch_page_data.append(page_data)

    return batch_page_data


def process_image_batch(models, batch_images_rgb, batch_paths):
    """이미지 배치에서 텍스트, 말풍선 등의 정보를 추출하여 PageData 리스트로 반환합니다."""
    # 1. 객체 탐지
    all_text_items, all_bubbles_by_page = detect_objects(models['detection'], batch_images_rgb)

    # 2. 텍스트 박스 병합
    merged_text_items = merge_text_boxes(all_text_items)
    
    # 3. 텍스트 속성 추출 (OCR, 폰트)
    processed_text_elements = extract_text_properties(models, batch_images_rgb, merged_text_items, batch_paths)

    # 4. 최종 데이터 구조화
    batch_page_data = structure_page_data(batch_paths, batch_images_rgb, all_bubbles_by_page, processed_text_elements)

    return batch_page_data