import os
from collections import defaultdict
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from collections import defaultdict
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from src import config
from src.utils import Letterbox, calculate_iou, merge_boxes, is_box_inside
from src.data_models import PageData, TextElement, SpeechBubble


def check_bubble_attachment(cropped_bubble_image_rgb):
    """잘라낸 말풍선 이미지를 분석하여 말꼬리의 방향을 판단합니다."""
    try:
        img = cropped_bubble_image_rgb
        img_h, img_w = img.shape[:2]
        scan_width = int(img_w * 0.10)
        if scan_width == 0: return 'none'

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

        if has_right_vertical_line and not has_left_vertical_line: return 'right'
        if has_left_vertical_line and not has_right_vertical_line: return 'left'
        return 'none'
    except Exception:
        return 'none'


def process_image_batch(models, batch_images_rgb, batch_paths):
    """이미지 배치에서 텍스트, 말풍선 등의 정보를 추출하여 PageData 리스트로 반환합니다."""
    detection_model = models['detection']
    ocr_model = models['ocr']
    font_classifier_model = models['font_classifier']
    font_size_model = models['font_size']

    tqdm.write(f"-> {len(batch_images_rgb)}개 페이지 일괄 탐지 중...")
    batch_results = detection_model(batch_images_rgb, conf=config.YOLO_CONF_THRESHOLD, verbose=False)

    all_items_to_process = []
    all_bubbles_by_page = [[] for _ in batch_images_rgb]
    for page_idx, results in enumerate(tqdm(batch_results, desc="Detection")):
        for box in results.boxes:
            class_name = results.names[int(box.cls[0])]
            coords = box.xyxy[0].cpu().numpy()
            if class_name == 'bubble':
                all_bubbles_by_page[page_idx].append(coords.astype(int))
            elif class_name in ['text', 'free_text']:
                all_items_to_process.append({
                    'page_idx': page_idx, 'box': coords, 'class_name': class_name
                })

    tqdm.write(f"-> {len(all_items_to_process)}개의 탐지된 객체에 대해 중복 박스 병합 처리 중...")
    grouped_items = defaultdict(list)
    for item in all_items_to_process:
        grouped_items[(item['page_idx'], item['class_name'])].append(item)

    final_items = []
    for (page_idx, class_name), items in grouped_items.items():
        if len(items) < 2:
            final_items.extend(items)
            continue

        # IoU 기반 클러스터링으로 겹치는 박스 병합
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
                q = [i]
                visited[i] = True
                while q:
                    u = q.pop(0)
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

    all_items_to_process = final_items
    tqdm.write(f"-> 병합 후 {len(all_items_to_process)}개의 객체로 정리되었습니다.")

    if not all_items_to_process: # 텍스트가 전혀 없으면 빈 PageData 반환
        return [PageData(source_page=os.path.basename(p), image_rgb=img) for p, img in zip(batch_paths, batch_images_rgb)]

    # OCR 및 폰트 분석을 위한 크롭 이미지 준비
    crops_for_ocr = []
    for item in all_items_to_process:
        image_rgb = batch_images_rgb[item['page_idx']]
        coords = item['box'].astype(int)
        crop_pil = Image.fromarray(image_rgb[coords[1]:coords[3], coords[0]:coords[2]])
        item['crop'] = crop_pil
        crops_for_ocr.append(crop_pil)

        if config.SAVE_DEBUG_CROPS:
            try:
                page_name = os.path.splitext(os.path.basename(batch_paths[item['page_idx']]))[0]
                x1, y1, x2, y2 = coords
                crop_filename = f"{page_name}_{item['class_name']}_{x1}_{y1}_{x2}_{y2}.png"
                crop_path = os.path.join(config.DEBUG_CROPS_DIR, crop_filename)
                crop_pil.save(crop_path)
            except Exception as e:
                tqdm.write(f"경고: 디버그 크롭 저장 실패 {crop_path}: {e}")

    tqdm.write(f"-> {len(crops_for_ocr)}개의 텍스트 조각을 Batch OCR 처리 중...")
    all_ocr_results = ocr_model(crops_for_ocr)

    # 폰트 스타일/크기/각도 예측
    all_props = []
    if font_classifier_model and font_size_model:
        tqdm.write(f"-> {len(all_items_to_process)}개의 텍스트 조각을 폰트 모델로 분석 중...")
        transform = transforms.Compose([
            Letterbox(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        font_batch_size = config.FONT_MODEL_BATCH_SIZE
        for i in tqdm(range(0, len(all_items_to_process), font_batch_size), desc="Font Model"):
            batch_items = all_items_to_process[i:i+font_batch_size]
            image_tensors = torch.stack([transform(item['crop'].convert("RGB")) for item in batch_items]).to(config.DEVICE)
            
            with torch.no_grad():
                outputs_classifier = font_classifier_model(image_tensors)
                pred_sizes = font_size_model(image_tensors)
                pred_angles = outputs_classifier['angle'].cpu().numpy()
                pred_style_indices = torch.argmax(outputs_classifier['style'], dim=1).cpu().numpy()
                pred_sizes_np = pred_sizes.cpu().numpy()

            for j in range(len(batch_items)):
                style_idx = int(pred_style_indices[j])
                style_name = font_classifier_model.style_mapping.get(style_idx, 'standard')
                all_props.append({
                    'font_size': int(round(pred_sizes_np[j])),
                    'angle': int(pred_angles[j]),
                    'font_style': style_name
                })
    else:
        all_props = [{'font_size': 20, 'angle': 0.0, 'font_style': 'standard'}] * len(all_items_to_process)

    # 최종적으로 처리된 TextElement 객체 생성
    processed_text_elements = []
    for i, item in enumerate(all_items_to_process):
        if all_ocr_results[i]:
            element = TextElement(
                text_box=item['box'].tolist(),
                original_text=all_ocr_results[i],
                **all_props[i]
            )
            processed_text_elements.append({'element': element, 'page_idx': item['page_idx'], 'class_name': item['class_name']})

    # PageData 객체 구성
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
            
            # 1. 'text' 클래스와의 거리 기반 매칭
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
                # 2. 매칭되는 'text'가 없으면, 버블 안에 포함되는 'free_text' 탐색
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

        # 남은 free_text들을 page_data에 추가
        page_data.freeform_texts = [page_free_texts[i] for i in sorted(list(unmatched_free_text_indices))]
        batch_page_data.append(page_data)

    return batch_page_data
