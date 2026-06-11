"""말풍선·텍스트 영역 탐지 (YOLO) + 겹치는 박스 병합."""
import logging
from collections import defaultdict, deque

import numpy as np
from tqdm import tqdm

from src import config
from src.utils import calculate_iou, merge_boxes

logger = logging.getLogger(__name__)


def detect_objects(detection_model, batch_images_rgb):
    """Run object detection on a batch of pages.

    - 추론 해상도는 모델 학습 해상도(DETECTION_IMGSZ)에 맞춘다. 기본 640으로 돌리면
      작은 글자 박스가 대량으로 누락된다.
    - ultralytics는 numpy 입력을 BGR로 가정하므로 RGB 페이지를 뒤집어 전달한다.
    - VRAM 보호를 위해 DETECTION_BATCH_SIZE 단위로 나눠 추론한다.
    """
    logger.info(f"Detecting objects for {len(batch_images_rgb)} pages...")
    imgsz = max(32, int(getattr(config, "DETECTION_IMGSZ", 1344)))
    use_half = bool(getattr(config, "DETECTION_HALF", True)) and config.DEVICE == "cuda"
    det_bs = max(1, int(getattr(config, "DETECTION_BATCH_SIZE", 8)))

    all_text_items = []
    all_bubbles_by_page = [[] for _ in batch_images_rgb]
    for start in tqdm(range(0, len(batch_images_rgb), det_bs), desc="Detection"):
        chunk_bgr = [
            np.ascontiguousarray(img[..., ::-1])
            for img in batch_images_rgb[start:start + det_bs]
        ]
        batch_results = detection_model(
            chunk_bgr,
            conf=config.YOLO_CONF_THRESHOLD,
            imgsz=imgsz,
            half=use_half,
            verbose=False,
        )
        for offset, results in enumerate(batch_results):
            page_idx = start + offset
            for box in results.boxes:
                class_name = results.names[int(box.cls[0])]
                coords = box.xyxy[0].cpu().numpy()
                if class_name == "bubble":
                    all_bubbles_by_page[page_idx].append(coords.astype(int))
                elif class_name in ["text", "free_text"]:
                    all_text_items.append({
                        "page_idx": page_idx,
                        "box": coords,
                        "class_name": class_name,
                    })
    return all_text_items, all_bubbles_by_page


def merge_text_boxes(text_items):
    """Merge overlapping text boxes."""
    logger.info(f"Merging overlapping boxes from {len(text_items)} detected text objects...")
    grouped_items = defaultdict(list)
    for item in text_items:
        grouped_items[(item["page_idx"], item["class_name"])].append(item)

    final_items = []
    for (page_idx, class_name), items in grouped_items.items():
        if len(items) < 2:
            final_items.extend(items)
            continue

        num_items = len(items)
        adj_matrix = np.zeros((num_items, num_items))
        for i in range(num_items):
            for j in range(i + 1, num_items):
                iou = calculate_iou(items[i]["box"], items[j]["box"])
                if iou > config.TEXT_MERGE_OVERLAP_THRESHOLD:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1

        visited = [False] * num_items
        for i in range(num_items):
            if visited[i]:
                continue

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
                merged_box = merge_boxes([item["box"] for item in cluster_items])
                final_items.append({"page_idx": page_idx, "box": merged_box, "class_name": class_name})
            else:
                final_items.append(items[component[0]])

    logger.info(f"Box merge reduced the set to {len(final_items)} objects.")
    return final_items
