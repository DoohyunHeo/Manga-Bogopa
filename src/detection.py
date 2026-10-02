"""말풍선·텍스트 영역 탐지 (YOLO, 놓친 자리만 RT-DETR로 보완) + 겹치는 박스 병합."""
import logging
from collections import defaultdict, deque

import numpy as np
import torch
from tqdm import tqdm

from src import config
from src.utils import calculate_iou, merge_boxes

logger = logging.getLogger(__name__)

RTDETR_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
# RT-DETR 분류 번호 → 파이프라인 분류 (말풍선 / 말풍선 안 글자 / 말풍선 밖 글자)
_RTDETR_CLASSES = {0: "bubble", 1: "text", 2: "free_text"}


class RTDetrDetector:
    """ogkalu RT-DETR-v2 글자 탐지기 — 학습 해상도 640 정사각으로 맞춰 추론하고 원본 좌표로 되돌린다."""

    def __init__(self, device):
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        self.device = device
        self.processor = RTDetrImageProcessor.from_pretrained(RTDETR_MODEL_ID, size={"width": 640, "height": 640})
        self.model = RTDetrV2ForObjectDetection.from_pretrained(RTDETR_MODEL_ID).to(device).eval()

    def predict(self, images_rgb, threshold):
        """페이지마다 [(분류, xyxy, 점수)] 목록."""
        inputs = self.processor(images=list(images_rgb), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        sizes = torch.tensor([img.shape[:2] for img in images_rgb], device=self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=threshold)
        pages = []
        for img, result in zip(images_rgb, results):
            h, w = img.shape[:2]
            boxes = result["boxes"].cpu().numpy().clip(0, [w, h, w, h])
            pages.append([(_RTDETR_CLASSES.get(int(label)), box, float(score))
                          for box, label, score in zip(boxes, result["labels"].cpu().tolist(),
                                                       result["scores"].cpu().tolist())])
        return pages


class HybridDetector:
    """기본 모델(YOLO) 결과는 그대로 두고, 기본 모델이 아무것도 못 찾은 자리만 RT-DETR 결과로 채운다.

    두 모델의 박스 모양이 달라(RT-DETR은 이름+내용을 한 덩어리로 잡는 등) 같은 대사를 두 번
    번역하지 않도록, 기본 모델 글자 박스와 조금이라도 겹치는 RT-DETR 글자 박스는 버린다.
    """

    # 겹침 = 교집합 / 두 박스 중 작은 쪽 넓이
    TEXT_OVERLAP_LIMIT = 0.2
    BUBBLE_OVERLAP_LIMIT = 0.3

    def __init__(self, yolo_model, rtdetr):
        self.yolo = yolo_model
        self.rtdetr = rtdetr


def _overlap_ratio(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


# 보완 상자가 기본 상자와 같은 한 줄(한 단)을 끝쪽으로 더 길게 잡았는지: 둘 다 길쭉하고(긴 변이 짧은 변의
# _LINE_ASPECT배 이상), 줄 굵기 쪽 양 끝이 굵기의 _LINE_SLACK배 안으로 같고, 읽는 방향의 끝(가로는 오른쪽, 세로는 아래)
# 너머로 굵기의 _LINE_GROW_MIN배 이상 더 나갈 때. 줄 앞쪽으로는 늘리지 않는다 — 앞쪽에는 잡지 로고·제목 딱지·아이콘이
# 붙어 있는 일이 많아 앞쪽으로 늘리면 대개 다른 글을 묶는다
_LINE_ASPECT = 3.0
_LINE_SLACK = 0.3
_LINE_GROW_MIN = 0.5


def _line_extension(base, extra):
    """보완 상자가 기본 상자의 줄을 끝쪽으로 더 길게 잡았으면 늘린 상자, 아니면 None — 기본 모델이 긴 해설 줄을 중간에서
    끊으면 뒷부분이 지워지지도 번역되지도 않고 남는다."""
    if base["class_name"] != extra["class_name"]:
        return None
    ax1, ay1, ax2, ay2 = (float(v) for v in base["box"][:4])
    bx1, by1, bx2, by2 = (float(v) for v in extra["box"][:4])
    width, height = ax2 - ax1, ay2 - ay1
    if width >= _LINE_ASPECT * height:
        if abs(ay1 - by1) > _LINE_SLACK * height or abs(ay2 - by2) > _LINE_SLACK * height:
            return None
        if bx2 - ax2 < _LINE_GROW_MIN * height:
            return None
        grown = (ax1, min(ay1, by1), bx2, max(ay2, by2))
    elif height >= _LINE_ASPECT * width:
        if abs(ax1 - bx1) > _LINE_SLACK * width or abs(ax2 - bx2) > _LINE_SLACK * width:
            return None
        if by2 - ay2 < _LINE_GROW_MIN * width:
            return None
        grown = (min(ax1, bx1), ay1, max(ax2, bx2), by2)
    else:
        return None
    return np.array(grown, dtype=base["box"].dtype)


def _detect_hybrid(detector, batch_images_rgb):
    texts, bubbles = detect_objects(detector.yolo, batch_images_rgb)
    extra_texts, extra_bubbles = _detect_rtdetr(detector.rtdetr, batch_images_rgb)
    added = extended = 0
    for item in extra_texts:
        base = [t for t in texts if t["page_idx"] == item["page_idx"]]
        hits = [t for t in base if _overlap_ratio(item["box"], t["box"]) >= HybridDetector.TEXT_OVERLAP_LIMIT]
        if not hits:
            item["supplement"] = True  # 기본 모델이 못 본 자리 — 읽기 단계에서 한 번 더 따진다
            texts.append(item)
            added += 1
        elif len(hits) == 1 and not hits[0].get("supplement"):
            grown = _line_extension(hits[0], item)
            if grown is not None:  # 기본 모델이 같은 줄을 짧게 잡았다 — 끝쪽으로 늘린다
                hits[0]["box"] = grown
                extended += 1
    if extended:
        logger.info(f"보완 탐지: 기본 모델이 짧게 잡은 글줄 {extended}개를 늘렸습니다.")
    for page_idx, page_bubbles in enumerate(extra_bubbles):
        for bubble in page_bubbles:
            if all(_overlap_ratio(bubble, b) < HybridDetector.BUBBLE_OVERLAP_LIMIT for b in bubbles[page_idx]):
                bubbles[page_idx].append(bubble)
    logger.info(f"보완 탐지: 기본 모델이 놓친 글자 {added}개를 더했습니다.")
    return texts, bubbles


def detect_objects(detection_model, batch_images_rgb):
    """Run object detection on a batch of pages.

    - 추론 해상도는 모델 학습 해상도(DETECTION_IMGSZ)에 맞춘다. 기본 640으로 돌리면
      작은 글자 박스가 대량으로 누락된다.
    - ultralytics는 numpy 입력을 BGR로 가정하므로 RGB 페이지를 뒤집어 전달한다.
    - VRAM 보호를 위해 DETECTION_BATCH_SIZE 단위로 나눠 추론한다.
    """
    logger.info(f"Detecting objects for {len(batch_images_rgb)} pages...")
    if isinstance(detection_model, HybridDetector):
        return _detect_hybrid(detection_model, batch_images_rgb)
    imgsz = max(32, int(config.DETECTION_IMGSZ))
    use_half = config.DEVICE == "cuda"
    det_bs = max(1, int(config.DETECTION_BATCH_SIZE))

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
                        "score": float(box.conf[0]),
                    })
    return all_text_items, all_bubbles_by_page


def _detect_rtdetr(detector, batch_images_rgb):
    """RT-DETR 경로 — YOLO 경로와 같은 형식(글자 항목 목록, 페이지별 말풍선)으로 돌려준다."""
    det_bs = max(1, int(config.DETECTION_BATCH_SIZE))
    all_text_items = []
    all_bubbles_by_page = [[] for _ in batch_images_rgb]
    for start in tqdm(range(0, len(batch_images_rgb), det_bs), desc="Detection"):
        chunk = batch_images_rgb[start:start + det_bs]
        for offset, boxes in enumerate(detector.predict(chunk, config.YOLO_CONF_THRESHOLD)):
            page_idx = start + offset
            for class_name, coords, score in boxes:
                if class_name == "bubble":
                    all_bubbles_by_page[page_idx].append(coords.astype(int))
                elif class_name in ("text", "free_text"):
                    all_text_items.append({"page_idx": page_idx, "box": coords, "class_name": class_name,
                                           "score": score})
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
                final_items.append({
                    "page_idx": page_idx, "box": merged_box, "class_name": class_name,
                    "score": max(item.get("score", 1.0) for item in cluster_items),
                    # 기본 모델 박스가 하나라도 섞이면 보완 탐지로 보지 않는다
                    "supplement": all(item.get("supplement", False) for item in cluster_items),
                })
            else:
                final_items.append(items[component[0]])

    logger.info(f"Box merge reduced the set to {len(final_items)} objects.")
    return final_items
