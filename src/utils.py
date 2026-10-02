import os
import time

import cv2
import numpy as np
from PIL import Image


PAGE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def replace_file(src: str, dst: str, attempts: int = 20, wait_sec: float = 0.05):
    """os.replace와 같지만, Windows에서 누가 dst를 잠깐 열어 두고 있으면(웹 화면이 상태를 읽는 중,
    백신 검사 등) 조금 기다렸다 다시 해 본다. 끝내 안 되면 마지막 오류를 그대로 낸다."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait_sec)


def list_page_images(folder):
    """원고 폴더에서 쪽으로 칠 그림 파일만 이름순으로 — 하위 폴더, Thumbs.db 같은 파일,
    '.'으로 시작하는 숨김 파일(맥이 압축에 끼워 넣는 '._001.jpg' 등)은 뺀다."""
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, name) for name in os.listdir(folder)
        if not name.startswith(".") and name.lower().endswith(PAGE_IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, name))
    )


def read_image_bgr(path):
    """cv2.imread와 같지만 한글·일본어 등이 든 경로도 읽는다 (실패하면 None)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image_bgr(path, image_bgr) -> bool:
    """cv2.imwrite와 같지만 비ASCII 경로도 쓰고, 성공 여부를 돌려준다."""
    ok, buffer = cv2.imencode(os.path.splitext(path)[1] or ".png", image_bgr)
    if not ok:
        return False
    try:
        buffer.tofile(path)
    except OSError:
        return False
    return True


class Letterbox:
    """글씨체 모델 입력 크기에 맞게 줄이고(키우지는 않는다) 남는 곳을 회색으로 채워 가운데에 놓는다."""
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


def calculate_iou(box_a, box_b):
    """두 바운딩 박스(x1, y1, x2, y2)의 IoU를 계산합니다."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    denominator = float(box_a_area + box_b_area - inter_area)
    if denominator == 0:
        return 0.0

    iou = inter_area / denominator
    return iou


def merge_boxes(boxes):
    """여러 바운딩 박스를 모두 포함하는 가장 작은 단일 박스를 반환합니다."""
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return np.array([min_x, min_y, max_x, max_y])


def rects_intersect(rect1, rect2):
    """두 사각형(x1, y1, x2, y2)이 겹치는지 확인하는 함수"""
    return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[3] < rect2[1] or rect1[1] > rect2[3])


def is_box_inside(inner_box, outer_box):
    """두 바운딩 박스(x1, y1, x2, y2)에 대해 inner_box가 outer_box 내부에 완전히 포함되는지 확인합니다."""
    return inner_box[0] >= outer_box[0] and \
           inner_box[1] >= outer_box[1] and \
           inner_box[2] <= outer_box[2] and \
           inner_box[3] <= outer_box[3]
