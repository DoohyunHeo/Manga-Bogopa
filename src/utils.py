import numpy as np
from PIL import Image


class Letterbox:
    """YOLO 모델의 입력 이미지 크기를 조절하는 유틸리티 클래스"""
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
