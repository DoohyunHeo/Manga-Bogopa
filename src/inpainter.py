import logging
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

from src import config
from src.data_models import PageData
from src.font_attributes import measure_rim, measure_text_look, rim_window_px


def _clip_coords_to_image(image, coords):
    """텍스트 박스를 이미지 경계로 자르고, 유효하지 않으면 None을 반환합니다."""
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = coords
    x1 = max(0, min(img_w, int(np.floor(x1))))
    y1 = max(0, min(img_h, int(np.floor(y1))))
    x2 = max(0, min(img_w, int(np.ceil(x2))))
    y2 = max(0, min(img_h, int(np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# 잔여 잉크 흡수 휴리스틱 (내부 상수)
_ABSORB_REACH_PX = 12          # 글자 마스크에서 이 거리 안에 닿아 있는 잉크를 후보로
_ABSORB_MAX_AREA_RATIO = 0.35  # 텍스트 박스 면적 대비 이보다 큰 덩어리는 그림으로 간주
# 아래가 얼굴로 열린 말풍선은 말풍선 박스가 얼굴까지 덮어, 글자 곁의 눈·머리카락 선이 흡수되어 지워진다.
# 글자 모양 모델이 글자로 본 픽셀이 거의 없는 덩어리는 대개 머리카락·눈·스크린톤이고, 말풍선 크기에 가까운
# 덩어리는 이중 테두리의 안쪽 선이라 아래 네 조건으로 흡수에서 뺀다.
_ABSORB_CHECK_MIN_AREA = 20        # 이 넓이 이상 덩어리는 글자 모양 모델로 글자인지 한 번 더 본다
_ABSORB_MIN_TEXT_RATIO = 0.3       # 덩어리 잉크 중 글자로 본 픽셀이 이 비율에 못 미치면 그림으로 두고 흡수하지 않는다
_ABSORB_MAX_SPAN_RATIO = 0.8       # 말풍선 가로·세로의 이 비율을 넘게 뻗은 덩어리는 테두리 선으로 보고 둔다
_ABSORB_MIN_INTERIOR_RATIO = 0.5   # 덩어리 둘레 흰 바탕 중 말풍선 속 흰 바탕이 이 비율에 못 미치면 흡수하지 않는다


def _absorb_residual_ink(image_rgb, full_page_mask, text_box, bubble_box, glyph=None):
    """말풍선 안에서 탐지 박스 곁에 붙은 미탐지 글자 조각을 마스크로 흡수합니다.

    후리가나 줄, 첨자, 박스를 살짝 벗어난 글자 끝처럼 탐지 박스 밖에 남은
    작은 잉크 덩어리가 인페인팅 후 검은 노이즈로 남는 것을 막는다.
    - 후보: 텍스트 마스크를 _ABSORB_REACH_PX 만큼 넓힌 영역과 겹치는 잉크 성분
    - 제외: 말풍선 크롭 가장자리에 닿는 성분(테두리 선), 큰 성분(그림), 말풍선만큼 뻗은 성분(안쪽 테두리),
      glyph(말풍선 크롭 크기 글자 모양 마스크)가 오면 글자로 본 픽셀이 적은 _ABSORB_CHECK_MIN_AREA 이상 성분(그림 선)
    """
    img_h, img_w = image_rgb.shape[:2]
    bx1 = max(0, int(bubble_box[0]))
    by1 = max(0, int(bubble_box[1]))
    bx2 = min(img_w, int(bubble_box[2]))
    by2 = min(img_h, int(bubble_box[3]))
    if bx2 - bx1 < 8 or by2 - by1 < 8:
        return

    crop = image_rgb[by1:by2, bx1:bx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    mask_crop = full_page_mask[by1:by2, bx1:bx2]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_ABSORB_REACH_PX * 2 + 1, _ABSORB_REACH_PX * 2 + 1)
    )
    # 이 말풍선의 글자 박스 곁만 본다 — 페이지 마스크 전체로 보면 이웃 박스 곁의 톤·그림까지 흡수했다
    tx1, ty1 = max(0, int(text_box[0]) - bx1), max(0, int(text_box[1]) - by1)
    tx2, ty2 = min(bx2 - bx1, int(np.ceil(text_box[2])) - bx1), min(by2 - by1, int(np.ceil(text_box[3])) - by1)
    if tx2 <= tx1 or ty2 <= ty1:
        return
    own = np.zeros_like(mask_crop)
    own[ty1:ty2, tx1:tx2] = 255
    seed = cv2.dilate(own, kernel)

    box_area = max(1.0, float((text_box[2] - text_box[0]) * (text_box[3] - text_box[1])))
    max_component_area = box_area * _ABSORB_MAX_AREA_RATIO

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    crop_h, crop_w = ink.shape
    if glyph is not None and glyph.shape == ink.shape:
        glyph = cv2.dilate(glyph.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # 가는 획 가장자리까지
    else:
        glyph = None
    # 말풍선 속 흰 바탕 = 글자 박스 안 흰 픽셀이 가장 많이 속한 흰 영역. 테두리 밖 톤 점·그림 선은 다른 흰 영역에 둘러싸인다
    _, white_labels = cv2.connectedComponents((ink == 0).astype(np.uint8), connectivity=4)
    inside = white_labels[ty1:ty2, tx1:tx2]
    inside = inside[inside > 0]
    interior = int(np.bincount(inside).argmax()) if inside.size else 0
    ring_kernel = np.ones((5, 5), np.uint8)
    absorbed = np.zeros_like(ink)
    absorbed_any = False
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area > max_component_area:
            continue
        # 말풍선 테두리 선은 크롭 가장자리에 닿으므로 제외
        if x <= 1 or y <= 1 or x + w >= crop_w - 1 or y + h >= crop_h - 1:
            continue
        if w > _ABSORB_MAX_SPAN_RATIO * crop_w or h > _ABSORB_MAX_SPAN_RATIO * crop_h:
            continue  # 이중 테두리의 안쪽 선
        component = labels == label
        if not (seed[component] > 0).any():
            continue
        if (mask_crop[component] > 0).all():
            continue  # 이미 전부 마스크 안
        if glyph is not None and area >= _ABSORB_CHECK_MIN_AREA \
                and glyph[component].mean() < _ABSORB_MIN_TEXT_RATIO:
            continue  # 글자 곁의 그림 선(열린 말풍선 아래 얼굴의 눈·머리카락)
        if interior:
            ring = cv2.dilate(component.astype(np.uint8), ring_kernel) > 0
            ring_white = white_labels[ring & ~component]
            ring_white = ring_white[ring_white > 0]
            if ring_white.size and (ring_white == interior).mean() < _ABSORB_MIN_INTERIOR_RATIO:
                continue  # 말풍선 속 흰 바탕에 떠 있지 않은 덩어리 — 테두리 밖 톤·그림
        absorbed[component] = 255
        absorbed_any = True

    if not absorbed_any:
        return

    # 흡수된 조각 주변 약간의 여유 (안티앨리어싱 가장자리) — 단, 흡수분만 팽창시키고
    # 말풍선 테두리 보호 링(바깥 2px)은 건드리지 않는다.
    absorbed = cv2.dilate(absorbed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    absorbed[:2, :] = 0
    absorbed[-2:, :] = 0
    absorbed[:, :2] = 0
    absorbed[:, -2:] = 0
    mask_crop[absorbed > 0] = 255


# 말풍선 밖 글자 지우기 — 박스 대신 글자 모양으로 지워 글자 뒤 그림·톤·먹칠을 살린다.
# 박스로 지우면 검은 라벨이 회색으로 뜨고 톤·칸 테두리가 뭉개진다.
_GLYPH_DILATE_PX = 5                 # 글자 마스크를 이만큼 넓힌다 — 3px는 테두리 흔적과 얼룩이 남았다
_MIN_GLYPH_RATIO = 0.01              # 마스크가 박스의 이 비율에 못 미치면 글자를 못 찾은 것으로 보고 박스로 지운다
# 모델이 놓친 곁 조각(후리가나·줄표·기호·테두리 획)을 마스크에 더하는 기준
_GLYPH_ABSORB_REACH_PX = 8           # 넓힌 마스크에서 이 거리 안에 닿는 덩어리만
_GLYPH_ABSORB_MIN_AREA = 8           # 이보다 작은 덩어리는 스크린톤 점으로 보고 둔다
_GLYPH_ABSORB_MAX_AREA_RATIO = 0.35  # 박스 넓이의 이 비율을 넘는 덩어리는 그림·먹칠 바탕으로 보고 둔다


def _bubble_crop_box(image_rgb, bubble_box):
    """_absorb_residual_ink가 보는 말풍선 크롭과 같은 (x1, y1, x2, y2), 너무 작으면 None."""
    img_h, img_w = image_rgb.shape[:2]
    x1, y1 = max(0, int(bubble_box[0])), max(0, int(bubble_box[1]))
    x2, y2 = min(img_w, int(bubble_box[2])), min(img_h, int(bubble_box[3]))
    return (x1, y1, x2, y2) if x2 - x1 >= 8 and y2 - y1 >= 8 else None


def _absorb_freeform_ink(crop, mask):
    """글자 마스크 곁의 작은 먹·흰 덩어리를 마스크에 더한다.

    박스 밝기를 OTSU로 둘로 나눠 어두운 덩어리와 밝은 덩어리를 모두 본다 (검은 글자·흰 글자·테두리 글자).
    박스 가장자리에 닿는 덩어리(박스를 가로지르는 그림 선, 바탕 종이·먹칠)는 건드리지 않는다.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    reach = 2 * _GLYPH_ABSORB_REACH_PX + 1
    seed = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)))
    added = np.zeros_like(mask)
    for candidates in (dark, 255 - dark):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
        for label in range(1, num_labels):
            x, y, bw, bh, area = stats[label]
            if area < _GLYPH_ABSORB_MIN_AREA or area > _GLYPH_ABSORB_MAX_AREA_RATIO * h * w:
                continue
            if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
                continue
            component = labels == label
            if not (seed[component] > 0).any() or (mask[component] > 0).all():
                continue
            added[component] = 255
    if not added.any():
        return mask
    contours, _ = cv2.findContours(added, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(added, contours, -1, 255, thickness=cv2.FILLED)  # 테두리 글자의 속까지
    return np.maximum(mask, cv2.dilate(added, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))))


# 말풍선 밖 글자 상자 곁(글자 크기의 이 배 안)에 남은 글자 조각 — 탐지 상자가 첫 글자나 후리가나, 따로 떨어진
# 기호를 덮지 못하면 원문이 남는다. 글자 모양 모델이 글자로 본 덩어리 가운데 지금 마스크에서
# 글자 크기의 _HALO_REACH배 안에 닿고, 다른 글자 상자와 겹치지 않고, 한 글자 넓이 남짓 이하이며, 살펴본 영역 가장자리에
# 닿지 않는(곁 띠·다른 글줄로 이어지지 않는) 것만 더한다 — 번역 제외된 옆 글자를 지우지 않게
_HALO_RATIO = 1.0
_HALO_REACH = 0.3
_HALO_MAX_AREA = 1.5  # (글자 크기)² 의 이 배


def _absorb_freeform_halo(image_rgb, full_page_mask, box, font_size, text_mask_model, keep_out):
    """상자 곁 글자 조각을 마스크에 더하고, 넓어진 지울 영역 (x1, y1, x2, y2)을 돌려준다. 더한 게 없으면 None."""
    if text_mask_model is None:
        return None
    x1, y1, x2, y2 = box
    size = float(font_size or (y2 - y1))
    margin = max(4, int(round(_HALO_RATIO * size)))
    img_h, img_w = image_rgb.shape[:2]
    hx1, hy1, hx2, hy2 = max(0, x1 - margin), max(0, y1 - margin), min(img_w, x2 + margin), min(img_h, y2 + margin)
    current = full_page_mask[hy1:hy2, hx1:hx2] > 0
    if not current.any():
        return None
    glyph = text_mask_model([image_rgb[hy1:hy2, hx1:hx2]])[0]
    if glyph is None:
        return None
    reach = max(3, int(round(_HALO_REACH * size)))
    near = cv2.dilate(current.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)) > 0
    blocked = np.zeros_like(current)
    for ox1, oy1, ox2, oy2 in keep_out:
        bx1, by1, bx2, by2 = max(0, int(ox1) - hx1), max(0, int(oy1) - hy1), int(ox2) - hx1, int(oy2) - hy1
        if bx2 > bx1 and by2 > by1:
            blocked[by1:max(by1, by2), bx1:max(bx1, bx2)] = True
    count, labels, stats, _ = cv2.connectedComponentsWithStats((glyph & ~current).astype(np.uint8), connectivity=8)
    added = np.zeros_like(current)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] > _HALO_MAX_AREA * size * size:
            continue
        left, top, width, height = stats[label, :4]
        if left <= 0 or top <= 0 or left + width >= hx2 - hx1 or top + height >= hy2 - hy1:
            continue
        component = labels == label
        if near[component].any() and not blocked[component].any():
            added |= component
    if not added.any():
        return None
    added = cv2.dilate(added.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    region = full_page_mask[hy1:hy2, hx1:hx2]
    region[added] = 255
    ys, xs = np.nonzero(added)
    return (min(x1, hx1 + int(xs.min())), min(y1, hy1 + int(ys.min())),
            max(x2, hx1 + int(xs.max()) + 1), max(y2, hy1 + int(ys.max()) + 1))


# 작은 이름표·네모 말풍선 — 글자 상자가 흰 바탕을 거의 다 덮으면 지울 구멍 둘레가 테두리와 그 밖의 먹·톤뿐이라 지우기
# 모델이 흰 바탕을 검게·톤으로 채우고, 기울어진 테두리까지 지운다.
# 흰 바탕(속 글자 포함) 윤곽의 이 비율 넘게 지울 때만: 지우기를 흰 바탕 윤곽(+몇 px) 안으로 줄여 테두리를 두고,
# 지운 자리는 흰 바탕색으로 칠한다
_PAPER_LIGHT = 170          # 이 밝기 이상 = 흰 바탕
_PAPER_COVERED = 0.9        # 흰 바탕 윤곽 중 지우는 비율이 이 이상일 때만
_PAPER_MIN_LIGHT = 0.55     # 흰 바탕 윤곽 중 흰 픽셀 비율 하한 (나머지는 글자)
_PAPER_MAX_DOTS = 0.02      # 윤곽 안 작은 먹 점(스크린톤) 픽셀 비율 상한
_PAPER_CROP_PAD = 8         # 흰 바탕을 찾을 때 말풍선 상자 밖으로 더 보는 폭
_PAPER_KEEP_PX = 1          # 흰 바탕 윤곽에서 이만큼 넓힌 곳까지만 지운다 (글자 가장자리 안티에일리어싱)


def _bubble_paper_fill(image_rgb, full_page_mask, text_box, bubble_box):
    """흰 바탕을 거의 다 덮는 말풍선: 지울 곳을 흰 바탕 안으로 줄이고 (칠할 픽셀 bool, 색)을 돌려준다. 아니면 None."""
    # 말풍선 상자를 조금 넓혀 본다 — 네모 상자(줄거리 칸)는 탐지 상자가 테두리 안쪽에 딱 붙어 흰 바탕이 상자 끝에
    # 닿으면 닫힌 말풍선이 아닌 것으로 보여 칠하지 않게 된다
    pad = _PAPER_CROP_PAD
    crop_box = _bubble_crop_box(image_rgb, (bubble_box[0] - pad, bubble_box[1] - pad,
                                            bubble_box[2] + pad, bubble_box[3] + pad))
    if crop_box is None:
        return None
    bx1, by1, bx2, by2 = crop_box
    gray = cv2.cvtColor(image_rgb[by1:by2, bx1:bx2], cv2.COLOR_RGB2GRAY)
    mask_crop = full_page_mask[by1:by2, bx1:bx2]
    mask = mask_crop > 0
    tx1, ty1 = max(0, int(text_box[0]) - bx1), max(0, int(text_box[1]) - by1)
    tx2, ty2 = min(bx2 - bx1, int(np.ceil(text_box[2])) - bx1), min(by2 - by1, int(np.ceil(text_box[3])) - by1)
    if tx2 <= tx1 or ty2 <= ty1 or not mask.any():
        return None
    light = gray >= _PAPER_LIGHT
    _, white_labels = cv2.connectedComponents(light.astype(np.uint8), connectivity=4)
    inside = white_labels[ty1:ty2, tx1:tx2]
    inside = inside[inside > 0]
    if not inside.size:
        return None
    interior = (white_labels == int(np.bincount(inside).argmax())).astype(np.uint8)
    h, w = interior.shape
    if interior[0].any() or interior[-1].any() or interior[:, 0].any() or interior[:, -1].any():
        return None  # 흰 바탕이 말풍선 상자 밖으로 이어진다 — 닫힌 말풍선이 아니다
    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(interior)
    cv2.drawContours(hull, contours, -1, 1, thickness=cv2.FILLED)  # 흰 바탕과 그 속 글자
    hull = hull > 0
    area = int(hull.sum())
    if area == 0 or (hull & mask).sum() < _PAPER_COVERED * area or light[hull].mean() < _PAPER_MIN_LIGHT:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats((hull & ~light).astype(np.uint8), connectivity=8)
    dots = sum(int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] < 6)
    if dots > _PAPER_MAX_DOTS * area:
        return None  # 스크린톤 바탕
    size = 2 * _PAPER_KEEP_PX + 1
    allowed = cv2.dilate(hull.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
    mask_crop[~allowed] = 0  # 테두리와 그 밖은 지우지 않는다
    fill = allowed & mask
    color = np.median(image_rgb[by1:by2, bx1:bx2][interior > 0], axis=0).astype(np.uint8)
    page_fill = np.zeros(full_page_mask.shape, dtype=bool)
    page_fill[by1:by2, bx1:bx2] = fill & hull
    return page_fill, color


def _glyph_masks(text_mask_model, image_rgb, boxes):
    """박스마다 넓히기 전 글자 모양 마스크(bool). 모델이 없으면 None들."""
    if text_mask_model is None or not boxes:
        return [None] * len(boxes)
    return text_mask_model([image_rgb[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes])


def _freeform_erase_masks(image_rgb, boxes, glyphs):
    """말풍선 밖 글자 박스마다 지울 마스크(박스 크기 uint8)를 만든다. None이면 박스 전체를 지운다."""
    size = 2 * _GLYPH_DILATE_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    masks = []
    for (x1, y1, x2, y2), glyph in zip(boxes, glyphs):
        if glyph is None or glyph.mean() < _MIN_GLYPH_RATIO:
            masks.append(None)  # 글자를 못 찾았다 — 글자를 남기느니 박스째 지운다
            continue
        masks.append(_absorb_freeform_ink(image_rgb[y1:y2, x1:x2], cv2.dilate(glyph.astype(np.uint8) * 255, kernel)))
    return masks


def _measure_looks(image_rgb, entries, glyphs, freeform=False):
    """(대사, 박스)마다 지우기 전 원문 글자 모양을 재서 대사에 붙인다 — 식자가 흰 글자·외곽선을 따른다.

    freeform=True는 말풍선 밖 글자 — 흰 종이 위 검은 외곽선 흰 글자 검사를 더 한다 (말풍선 속 글자는 이 검사 없이 잰다).
    """
    for (element, (x1, y1, x2, y2)), glyph in zip(entries, glyphs):
        element.look = (measure_text_look(image_rgb[y1:y2, x1:x2], glyph, element.font_size, freeform=freeform)
                        if glyph is not None else None)


def _measure_rims(all_page_data, inpainted_pages, rim_jobs):
    """말풍선 밖 글자마다 원문 테두리 폭을 지운 그림을 바탕 삼아 재서 look["rim_px"]·["rim_polarity"]에 붙인다.

    지우기가 다 끝난 뒤라야 글자 뒤 바탕(지운 그림)을 안다. 흰 종이·먹칠 바탕(basis "plain")은 테두리가 바탕과 같은 색이라
    보이지도 재지지도 않아 건너뛴다. look은 실행 중 값이라 대사집에 쓰지 않는다 (serialization).
    """
    for page_idx, entries, glyphs in rim_jobs:
        original, erased = all_page_data[page_idx].image_rgb, inpainted_pages[page_idx]
        img_h, img_w = original.shape[:2]
        for (element, (x1, y1, x2, y2)), glyph in zip(entries, glyphs):
            look = element.look
            if glyph is None or not look or look.get("basis") == "plain":
                continue
            pad = rim_window_px(element.font_size)
            wx1, wy1, wx2, wy2 = max(0, x1 - pad), max(0, y1 - pad), min(img_w, x2 + pad), min(img_h, y2 + pad)
            mask = np.zeros((wy2 - wy1, wx2 - wx1), bool)
            mask[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1] = glyph
            rim = measure_rim(original[wy1:wy2, wx1:wx2], erased[wy1:wy2, wx1:wx2], mask, element.font_size)
            if rim is not None:
                look["rim_px"], look["rim_polarity"] = rim


def _is_grayscale(image_rgb):
    """흑백 페이지인가 — 8픽셀 간격으로 본 모든 점에서 RGB 채널 차이가 8 미만."""
    sample = image_rgb[::8, ::8].astype(np.int16)
    return bool(np.abs(sample[..., 0] - sample[..., 1]).max() < 8
                and np.abs(sample[..., 1] - sample[..., 2]).max() < 8)


def _expand_box_within(text_box, pad, outer_box=None, border_keep=2):
    """텍스트 박스를 pad만큼 넓히되, outer_box(말풍선) 테두리는 침범하지 않습니다.

    탐지 박스가 글자에 빠듯하면 가장자리에 글자가 미세하게 남아 인페인팅 시
    검은 노이즈가 생기므로 넉넉하게 확장한다. 단:
    - 말풍선 테두리 안쪽 border_keep px까지만 확장 (테두리 선 보존)
    - 어떤 경우에도 원래 텍스트 박스보다 작아지지 않음 (잔상 방지 우선)
    """
    x1, y1, x2, y2 = (float(v) for v in text_box[:4])
    ex1, ey1, ex2, ey2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    if outer_box is not None:
        ox1, oy1, ox2, oy2 = (float(v) for v in outer_box[:4])
        ex1 = max(ex1, ox1 + border_keep)
        ey1 = max(ey1, oy1 + border_keep)
        ex2 = min(ex2, ox2 - border_keep)
        ey2 = min(ey2, oy2 - border_keep)
    return [min(ex1, x1), min(ey1, y1), max(ex2, x2), max(ey2, y2)]


def inpaint_pages_in_batch(models, all_page_data: List[PageData]) -> List[np.ndarray]:
    """
    여러 페이지의 모든 텍스트 영역을 하나의 배치로 Inpaint합니다.

    결과 합성은 마스크 영역(지운 글자 부분)에만 적용한다. 컨텍스트 패치의
    나머지 픽셀은 원본 그대로 유지되므로 글자 주변 그림이 열화되지 않는다.
    """
    lama_model = models['inpainting']
    text_mask_model = models.get('text_mask')
    all_patches_to_inpaint = []
    all_patches_metadata = []
    gray_pages = set()
    paper_fills = []  # (쪽 번호, 칠할 픽셀, 색) — 지우기 모델 결과 위에 말풍선 흰 바탕을 다시 칠한다
    rim_jobs = []  # (쪽 번호, 말풍선 밖 (대사, 박스), 글자 마스크) — 다 지운 뒤 원문 테두리 폭을 잰다

    # 원본 이미지를 복사하여 최종 결과물로 사용할 리스트를 초기화합니다.
    inpainted_pages = [p.image_rgb.copy() for p in all_page_data]

    bubble_pad = max(0, int(config.INPAINT_BUBBLE_MASK_PADDING))
    freeform_pad = max(0, int(config.INPAINT_MASK_PADDING))

    logger.info("모든 페이지에서 Inpaint할 영역을 수집 중...")
    for page_idx, page_data in enumerate(all_page_data):
        # (지울 박스, 합성 소유 영역) — 소유 영역 밖 픽셀은 이 패치가 절대 덮어쓰지 않는다.
        # 공유 마스크 때문에 큰 패치(축소→복원으로 화질이 떨어진 것)가 이웃 말풍선
        # 영역을 흐릿한 버전으로 덮어쓰는 사고를 막는다.
        all_coords_to_erase = []
        own_regions = []
        bubble_erase_entries = []
        for bubble in page_data.speech_bubbles:
            expanded_box = _expand_box_within(bubble.text_element.text_box, bubble_pad, bubble.bubble_box)
            bubble_erase_entries.append((expanded_box, bubble.bubble_box))
            all_coords_to_erase.append(expanded_box)
            own_regions.append([
                min(expanded_box[0], bubble.bubble_box[0]),
                min(expanded_box[1], bubble.bubble_box[1]),
                max(expanded_box[2], bubble.bubble_box[2]),
                max(expanded_box[3], bubble.bubble_box[3]),
            ])
        freeform_entries = []
        freeform_slots = []  # all_coords_to_erase에서 그 말풍선 밖 글자의 자리
        for ff_text in page_data.freeform_texts:
            expanded_box = _expand_box_within(ff_text.text_box, freeform_pad)
            all_coords_to_erase.append(expanded_box)
            own_regions.append(expanded_box)
            clipped_box = _clip_coords_to_image(page_data.image_rgb, expanded_box)
            if clipped_box is not None:
                freeform_entries.append((ff_text, clipped_box))
                freeform_slots.append(len(all_coords_to_erase) - 1)
        freeform_boxes = [box for _, box in freeform_entries]

        if not all_coords_to_erase:
            continue
        if _is_grayscale(page_data.image_rgb):
            gray_pages.add(page_idx)

        # 글자 모양 마스크(넓히기 전) — 말풍선 밖 글자 지우기와 원문 글자 모양 실측에 함께 쓴다
        freeform_glyphs = _glyph_masks(text_mask_model, page_data.image_rgb, freeform_boxes)
        bubble_entries = [(bubble.text_element, box) for bubble, (expanded_box, _) in
                          zip(page_data.speech_bubbles, bubble_erase_entries)
                          if (box := _clip_coords_to_image(page_data.image_rgb, expanded_box)) is not None]
        _measure_looks(page_data.image_rgb, freeform_entries, freeform_glyphs, freeform=True)
        rim_jobs.append((page_idx, freeform_entries, freeform_glyphs))
        _measure_looks(page_data.image_rgb, bubble_entries,
                       _glyph_masks(text_mask_model, page_data.image_rgb, [box for _, box in bubble_entries]))

        # 말풍선 안 글자는 박스로, 말풍선 밖 글자는 글자 모양으로 지운다 (글자를 못 찾으면 박스)
        glyph_masks = _freeform_erase_masks(page_data.image_rgb, freeform_boxes, freeform_glyphs)
        box_coords = [expanded_box for expanded_box, _ in bubble_erase_entries]
        box_coords += [box for box, glyph_mask in zip(freeform_boxes, glyph_masks) if glyph_mask is None]
        full_page_mask = create_mask_from_coords(page_data.image_rgb, box_coords)
        for (x1, y1, x2, y2), glyph_mask in zip(freeform_boxes, glyph_masks):
            if glyph_mask is not None:
                region = full_page_mask[y1:y2, x1:x2]
                np.maximum(region, glyph_mask, out=region)
        # 상자 곁에 남은 글자 조각(첫 글자·후리가나·떨어진 기호)도 지운다 — 다른 글자 상자는 건드리지 않는다
        all_text_boxes = ([b.text_element.text_box for b in page_data.speech_bubbles]
                          + [t.text_box for t in page_data.freeform_texts]
                          + list(getattr(page_data, "untouched_boxes", [])))  # 원문으로 둘 글자 (pipeline이 붙임)
        for (ff_text, box), slot, glyph_mask in zip(freeform_entries, freeform_slots, glyph_masks):
            if glyph_mask is None:
                continue
            keep_out = [b for b in all_text_boxes if b is not ff_text.text_box]
            grown = _absorb_freeform_halo(page_data.image_rgb, full_page_mask, box, ff_text.font_size,
                                          text_mask_model, keep_out)
            if grown is not None:
                all_coords_to_erase[slot] = list(grown)
                own_regions[slot] = list(grown)
        # 박스 곁에 남은 후리가나·첨자 같은 미탐지 글자 조각을 마스크로 흡수 — 글자 모양 모델로 그림 선은 거른다
        bubble_crops = [_bubble_crop_box(page_data.image_rgb, bubble_box) for _, bubble_box in bubble_erase_entries]
        valid_crops = [box for box in bubble_crops if box is not None]
        crop_glyphs = iter(_glyph_masks(text_mask_model, page_data.image_rgb, valid_crops))
        for (expanded_box, bubble_box), crop_box in zip(bubble_erase_entries, bubble_crops):
            glyph = next(crop_glyphs) if crop_box is not None else None
            _absorb_residual_ink(page_data.image_rgb, full_page_mask, expanded_box, bubble_box, glyph)
        for expanded_box, bubble_box in bubble_erase_entries:
            paper = _bubble_paper_fill(page_data.image_rgb, full_page_mask, expanded_box, bubble_box)
            if paper is not None:
                paper_fills.append((page_idx, *paper))
        img_h, img_w = page_data.image_rgb.shape[:2]

        for coords, own_region in zip(all_coords_to_erase, own_regions):
            clipped_coords = _clip_coords_to_image(page_data.image_rgb, coords)
            if clipped_coords is None:
                logger.debug(f"Skipping invalid inpaint box: {coords}")
                continue

            x1, y1, x2, y2 = clipped_coords
            # 컨텍스트는 박스 크기에 비례해 적응적으로 넓힌다. 구멍이 패치의
            # 절반을 넘으면 LaMa(FFC)가 주변 세로선 텍스처를 구멍 안으로
            # 환각(줄무늬)하므로, 큰 박스일수록 주변을 더 넓게 보여줘야 한다.
            pad = max(int(config.INPAINT_CONTEXT_PADDING), int(0.6 * max(x2 - x1, y2 - y1)))
            pad = min(pad, 256)
            ctx_x1, ctx_y1 = max(0, x1 - pad), max(0, y1 - pad)
            ctx_x2, ctx_y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
            if ctx_x2 <= ctx_x1 or ctx_y2 <= ctx_y1:
                logger.debug(f"Skipping empty inpaint context: {coords} -> {(ctx_x1, ctx_y1, ctx_x2, ctx_y2)}")
                continue

            context_patch = page_data.image_rgb[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
            patch_mask = full_page_mask[ctx_y1:ctx_y2, ctx_x1:ctx_x2]

            all_patches_to_inpaint.append((context_patch, patch_mask))
            all_patches_metadata.append({
                'page_idx': page_idx,
                'coords': (ctx_x1, ctx_y1, ctx_x2, ctx_y2),
                'own_region': own_region,
            })

    if not all_patches_to_inpaint:
        logger.info("Inpaint할 텍스트가 없습니다.")
        return inpainted_pages

    # 모든 페이지의 모든 패치를 한 번에 처리
    inpainted_patches = erase_patches_in_batch(lama_model, all_patches_to_inpaint)

    logger.info("Inpaint된 패치를 원본 페이지에 다시 적용 중...")
    for i, patch_meta in enumerate(tqdm(all_patches_metadata, desc="Applying Patches")):
        page_idx = patch_meta['page_idx']
        ctx_x1, ctx_y1, ctx_x2, ctx_y2 = patch_meta['coords']
        inpainted_patch = inpainted_patches[i]
        if page_idx in gray_pages:  # 흑백 페이지 — LaMa가 지어낸 옅은 색을 없앤다
            inpainted_patch = cv2.cvtColor(cv2.cvtColor(inpainted_patch, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        patch_mask = all_patches_to_inpaint[i][1]

        h, w = ctx_y2 - ctx_y1, ctx_x2 - ctx_x1
        if inpainted_patch.shape[0] != h or inpainted_patch.shape[1] != w:
            inpainted_patch = cv2.resize(inpainted_patch, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # 마스크 ∩ 자기 소유 영역 픽셀만 교체 — 글자 주변 원본 그림을 보존하고,
        # 다른 박스 영역을 (화질이 다른) 이 패치 결과로 덮어쓰지 않는다.
        region = inpainted_pages[page_idx][ctx_y1:ctx_y2, ctx_x1:ctx_x2]
        mask_bool = patch_mask > 127
        ox1, oy1, ox2, oy2 = patch_meta['own_region']
        own_bool = np.zeros_like(mask_bool)
        sy1 = max(0, int(oy1) - ctx_y1)
        sy2 = min(h, int(oy2) - ctx_y1)
        sx1 = max(0, int(ox1) - ctx_x1)
        sx2 = min(w, int(ox2) - ctx_x1)
        if sy2 > sy1 and sx2 > sx1:
            own_bool[sy1:sy2, sx1:sx2] = True
        mask_bool &= own_bool
        region[mask_bool] = inpainted_patch[mask_bool]

    for page_idx, fill, color in paper_fills:
        inpainted_pages[page_idx][fill] = color
    _measure_rims(all_page_data, inpainted_pages, rim_jobs)
    return inpainted_pages


def _prepare_patch_canvas(patch_np, mask_np, target_size):
    """패치를 비율 유지한 채 target_size 정사각 캔버스에 배치합니다.

    - target_size 이하 패치: 리샘플링 없이 그대로 배치 (원본 화질 유지)
    - 초과 패치: 비율 유지 축소 후 배치
    이미지 패딩은 가장자리 복제(인페인팅 컨텍스트로 자연스러움), 마스크 패딩은 0.
    """
    h, w = patch_np.shape[:2]
    scale = min(1.0, target_size / max(h, w))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_small = cv2.resize(patch_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        new_w, new_h = w, h
        img_small, mask_small = patch_np, mask_np

    img_canvas = cv2.copyMakeBorder(
        img_small, 0, target_size - new_h, 0, target_size - new_w, cv2.BORDER_REPLICATE
    )
    mask_canvas = cv2.copyMakeBorder(
        mask_small, 0, target_size - new_h, 0, target_size - new_w,
        cv2.BORDER_CONSTANT, value=0,
    )
    return img_canvas, mask_canvas, (w, h, new_w, new_h, scale)


def erase_patches_in_batch(lama_model, patch_mask_list, target_size=512):
    """(이미지, 마스크) 조각 리스트를 받아 일괄적으로 텍스트를 제거합니다."""
    if not patch_mask_list:
        return []

    logger.info(f"총 {len(patch_mask_list)}개의 텍스트 조각을 미니 배치로 나누어 Inpainting 시작...")

    all_output_patches = []
    batch_size = config.INPAINT_BATCH_SIZE

    for i in tqdm(range(0, len(patch_mask_list), batch_size), desc="Inpainting Batches"):
        mini_batch = patch_mask_list[i:i + batch_size]

        img_canvases = []
        mask_canvases = []
        metas = []
        for patch_np, mask_np in mini_batch:
            img_canvas, mask_canvas, meta = _prepare_patch_canvas(patch_np, mask_np, target_size)
            img_canvases.append(img_canvas)
            mask_canvases.append(mask_canvas)
            metas.append(meta)

        img_batch = (
            torch.from_numpy(np.stack(img_canvases))
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
            .to(lama_model.device)
        )
        mask_batch = (
            torch.from_numpy((np.stack(mask_canvases) > 127).astype(np.float32))
            .unsqueeze(1)
            .to(lama_model.device)
        )

        with torch.inference_mode():
            inpainted_batch = lama_model.model(img_batch, mask_batch)

        out_np = (
            inpainted_batch.clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        for j, (w, h, new_w, new_h, scale) in enumerate(metas):
            crop = out_np[j][:new_h, :new_w]
            if scale < 1.0:
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
            all_output_patches.append(crop)

    logger.info("배치 Inpainting 완료.")
    return all_output_patches


def create_mask_from_coords(image, list_of_coords):
    """
    좌표 리스트([x1, y1, x2, y2], ...)를 기반으로 Inpainting 마스크를 생성합니다.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for coords in list_of_coords:
        clipped_coords = _clip_coords_to_image(image, coords)
        if clipped_coords is None:
            continue

        x1, y1, x2, y2 = clipped_coords
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask
