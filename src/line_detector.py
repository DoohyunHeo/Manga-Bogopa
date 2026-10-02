"""Vertical-line detection for speech bubble / free-text attachment.

Uses morphological opening with a tall vertical kernel to find long dark
vertical structures (panel borders) in a region of interest. This is
resolution-agnostic: everything scales with the region height.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.data_models import Attachment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SideResult:
    """Internal carrier for per-side detection output."""
    has_line: bool
    strength: float


def _has_vertical_line(
    image_gray: np.ndarray,
    min_length_ratio: float,
) -> _SideResult:
    """Detect a long vertical dark line inside a grayscale region.

    Returns strength = max column activation (0..1) after morphology, so that
    callers can compare left vs. right sides when both trigger.
    """
    if image_gray.size == 0:
        return _SideResult(has_line=False, strength=0.0)

    height, width = image_gray.shape[:2]
    if height < 8 or width < 1:
        return _SideResult(has_line=False, strength=0.0)

    _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    min_length = max(3, int(height * min_length_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if not vertical.any():
        return _SideResult(has_line=False, strength=0.0)

    # Per-column activation: how much of the column survived the opening.
    column_energy = vertical.sum(axis=0, dtype=np.int32) // 255
    best_column = int(column_energy.max())
    strength = best_column / float(height)
    return _SideResult(has_line=strength >= min_length_ratio, strength=strength)


def _clip_strip(
    image_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Optional[np.ndarray]:
    """Clip a strip to image bounds. Returns None if the clipped region is empty."""
    h, w = image_rgb.shape[:2]
    x1c = max(0, min(int(x1), w))
    x2c = max(0, min(int(x2), w))
    y1c = max(0, min(int(y1), h))
    y2c = max(0, min(int(y2), h))
    if x2c - x1c <= 0 or y2c - y1c <= 0:
        return None
    return image_rgb[y1c:y2c, x1c:x2c]


def _to_gray(strip_rgb: np.ndarray) -> np.ndarray:
    if strip_rgb.ndim == 2:
        return strip_rgb
    return cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)


def _pick_side(left: _SideResult, right: _SideResult) -> Attachment:
    """Pick the stronger side when both trigger; fall back to NONE on tie."""
    if left.has_line and right.has_line:
        if left.strength > right.strength:
            return Attachment.LEFT
        if right.strength > left.strength:
            return Attachment.RIGHT
        return Attachment.NONE
    if left.has_line:
        return Attachment.LEFT
    if right.has_line:
        return Attachment.RIGHT
    return Attachment.NONE


def detect_bubble_attachment(
    bubble_crop_rgb: np.ndarray,
    edge_ratio: float = 0.10,
    min_length_ratio: float = 0.8,
) -> Attachment:
    """Detect whether a speech bubble is attached on its left or right edge.

    Scans a thin strip along each vertical edge of the bubble crop.
    """
    try:
        if bubble_crop_rgb is None or bubble_crop_rgb.size == 0:
            return Attachment.NONE
        h, w = bubble_crop_rgb.shape[:2]
        strip_width = max(2, int(w * edge_ratio))
        if w < 2 * strip_width:
            return Attachment.NONE

        gray = _to_gray(bubble_crop_rgb)
        left = _has_vertical_line(gray[:, :strip_width], min_length_ratio)
        right = _has_vertical_line(gray[:, -strip_width:], min_length_ratio)
        return _pick_side(left, right)
    except Exception as exc:
        logger.warning(f"Bubble attachment detection failed: {exc}")
        return Attachment.NONE


def attachment_continues(
    image_gray: np.ndarray,
    bubble_box,
    attachment: Attachment,
    edge_ratio: float = 0.10,
    min_length_ratio: float = 0.8,
) -> bool:
    """붙은 쪽 세로 줄이 컷 테두리인지 — 말풍선 위나 아래로 이어지거나, 말풍선 끝에서 컷 모서리로 꺾이는지.

    컷 테두리는 말풍선 밖으로 이어진다. 말풍선이 컷 높이를 꽉 채우면 위아래 끝에서 가로 테두리로 꺾이는데, 그 가로
    줄은 벽 쪽부터 말풍선 폭을 지나 안쪽 옆 밖까지 이어진다. 번쩍 말풍선의 뾰족한 테두리나 네모·팔각 상자의 옆선은
    말풍선 안에서 끝난다. 테두리는 칸 사이 여백 쪽에서 들어와 처음 만나는 잉크가 곧게 늘어선 것으로 본다 — 컷 안이
    검은 장면이면 가는 줄 대신 검은 면의 곧은 가장자리가 테두리이고, 집중선·톤이 빽빽한
    곳은 처음 만나는 잉크가 들쭉날쭉하다. 위·아래가 모두 쪽 끝이라 볼 곳이 없으면 이어진다고 본다.
    """
    if attachment == Attachment.NONE or image_gray is None or image_gray.size == 0:
        return attachment != Attachment.NONE
    height, width = image_gray.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bubble_box[:4])
    strip = max(2, int((x2 - x1) * edge_ratio))
    crop = image_gray[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    sides = (crop[:, :strip], crop[:, -strip:])
    if crop.shape[1] >= 2 * strip and all(_side_line(side, min_length_ratio) for side in sides):
        return False  # 양옆이 다 곧은 가는 줄 — 네모·팔각 상자다 (컷 벽에 닿아 있어도 가운데로)
    wall = x1 if attachment == Attachment.LEFT else x2
    left, right = max(0, wall - strip), min(width, wall + strip)
    outer_first = attachment == Attachment.LEFT  # 여백은 벽 바깥 — 왼쪽 벽이면 작은 좌표 쪽
    reach = max(10, int(0.25 * (y2 - y1)))
    looked = False
    for top, bottom in ((y1 - reach, y1), (y2, y2 + reach)):
        top, bottom = max(0, top), min(height, bottom)
        if bottom - top < 8 or right - left < 2:
            continue
        looked = True
        if _straight_edge(image_gray[top:bottom, left:right], outer_first, min_length_ratio):
            return True
    # 컷 모서리: 말풍선 위·아래 끝 둘레의 가로 테두리가 벽 쪽부터 안쪽 옆으로 말풍선 폭의 절반만큼 더 이어진다
    reach_x = max(10, int(0.5 * (x2 - x1)))
    span = (x1, x2 + reach_x) if attachment == Attachment.LEFT else (x1 - reach_x, x2)
    span = (max(0, span[0]), min(width, span[1]))
    band = max(4, int(0.05 * (y2 - y1)))
    for edge, above in ((y1, True), (y2, False)):
        top, bottom = max(0, edge - band), min(height, edge + band)
        if bottom - top < 2 or span[1] - span[0] < 8:
            continue
        looked = True
        if _straight_edge(image_gray[top:bottom, span[0]:span[1]].T, above, min_length_ratio):
            return True
    return not looked


_EDGE_MAX_SPREAD = 0.15  # 처음 만나는 잉크 위치의 퍼짐(10~90%) 상한 — 영역 폭 대비 (또는 2px)
_SIDE_LINE_MAX_COLUMNS = 0.5  # 말풍선 옆 띠에서 줄로 남은 열이 띠 폭의 이 비율(또는 6px)을 넘으면 가는 줄이 아니다


def _side_line(strip: np.ndarray, min_length_ratio: float) -> bool:
    """말풍선 옆 띠에 곧은 가는 세로 줄(상자 옆선)이 있는지 — 집중선·먹칠처럼 넓게 검은 것은 줄이 아니다."""
    height, width = strip.shape[:2]
    if height < 8 or width < 1:
        return False
    _, binary = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(height * min_length_ratio))))
    energy = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel).sum(axis=0) // 255
    columns = int((energy >= min_length_ratio * height).sum())
    return 0 < columns <= max(6, int(_SIDE_LINE_MAX_COLUMNS * width))


def _straight_edge(region: np.ndarray, outer_first: bool, min_share: float) -> bool:
    """영역의 줄(행)마다 바깥에서 들어와 처음 만나는 잉크가 한 세로선 위에 있는지 — min_share 이상의 줄에서.

    outer_first: 바깥(여백)이 작은 좌표 쪽인지.
    """
    height, width = region.shape[:2]
    if height < 8 or width < 2:
        return False
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = binary > 0
    if not outer_first:
        ink = ink[:, ::-1]
    found = ink.any(axis=1)
    if found.sum() < min_share * height:
        return False
    first = ink.argmax(axis=1)[found]
    if first.min() == 0 and np.mean(first == 0) > 0.5:
        return False  # 바깥 끝부터 잉크 — 여백이 안 보인다 (검은 면 한가운데)
    spread = float(np.percentile(first, 90) - np.percentile(first, 10))
    return spread <= max(2.0, _EDGE_MAX_SPREAD * width)


def detect_freeform_attachment(
    image_rgb: np.ndarray,
    text_box: Tuple[int, int, int, int],
    search_px: int,
    min_length_ratio: float = 0.7,
) -> Attachment:
    """Detect a vertical line within `search_px` of the left/right side of a freeform text box.

    Uses whatever pixels are available when the text box sits near a page edge
    (no skipping). Returns NONE if neither side has enough context.
    """
    try:
        if image_rgb is None or image_rgb.size == 0 or search_px <= 0:
            return Attachment.NONE

        x1, y1, x2, y2 = (int(v) for v in text_box[:4])
        left_strip = _clip_strip(image_rgb, x1 - search_px, y1, x1, y2)
        right_strip = _clip_strip(image_rgb, x2, y1, x2 + search_px, y2)

        left = (
            _has_vertical_line(_to_gray(left_strip), min_length_ratio)
            if left_strip is not None
            else _SideResult(has_line=False, strength=0.0)
        )
        right = (
            _has_vertical_line(_to_gray(right_strip), min_length_ratio)
            if right_strip is not None
            else _SideResult(has_line=False, strength=0.0)
        )
        return _pick_side(left, right)
    except Exception as exc:
        logger.warning(f"Freeform attachment detection failed: {exc}")
        return Attachment.NONE
