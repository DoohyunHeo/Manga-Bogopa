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
    dark_threshold: Optional[int] = None,
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

    if dark_threshold is None:
        _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(image_gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

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
