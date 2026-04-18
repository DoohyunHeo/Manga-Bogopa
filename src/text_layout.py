"""Layout planning for translated text.

This module is the thin assembler that decides:
- which style preset to use (resolve_bubble_style / resolve_freeform_style)
- whether to use vertical or horizontal layout (_is_vertical + _should_switch_to_vertical)
- which anchor/align to use based on attachment
- final TextRenderPlan for the renderer

Heavy lifting (wrapping, font-size fitting) lives in text_wrapping.py and
text_fitting.py.
"""
import logging
from dataclasses import dataclass, replace
from typing import Optional

from src import config
from src.data_models import Attachment
from src.text_fitting import (
    find_best_fit_font,
    find_best_fit_font_vertical,
    get_char_ratio_target,
)
from src.text_renderer import DEFAULT_STYLES, FREEFORM_STYLE, TextStyle, measure_text, measure_vertical_text
from src.text_wrapping import TALL_BUBBLE_MIN_CHARS, TALL_BUBBLE_RATIO, text_density
from src.utils import rects_intersect

logger = logging.getLogger(__name__)

# Internal layout heuristics (not user-tunable).
_VERTICAL_TOLERANCE_RATIO = 0.05
_DEFAULT_TEXT_OVERSAMPLE = 2
_SMALL_TEXT_OVERSAMPLE = 3


@dataclass(frozen=True)
class TextRenderPlan:
    text: str
    font_path: str
    font_size: int
    style: TextStyle
    center_x: float
    center_y: float
    angle: float = 0.0
    align: str = "center"
    anchor: str = "mm"
    vertical: bool = False
    initial_bbox: Optional[tuple] = None


def resolve_bubble_style(element, bubble_box, target_width, target_height):
    base_style = DEFAULT_STYLES.get(element.font_style, DEFAULT_STYLES["standard"])
    bubble_width = max(1, bubble_box[2] - bubble_box[0])
    bubble_height = max(1, bubble_box[3] - bubble_box[1])
    bubble_ratio = bubble_height / bubble_width
    density = text_density(element.translated_text)

    style = replace(base_style, oversample_scale=max(base_style.oversample_scale, _DEFAULT_TEXT_OVERSAMPLE))
    target_scale = style.horizontal_scale
    target_letter_spacing = style.letter_spacing
    target_line_spacing = style.line_spacing
    embolden = style.embolden
    oversample = style.oversample_scale

    if bubble_ratio >= TALL_BUBBLE_RATIO and density >= TALL_BUBBLE_MIN_CHARS:
        target_scale = min(target_scale, 0.90)
        target_letter_spacing = min(target_letter_spacing, -0.8)
        target_line_spacing = max(target_line_spacing, 1.22)
        oversample = max(oversample, _DEFAULT_TEXT_OVERSAMPLE)

    if min(target_width, target_height) <= 70 or element.font_size <= config.MIN_READABLE_TEXT_SIZE:
        target_scale = max(target_scale, 0.92)
        target_letter_spacing = max(target_letter_spacing, -0.25)
        target_line_spacing = max(target_line_spacing, 1.16)
        embolden = True
        oversample = max(oversample, _SMALL_TEXT_OVERSAMPLE)

    return replace(
        style,
        horizontal_scale=target_scale,
        letter_spacing=target_letter_spacing,
        line_spacing=target_line_spacing,
        embolden=embolden,
        oversample_scale=oversample,
    )


def resolve_freeform_style(element, box_width, box_height):
    style = FREEFORM_STYLE
    if min(box_width, box_height) <= 60 or element.font_size <= config.MIN_READABLE_TEXT_SIZE:
        return replace(
            style,
            horizontal_scale=max(style.horizontal_scale, 0.95),
            letter_spacing=max(style.letter_spacing, -0.1),
            embolden=True,
            oversample_scale=max(style.oversample_scale, _SMALL_TEXT_OVERSAMPLE),
        )
    return style


def _is_vertical(element, box_width, box_height):
    if not config.ENABLE_VERTICAL_TEXT or box_width <= 0:
        return False

    aspect = box_height / box_width
    # Extreme aspect: force vertical regardless of predicted font size or whitespace.
    force_threshold = float(getattr(config, "VERTICAL_FORCE_ASPECT_RATIO", 6.0))
    if aspect >= force_threshold:
        return True

    return (
        box_height > box_width * 1.2
        and element.font_size >= config.MIN_READABLE_TEXT_SIZE
        and ' ' not in element.translated_text
        and aspect >= config.VERTICAL_TEXT_THRESHOLD
    )


def _should_switch_to_vertical(element, fitted_size):
    """Fall back to vertical when horizontal fit shrank too far below the model's prediction."""
    if not config.ENABLE_VERTICAL_TEXT:
        return False
    if not element.translated_text or ' ' in element.translated_text:
        return False
    predicted = max(1, int(element.font_size))
    threshold = float(getattr(config, "FONT_SHRINK_THRESHOLD_RATIO", 0.75))
    return fitted_size < predicted * threshold


def _fit_text(element, target_width, target_height, font_path, style):
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    vertical = _is_vertical(element, box_width, box_height)
    char_ratio_target = get_char_ratio_target(element)
    char_ratio_reference_height = max(box_height, 1)

    if vertical:
        wrapped_text, font_size = find_best_fit_font_vertical(
            element.translated_text, element.font_size,
            box_width,
            box_height * (1 + _VERTICAL_TOLERANCE_RATIO),
            font_path,
            style,
            char_ratio_target=char_ratio_target,
            char_ratio_reference_height=char_ratio_reference_height,
        )
        return wrapped_text, font_size, vertical

    wrapped_text, font_size = find_best_fit_font(
        element.translated_text, element.font_size,
        target_width,
        target_height,
        font_path,
        style,
        char_ratio_target=char_ratio_target,
        char_ratio_reference_height=char_ratio_reference_height,
    )

    if _should_switch_to_vertical(element, font_size):
        logger.debug(
            f"[vertical-switch] text='{(element.translated_text or '')[:20]}...' "
            f"horizontal={font_size} < threshold({config.FONT_SHRINK_THRESHOLD_RATIO}) "
            f"× predicted({element.font_size}); retrying vertical from predicted size"
        )
        vertical_wrapped, vertical_size = find_best_fit_font_vertical(
            element.translated_text, element.font_size,
            box_width,
            box_height * (1 + _VERTICAL_TOLERANCE_RATIO),
            font_path,
            style,
            char_ratio_target=char_ratio_target,
            char_ratio_reference_height=char_ratio_reference_height,
        )
        return vertical_wrapped, vertical_size, True

    return wrapped_text, font_size, vertical


def _adjust_freeform_position(freeform_bbox, anchor_x, anchor_y, bubble_text_rects, img_size=None):
    """Nudge a freeform text to avoid overlap with bubble texts.

    `freeform_bbox` is the initial bounding box produced by the anchor-aware
    layout; the function preserves the anchor→bbox offset while moving the
    anchor, so left/right-aligned texts keep their alignment after adjustment.
    """
    adj_x, adj_y = anchor_x, anchor_y
    dx = freeform_bbox[0] - anchor_x
    dy = freeform_bbox[1] - anchor_y
    w = freeform_bbox[2] - freeform_bbox[0]
    h = freeform_bbox[3] - freeform_bbox[1]

    for _ in range(3):
        moved = False
        for bubble_text_rect in bubble_text_rects:
            current_bbox = (adj_x + dx, adj_y + dy, adj_x + dx + w, adj_y + dy + h)
            if rects_intersect(current_bbox, bubble_text_rect):
                moves = {
                    'up': current_bbox[3] - bubble_text_rect[1],
                    'down': bubble_text_rect[3] - current_bbox[1],
                    'left': current_bbox[2] - bubble_text_rect[0],
                    'right': bubble_text_rect[2] - current_bbox[0]
                }
                min_move_dir = min(moves, key=moves.get)
                if min_move_dir == 'up':
                    adj_y -= moves['up']
                elif min_move_dir == 'down':
                    adj_y += moves['down']
                elif min_move_dir == 'left':
                    adj_x -= moves['left']
                elif min_move_dir == 'right':
                    adj_x += moves['right']
                moved = True
        if not moved:
            break

    if img_size:
        img_w, img_h = img_size
        adj_x = max(-dx, min(adj_x, img_w - dx - w))
        adj_y = max(-dy, min(adj_y, img_h - dy - h))

    return adj_x, adj_y


def plan_bubble_text(element, alignment, target_width, target_height, font_path, style):
    wrapped_text, font_size, vertical = _fit_text(element, target_width, target_height, font_path, style)

    if vertical:
        center_x = (element.text_box[0] + element.text_box[2]) / 2
        center_y = (element.text_box[1] + element.text_box[3]) / 2
        return TextRenderPlan(
            text=wrapped_text,
            font_path=font_path,
            font_size=font_size,
            style=style,
            center_x=center_x,
            center_y=center_y,
            angle=0,
            align="center",
            anchor="mm",
            vertical=True,
        )

    align, anchor, center_x, center_y = alignment
    return TextRenderPlan(
        text=wrapped_text,
        font_path=font_path,
        font_size=font_size,
        style=style,
        center_x=center_x,
        center_y=center_y,
        angle=0,
        align=align,
        anchor=anchor,
        vertical=False,
    )


def plan_freeform_text(element, bubble_text_rects, img_size, font_path, style):
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2))
    target_height = box_height * (1 + _VERTICAL_TOLERANCE_RATIO) if box_width <= box_height else box_height
    wrapped_text, font_size, vertical = _fit_text(element, target_width, target_height, font_path, style)

    # Vertical text renders as a character stack; its actual w/h differs from
    # the horizontal measure result (width ≈ one glyph, height ≈ sum of glyphs).
    if vertical:
        text_w, text_h = measure_vertical_text(wrapped_text, font_path, font_size, style)
    else:
        text_w, text_h = measure_text(wrapped_text, font_path, font_size, style)

    x1, y1, x2, y2 = element.text_box
    if vertical:
        center_y = (y1 + y2) / 2
    else:
        center_y = y1 + text_h / 2

    attachment = getattr(element, "attachment", Attachment.NONE)
    # Vertical layout keeps its own geometry; horizontal layout honors attachment.
    if vertical or attachment == Attachment.NONE:
        align, anchor = "center", "mm"
        anchor_x = (x1 + x2) / 2
        initial_bbox = (
            anchor_x - text_w / 2,
            center_y - text_h / 2,
            anchor_x + text_w / 2,
            center_y + text_h / 2,
        )
    elif attachment == Attachment.LEFT:
        align, anchor = "left", "lm"
        anchor_x = x1 - config.FREEFORM_ATTACHMENT_TEXT_MARGIN
        initial_bbox = (anchor_x, center_y - text_h / 2, anchor_x + text_w, center_y + text_h / 2)
    else:  # Attachment.RIGHT
        align, anchor = "right", "rm"
        anchor_x = x2 + config.FREEFORM_ATTACHMENT_TEXT_MARGIN
        initial_bbox = (anchor_x - text_w, center_y - text_h / 2, anchor_x, center_y + text_h / 2)

    adj_x, adj_y = _adjust_freeform_position(initial_bbox, anchor_x, center_y, bubble_text_rects, img_size)

    return TextRenderPlan(
        text=wrapped_text,
        font_path=font_path,
        font_size=font_size,
        style=style,
        center_x=adj_x,
        center_y=adj_y,
        angle=element.angle if not vertical else 0,
        align=align,
        anchor=anchor,
        vertical=vertical,
        initial_bbox=initial_bbox,
    )
