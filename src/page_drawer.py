import logging

import numpy as np
from PIL import Image

from src import config
from src.data_models import Attachment, PageData
from src.text_layout import (
    TextRenderPlan,
    plan_bubble_text,
    plan_freeform_text,
    resolve_bubble_style,
    resolve_freeform_style,
)
from src.text_renderer import render_text_on_image, render_rotated_text_on_image, replace_unsupported_chars

logger = logging.getLogger(__name__)


def _get_alignment_for_bubble(attachment, text_box, bubble_box):
    """Position text inside a speech bubble."""
    text_x1, text_y1, text_x2, text_y2 = text_box
    bubble_x1, _, bubble_x2, _ = bubble_box
    center_y = (text_y1 + text_y2) // 2

    if attachment == Attachment.LEFT:
        align, anchor = 'left', 'lm'
        center_x = max(text_x1 - config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x1 + config.BUBBLE_EDGE_SAFE_MARGIN)
    elif attachment == Attachment.RIGHT:
        align, anchor = 'right', 'rm'
        center_x = min(text_x2 + config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x2 - config.BUBBLE_EDGE_SAFE_MARGIN)
    else:
        align, anchor = 'center', 'mm'
        center_x = (text_x1 + text_x2) // 2
    return align, anchor, center_x, center_y


def _render_text(img_pil, wrapped_text, font_path, font_size, center_x, center_y, angle, style,
                 align='center', anchor='mm'):
    """Render text and return the resulting bounding box."""
    if abs(angle) > config.MIN_ROTATION_ANGLE:
        return render_rotated_text_on_image(
            img_pil, wrapped_text, center_x, center_y, angle,
            font_path, font_size, style, align
        )
    return render_text_on_image(
        img_pil, wrapped_text, center_x, center_y,
        font_path, font_size, style, align, anchor
    )


def _render_planned_text(img_pil, plan: TextRenderPlan):
    """Layout planning result to a concrete render call."""
    return _render_text(
        img_pil,
        plan.text,
        plan.font_path,
        plan.font_size,
        plan.center_x,
        plan.center_y,
        plan.angle,
        plan.style,
        plan.align,
        plan.anchor,
    )


def _draw_speech_bubble_texts(img_pil, page_data):
    """Draw translated speech-bubble text."""
    bubble_text_rects = []
    for bubble in page_data.speech_bubbles:
        element = bubble.text_element
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        element.translated_text = replace_unsupported_chars(element.translated_text, font_path)

        bubble_width = bubble.bubble_box[2] - bubble.bubble_box[0]
        bubble_height = bubble.bubble_box[3] - bubble.bubble_box[1]
        text_w = element.text_box[2] - element.text_box[0]
        text_h = element.text_box[3] - element.text_box[1]
        target_width = bubble_width * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
        target_height = bubble_height * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
        logger.debug(
            f"[bubble] '{element.translated_text[:15]}...' bubble=({bubble_width}x{bubble_height}) "
            f"text_box=({text_w:.0f}x{text_h:.0f}) target=({target_width:.0f}x{target_height:.0f}) "
            f"pred_font={element.font_size}"
        )

        style = resolve_bubble_style(element, bubble.bubble_box, target_width, target_height)
        alignment = _get_alignment_for_bubble(bubble.attachment, element.text_box, bubble.bubble_box)
        plan = plan_bubble_text(element, alignment, target_width, target_height, font_path, style)
        bbox = _render_planned_text(img_pil, plan)

        bubble_text_rects.append(bbox)
    return bubble_text_rects


def _draw_freeform_texts(img_pil, page_data, bubble_text_rects):
    """Draw translated freeform text."""
    for element in page_data.freeform_texts:
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        element.translated_text = replace_unsupported_chars(element.translated_text, font_path)

        box_width = element.text_box[2] - element.text_box[0]
        box_height = element.text_box[3] - element.text_box[1]
        style = resolve_freeform_style(element, box_width, box_height)
        img_size = (img_pil.width, img_pil.height)
        plan = plan_freeform_text(element, bubble_text_rects, img_size, font_path, style)
        _render_planned_text(img_pil, plan)


def draw_text_on_image(inpainted_image, page_data: PageData):
    """Draw translated text on the inpainted image."""
    img_pil = Image.fromarray(inpainted_image)

    bubble_text_rects = _draw_speech_bubble_texts(img_pil, page_data)
    _draw_freeform_texts(img_pil, page_data, bubble_text_rects)

    return np.array(img_pil)
