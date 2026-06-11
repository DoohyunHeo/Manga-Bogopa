import logging
from dataclasses import replace

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
from src.text_renderer import (
    render_rotated_text_on_image,
    render_text_on_image,
    render_vertical_text_on_image,
    replace_unsupported_chars,
)

logger = logging.getLogger(__name__)


def _get_alignment_for_bubble(attachment, text_box, bubble_box):
    """Position text inside a speech bubble.

    앵커는 원문 텍스트 박스가 아니라 **말풍선 중심** 기준이다. 일본어 세로
    텍스트는 말풍선 한쪽에 치우쳐 있는 경우가 많아, 그 박스 중심을 따라가면
    한국어 가로 식자가 위/옆으로 떠 보인다.
    """
    text_x1, _, text_x2, _ = text_box
    bubble_x1, bubble_y1, bubble_x2, bubble_y2 = bubble_box
    center_y = (bubble_y1 + bubble_y2) // 2

    if attachment == Attachment.LEFT:
        align, anchor = 'left', 'lm'
        center_x = max(text_x1 - config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x1 + config.BUBBLE_EDGE_SAFE_MARGIN)
    elif attachment == Attachment.RIGHT:
        align, anchor = 'right', 'rm'
        center_x = min(text_x2 + config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x2 - config.BUBBLE_EDGE_SAFE_MARGIN)
    else:
        align, anchor = 'center', 'mm'
        center_x = (bubble_x1 + bubble_x2) // 2
    return align, anchor, center_x, center_y


def _render_planned_text(img_pil, plan: TextRenderPlan):
    """Layout planning result to a concrete render call."""
    if plan.vertical:
        return render_vertical_text_on_image(
            img_pil, plan.text, plan.center_x, plan.center_y,
            plan.font_path, plan.font_size, plan.style,
            max_column_height=plan.vertical_column_height,
            max_columns=1,
        )
    if abs(plan.angle) > config.MIN_ROTATION_ANGLE:
        return render_rotated_text_on_image(
            img_pil, plan.text, plan.center_x, plan.center_y, plan.angle,
            plan.font_path, plan.font_size, plan.style, plan.align
        )
    return render_text_on_image(
        img_pil, plan.text, plan.center_x, plan.center_y,
        plan.font_path, plan.font_size, plan.style, plan.align, plan.anchor
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
        plan = plan_bubble_text(element, alignment, target_width, target_height, font_path, style,
                                bubble_box=bubble.bubble_box)
        bbox = _render_planned_text(img_pil, plan)

        bubble_text_rects.append(bbox)
    return bubble_text_rects


_DARK_BACKGROUND_LUMA = 100  # 이보다 어두운 배경에선 글자/외곽선 색을 반전


def _invert_style_for_dark_background(img_pil, text_box, style):
    """어두운 컷(먹칠 배경) 위 프리텍스트는 글자색·외곽선색을 맞바꿔 대비를 확보."""
    x1, y1, x2, y2 = (max(0, int(v)) for v in text_box[:4])
    x2 = min(img_pil.width, x2)
    y2 = min(img_pil.height, y2)
    if x2 <= x1 or y2 <= y1:
        return style
    region = np.asarray(img_pil.crop((x1, y1, x2, y2)).convert("L"))
    if region.size == 0 or region.mean() >= _DARK_BACKGROUND_LUMA:
        return style
    return replace(style, color=style.stroke_color, stroke_color=style.color)


def _draw_freeform_texts(img_pil, page_data, bubble_text_rects):
    """Draw translated freeform text.

    겹침 회피 대상에 '이미 그린 프리텍스트'도 누적한다 — 효과음이 군집한
    컷(チャッチャッ 연타 등)에서 프리텍스트끼리 포개지는 것 방지.
    """
    occupied_rects = list(bubble_text_rects)
    # 컷 경계 스캔용 그레이스케일 (가로 확장이 옆 칸/이미지 끝을 넘지 않게)
    page_gray = np.asarray(img_pil.convert("L"))
    for element in page_data.freeform_texts:
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        element.translated_text = replace_unsupported_chars(element.translated_text, font_path)

        box_width = element.text_box[2] - element.text_box[0]
        box_height = element.text_box[3] - element.text_box[1]
        style = resolve_freeform_style(element, box_width, box_height)
        style = _invert_style_for_dark_background(img_pil, element.text_box, style)
        img_size = (img_pil.width, img_pil.height)
        plan = plan_freeform_text(element, occupied_rects, img_size, font_path, style, page_gray=page_gray)
        drawn_bbox = _render_planned_text(img_pil, plan)
        if drawn_bbox is not None:
            occupied_rects.append(drawn_bbox)


def draw_text_on_image(inpainted_image, page_data: PageData):
    """Draw translated text on the inpainted image."""
    img_pil = Image.fromarray(inpainted_image)

    bubble_text_rects = _draw_speech_bubble_texts(img_pil, page_data)
    _draw_freeform_texts(img_pil, page_data, bubble_text_rects)

    return np.array(img_pil)
