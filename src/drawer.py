import functools

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re

from src import config
from src.data_models import PageData, TextElement
from src.utils import rects_intersect


@functools.lru_cache(maxsize=256)
def _get_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)


def _wrap_text(text, font, max_width):
    """단어/글자 단위로 줄 바꿈을 수행하는 함수입니다."""
    lines = []
    for paragraph in text.split('\n'):
        words = re.findall(r'(·+|[!?]+|\S+)', paragraph)
        if not words:
            continue
        current_line = words[0]
        for word in words[1:]:
            joiner = "" if re.match(r'^(·+|[!?⋯]+)$', word) else " "
            if font.getlength(current_line + joiner + word) <= max_width:
                current_line += joiner + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return "\n".join(lines)


def _is_vertical(element, box_width, box_height):
    """텍스트가 세로 쓰기인지 판단합니다."""
    return (config.ENABLE_VERTICAL_TEXT
            and ' ' not in element.translated_text
            and box_width > 0
            and (box_height / box_width >= config.VERTICAL_TEXT_THRESHOLD))


def _find_best_fit_font(draw, text, initial_font_size, target_width, target_height, font_path):
    """영역에 맞는 최적의 폰트 크기를 찾고 텍스트를 정렬합니다."""
    font_size = min(initial_font_size, config.MAX_FONT_SIZE)
    font = ImageFont.load_default()
    wrapped_text = text
    force_break = False

    while True:
        current_font_size = font_size
        while current_font_size >= config.MIN_FONT_SIZE:
            try:
                font = _get_font(font_path, current_font_size)
            except IOError:
                font = ImageFont.load_default()
                break

            wrapped_text = _wrap_text(text, font, target_width)
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
            if (text_bbox[2] - text_bbox[0]) <= target_width and (text_bbox[3] - text_bbox[1]) <= target_height:
                break
            current_font_size -= 1

        if not force_break and (current_font_size / initial_font_size) < config.FONT_SHRINK_THRESHOLD_RATIO:
            lines = wrapped_text.split('\n')
            if lines:
                longest_line_index = max(range(len(lines)), key=lambda i: len(lines[i]))
                longest_line = lines[longest_line_index]
                if len(longest_line) > 1:
                    break_point = len(longest_line) // 2
                    lines[longest_line_index] = longest_line[:break_point] + "\n" + longest_line[break_point:]
                    wrapped_text = "\n".join(lines)
                    force_break = True
                    font_size = min(initial_font_size, config.MAX_FONT_SIZE)
                    continue
        break

    if config.FONT_UPSCALE_IF_TOO_SMALL:
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
        text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
        target_area = target_width * target_height
        if target_area > 0 and (text_area / target_area) < config.FONT_AREA_FILL_RATIO:
            last_good_font, last_good_wrapped_text = font, wrapped_text
            while current_font_size < config.MAX_FONT_SIZE:
                current_font_size += 1
                try:
                    temp_font = _get_font(font_path, current_font_size)
                except IOError:
                    break
                temp_wrapped_text = _wrap_text(wrapped_text, temp_font, target_width)
                temp_bbox = draw.multiline_textbbox((0, 0), temp_wrapped_text, font=temp_font, align="center")
                if (temp_bbox[2] - temp_bbox[0]) > target_width or (temp_bbox[3] - temp_bbox[1]) > target_height:
                    break
                last_good_font, last_good_wrapped_text = temp_font, temp_wrapped_text
            font, wrapped_text = last_good_font, last_good_wrapped_text

    return font, wrapped_text


def _find_best_fit_font_vertical(draw, text, initial_font_size, target_height, font_path):
    """세로 쓰기 텍스트에 맞는 최적의 폰트 크기를 찾습니다."""
    font_size = min(initial_font_size, config.MAX_FONT_SIZE)
    font = ImageFont.load_default()
    text = text.replace("⋯", "︙")
    tokens = re.findall(r'[!?]+|.', text)
    vertical_text = "\n".join(tokens)

    current_font_size = font_size
    while current_font_size >= config.MIN_FONT_SIZE:
        try:
            font = _get_font(font_path, current_font_size)
        except IOError:
            font = ImageFont.load_default()
            break
        text_bbox = draw.multiline_textbbox((0, 0), vertical_text, font=font, align="center")
        if (text_bbox[3] - text_bbox[1]) <= target_height:
            break
        current_font_size -= 2

    try:
        font = _get_font(font_path, current_font_size)
    except IOError:
        font = ImageFont.load_default()

    if config.FONT_UPSCALE_IF_TOO_SMALL:
        text_bbox = draw.multiline_textbbox((0, 0), vertical_text, font=font, align="center")
        text_height = text_bbox[3] - text_bbox[1]
        if target_height > 0 and (text_height / target_height) < config.FONT_AREA_FILL_RATIO:
            last_good_font = font
            while current_font_size < config.MAX_FONT_SIZE:
                current_font_size += 1
                try:
                    temp_font = _get_font(font_path, current_font_size)
                except IOError:
                    break
                temp_bbox = draw.multiline_textbbox((0, 0), vertical_text, font=temp_font, align="center")
                if (temp_bbox[3] - temp_bbox[1]) > target_height:
                    break
                last_good_font = temp_font
            font = last_good_font

    return font, vertical_text


def _fit_text(draw, element, target_width, target_height, font_path):
    """수직/수평을 판단하여 최적 폰트와 텍스트를 반환합니다."""
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    vertical = _is_vertical(element, box_width, box_height)

    if vertical:
        font, wrapped_text = _find_best_fit_font_vertical(
            draw, element.translated_text, element.font_size,
            box_height * (1 + config.VERTICAL_TOLERANCE_RATIO), font_path
        )
    else:
        font, wrapped_text = _find_best_fit_font(
            draw, element.translated_text, element.font_size,
            target_width, target_height, font_path
        )

    return font, wrapped_text, vertical


def _get_alignment_for_bubble(attachment, text_box, bubble_box):
    """말풍선 내 텍스트 위치를 정렬합니다."""
    text_x1, text_y1, text_x2, text_y2 = text_box
    bubble_x1, _, bubble_x2, _ = bubble_box
    center_y = (text_y1 + text_y2) // 2

    if attachment == 'left':
        align, anchor = 'left', 'lm'
        center_x = max(text_x1 - config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x1 + config.BUBBLE_EDGE_SAFE_MARGIN)
    elif attachment == 'right':
        align, anchor = 'right', 'rm'
        center_x = min(text_x2 + config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x2 - config.BUBBLE_EDGE_SAFE_MARGIN)
    else:  # 'none'
        align, anchor = 'center', 'mm'
        center_x = (text_x1 + text_x2) // 2
    return align, anchor, center_x, center_y


def _adjust_freeform_position(freeform_bbox, center_x, center_y, bubble_text_rects):
    """자유 텍스트가 다른 텍스트와 겹치지 않도록 위치를 조정합니다."""
    adj_x, adj_y = center_x, center_y
    for bubble_text_rect in bubble_text_rects:
        w, h = freeform_bbox[2] - freeform_bbox[0], freeform_bbox[3] - freeform_bbox[1]
        current_bbox = (adj_x - w / 2, adj_y - h / 2, adj_x + w / 2, adj_y + h / 2)
        if rects_intersect(current_bbox, bubble_text_rect):
            moves = {
                'up': current_bbox[3] - bubble_text_rect[1],
                'down': bubble_text_rect[3] - current_bbox[1],
                'left': current_bbox[2] - bubble_text_rect[0],
                'right': bubble_text_rect[2] - current_bbox[0]
            }
            min_move_dir = min(moves, key=moves.get)
            if min_move_dir == 'up': adj_y -= moves['up']
            elif min_move_dir == 'down': adj_y += moves['down']
            elif min_move_dir == 'left': adj_x -= moves['left']
            elif min_move_dir == 'right': adj_x += moves['right']
    return adj_x, adj_y


def _draw_rotated_text(img_pil, text, center_x, center_y, angle, font, **kwargs):
    """회전된 텍스트를 이미지에 그립니다. 텍스트 박스 중앙을 기준으로 회전합니다."""
    align = kwargs.get('align', 'center')
    stroke_width = kwargs.get('stroke_width', 0)

    bbox_draw = ImageDraw.Draw(img_pil)
    text_bbox = bbox_draw.multiline_textbbox((0, 0), text, font=font, align=align, stroke_width=stroke_width)

    txt_img = Image.new('RGBA', (int(text_bbox[2]), int(text_bbox[3])), (255, 255, 255, 0))
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((0, 0), text, font=font, **kwargs)

    text_center_x = (text_bbox[0] + text_bbox[2]) / 2
    text_center_y = (text_bbox[1] + text_bbox[3]) / 2
    rotated_txt = txt_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, center=(text_center_x, text_center_y))

    paste_x = int(center_x - rotated_txt.width / 2)
    paste_y = int(center_y - rotated_txt.height / 2)
    img_pil.paste(rotated_txt, (paste_x, paste_y), rotated_txt)

    return (paste_x, paste_y, paste_x + rotated_txt.width, paste_y + rotated_txt.height)


def _render_text(draw, img_pil, wrapped_text, font, center_x, center_y, angle,
                 align='center', anchor='mm', draw_kwargs=None):
    """텍스트를 렌더링하고 바운딩 박스를 반환합니다. 회전이 필요하면 회전 처리합니다."""
    if draw_kwargs is None:
        draw_kwargs = {'fill': (0, 0, 0)}

    if abs(angle) > config.MIN_ROTATION_ANGLE:
        return _draw_rotated_text(img_pil, wrapped_text, center_x, center_y, angle, font, align=align, **draw_kwargs)
    else:
        text_bbox = draw.multiline_textbbox((center_x, center_y), wrapped_text, font=font, anchor=anchor, align=align)
        draw.text((center_x, center_y), wrapped_text, font=font, anchor=anchor, align=align, **draw_kwargs)
        return text_bbox


def _draw_speech_bubble_texts(draw, img_pil, page_data):
    """말풍선 텍스트를 그립니다."""
    bubble_text_rects = []
    for bubble in page_data.speech_bubbles:
        element = bubble.text_element
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        box_height = element.text_box[3] - element.text_box[1]
        target_width = (bubble.bubble_box[2] - bubble.bubble_box[0]) * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
        target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO)

        font, wrapped_text, vertical = _fit_text(draw, element, target_width, target_height, font_path)

        if vertical:
            center_x = (element.text_box[0] + element.text_box[2]) / 2
            center_y = (element.text_box[1] + element.text_box[3]) / 2
            bbox = _render_text(draw, img_pil, wrapped_text, font, center_x, center_y, 0)
        else:
            align, anchor, center_x, center_y = _get_alignment_for_bubble(bubble.attachment, element.text_box, bubble.bubble_box)
            bbox = _render_text(draw, img_pil, wrapped_text, font, center_x, center_y, element.angle, align=align, anchor=anchor)

        bubble_text_rects.append(bbox)
    return bubble_text_rects


def _draw_freeform_texts(draw, img_pil, page_data, bubble_text_rects):
    """자유 텍스트를 그립니다."""
    for element in page_data.freeform_texts:
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        box_width = element.text_box[2] - element.text_box[0]
        box_height = element.text_box[3] - element.text_box[1]
        target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2))
        target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO) if box_width <= box_height else box_height

        draw_kwargs = {
            'fill': config.FREEFORM_FONT_COLOR,
            'stroke_width': config.FREEFORM_STROKE_WIDTH,
            'stroke_fill': config.FREEFORM_STROKE_COLOR,
        }
        bbox_kwargs = {
            'align': 'center',
            'stroke_width': config.FREEFORM_STROKE_WIDTH
        }

        font, wrapped_text, vertical = _fit_text(draw, element, target_width, target_height, font_path)

        if vertical:
            center_x = (element.text_box[0] + element.text_box[2]) / 2
            center_y = (element.text_box[1] + element.text_box[3]) / 2
            text_bbox = draw.multiline_textbbox((center_x, center_y), wrapped_text, font=font, anchor="mm", **bbox_kwargs)
            initial_bbox = (center_x - (text_bbox[2]-text_bbox[0])/2, center_y - (text_bbox[3]-text_bbox[1])/2, center_x + (text_bbox[2]-text_bbox[0])/2, center_y + (text_bbox[3]-text_bbox[1])/2)
            adj_center_x, adj_center_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects)
            draw.text((adj_center_x, adj_center_y), wrapped_text, font=font, anchor="mm", align='center', **draw_kwargs)
        else:
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, **bbox_kwargs)
            center_x = (element.text_box[0] + element.text_box[2]) / 2
            center_y = element.text_box[1] + (text_bbox[3] - text_bbox[1]) / 2
            initial_bbox = (center_x - (text_bbox[2]-text_bbox[0])/2, center_y - (text_bbox[3]-text_bbox[1])/2, center_x + (text_bbox[2]-text_bbox[0])/2, center_y + (text_bbox[3]-text_bbox[1])/2)
            adj_center_x, adj_center_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects)
            _render_text(draw, img_pil, wrapped_text, font, adj_center_x, adj_center_y, element.angle, draw_kwargs=draw_kwargs)


def draw_text_on_image(inpainted_image, page_data: PageData):
    """Inpainted된 이미지 위에 PageData의 번역된 텍스트를 그립니다."""
    img_pil = Image.fromarray(inpainted_image)
    draw = ImageDraw.Draw(img_pil)

    bubble_text_rects = _draw_speech_bubble_texts(draw, img_pil, page_data)
    _draw_freeform_texts(draw, img_pil, page_data, bubble_text_rects)

    return np.array(img_pil)
