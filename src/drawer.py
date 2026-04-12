import logging
import re

import numpy as np
from PIL import Image

from src import config
from src.data_models import PageData
from src.text_renderer import (
    DEFAULT_STYLES, FREEFORM_STYLE,
    measure_line, measure_text, render_text_on_image, render_rotated_text_on_image,
    replace_unsupported_chars,
)
from src.utils import rects_intersect

logger = logging.getLogger(__name__)


def _wrap_text(text, font_path, font_size, style, max_width):
    """단어/글자 단위로 줄 바꿈을 수행하는 함수입니다."""
    lines = []
    for paragraph in text.split('\n'):
        words = re.findall(r'(·+|[!?]+|\S+)', paragraph)
        if not words:
            continue
        current_line = words[0]
        for word in words[1:]:
            joiner = "" if re.match(r'^(·+|[!?⋯]+)$', word) else " "
            if measure_line(current_line + joiner + word, font_path, font_size, style) <= max_width:
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


def _aggressive_wrap(text, font_path, font_size, style, max_width):
    """모든 공백에서 줄바꿈하되, 여전히 넘치는 줄은 가장 가까운 공백에서 한번 더 쪼갭니다.
    단어 자체는 절대 쪼개지 않습니다."""
    raw_lines = text.replace('\n', ' ').split(' ')
    lines = []
    current = ""
    for word in raw_lines:
        if not word:
            continue
        if not current:
            current = word
        elif measure_line(current + " " + word, font_path, font_size, style) <= max_width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _find_best_font_size(text, font_path, font_size, target_width, target_height, wrap_fn, style):
    """주어진 wrap 함수로 텍스트를 감싸면서 맞는 폰트 크기를 찾습니다."""
    wrapped_text = text
    current_size = font_size

    while current_size >= config.MIN_FONT_SIZE:
        wrapped_text = wrap_fn(text, font_path, current_size, style, target_width)
        text_w, text_h = measure_text(wrapped_text, font_path, current_size, style)
        if text_w <= target_width and text_h <= target_height:
            break
        current_size -= 1

    return wrapped_text, current_size


def _find_best_fit_font(text, initial_font_size, target_width, target_height, font_path, style):
    """영역에 맞는 최적의 폰트 크기를 찾고 텍스트를 정렬합니다.

    전략:
    1) 일반 줄바꿈으로 시도
    2) 폰트가 10% 이상 줄어들면 → 공격적 줄바꿈으로 폰트 초기화 후 재시도
    3) 최종 결과 중 더 큰 폰트를 채택
    4) 영역 대비 너무 작으면 업스케일
    5) 세로 공간이 많이 남으면 강제 개행으로 폰트 확대 시도
    """
    start_size = min(initial_font_size, config.MAX_FONT_SIZE)

    # 1단계: 일반 줄바꿈
    wrapped_text, found_size = _find_best_font_size(
        text, font_path, start_size, target_width, target_height, _wrap_text, style
    )
    logger.debug(f"[font-fit] text='{text[:20]}...' target=({target_width:.0f}x{target_height:.0f}) "
                 f"initial={start_size} → normal_wrap={found_size}")

    # 2단계: 10% 이상 줄었으면 공격적 줄바꿈으로 재시도
    if found_size < start_size * 0.9:
        agg_text, agg_size = _find_best_font_size(
            text, font_path, start_size, target_width, target_height, _aggressive_wrap, style
        )
        logger.debug(f"[font-fit] aggressive_wrap={agg_size} (normal was {found_size})")
        if agg_size > found_size:
            wrapped_text, found_size = agg_text, agg_size

    # 3단계: 영역 대비 텍스트가 작으면 업스케일
    text_w, text_h = measure_text(wrapped_text, font_path, found_size, style)
    text_area = text_w * text_h
    target_area = target_width * target_height
    if target_area > 0 and (text_area / target_area) < config.FONT_AREA_FILL_RATIO:
        last_good_text, last_good_size = wrapped_text, found_size
        up_size = found_size
        while up_size < config.MAX_FONT_SIZE:
            up_size += 1
            temp_text = _aggressive_wrap(text, font_path, up_size, style, target_width)
            temp_w, temp_h = measure_text(temp_text, font_path, up_size, style)
            if temp_w > target_width or temp_h > target_height:
                break
            last_good_text, last_good_size = temp_text, up_size
        final_w, final_h = measure_text(last_good_text, font_path, last_good_size, style)
        logger.debug(f"[font-fit] upscaled to {last_good_size}, "
                     f"text_bbox=({final_w:.0f}x{final_h:.0f}), "
                     f"fill={text_area/target_area:.2f}→{(final_w*final_h)/target_area:.2f}")
        wrapped_text, found_size = last_good_text, last_good_size

    # 5단계: 세로 공간이 많이 남으면 강제 개행으로 폰트 확대 시도
    text_w, text_h = measure_text(wrapped_text, font_path, found_size, style)
    vertical_fill = text_h / target_height if target_height > 0 else 1.0
    if vertical_fill < 0.6:
        best_text, best_size = wrapped_text, found_size
        for width_ratio in [0.75, 0.6, 0.5]:
            narrow_width = target_width * width_ratio
            trial_text, trial_size = _find_best_font_size(
                text, font_path, config.MAX_FONT_SIZE, narrow_width, target_height,
                _aggressive_wrap, style
            )
            if trial_size > best_size:
                best_text, best_size = trial_text, trial_size
        if best_size > found_size:
            logger.debug(f"[font-fit] vertical rebalance: {found_size} → {best_size}")
            wrapped_text, found_size = best_text, best_size

    return wrapped_text, found_size


def _find_best_fit_font_vertical(text, initial_font_size, target_width, target_height, font_path, style):
    """세로 쓰기 텍스트에 맞는 최적의 폰트 크기를 찾습니다."""
    font_size = min(initial_font_size, config.MAX_FONT_SIZE)
    text = text.replace("⋯", "︙")
    tokens = re.findall(r'[!?]+|.', text)
    vertical_text = "\n".join(tokens)

    current_size = font_size
    while current_size >= config.MIN_FONT_SIZE:
        text_w, text_h = measure_text(vertical_text, font_path, current_size, style)
        if text_h <= target_height and text_w <= target_width:
            break
        current_size -= 1

    # 영역 대비 텍스트가 작으면 업스케일
    text_w, text_h = measure_text(vertical_text, font_path, current_size, style)
    if target_height > 0 and (text_h / target_height) < config.FONT_AREA_FILL_RATIO:
        last_good_size = current_size
        while current_size < config.MAX_FONT_SIZE:
            current_size += 1
            temp_w, temp_h = measure_text(vertical_text, font_path, current_size, style)
            if temp_h > target_height or temp_w > target_width:
                break
            last_good_size = current_size
        current_size = last_good_size

    return vertical_text, current_size


def _fit_text(element, target_width, target_height, font_path, style):
    """수직/수평을 판단하여 최적 폰트 크기와 텍스트를 반환합니다."""
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    vertical = _is_vertical(element, box_width, box_height)

    if vertical:
        wrapped_text, font_size = _find_best_fit_font_vertical(
            element.translated_text, element.font_size,
            box_width, box_height * (1 + config.VERTICAL_TOLERANCE_RATIO), font_path, style
        )
    else:
        wrapped_text, font_size = _find_best_fit_font(
            element.translated_text, element.font_size,
            target_width, target_height, font_path, style
        )

    return wrapped_text, font_size, vertical


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


def _adjust_freeform_position(freeform_bbox, center_x, center_y, bubble_text_rects, img_size=None):
    """말풍선 밖 텍스트가 다른 텍스트와 겹치지 않도록 위치를 조정합니다."""
    adj_x, adj_y = center_x, center_y
    w, h = freeform_bbox[2] - freeform_bbox[0], freeform_bbox[3] - freeform_bbox[1]

    for _ in range(3):  # 최대 3회 재검사
        moved = False
        for bubble_text_rect in bubble_text_rects:
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
                moved = True
        if not moved:
            break

    # 이미지 경계 클램핑
    if img_size:
        img_w, img_h = img_size
        adj_x = max(w / 2, min(adj_x, img_w - w / 2))
        adj_y = max(h / 2, min(adj_y, img_h - h / 2))

    return adj_x, adj_y


def _render_text(img_pil, wrapped_text, font_path, font_size, center_x, center_y, angle, style,
                 align='center', anchor='mm'):
    """텍스트를 렌더링하고 바운딩 박스를 반환합니다. 회전이 필요하면 회전 처리합니다."""
    if abs(angle) > config.MIN_ROTATION_ANGLE:
        return render_rotated_text_on_image(
            img_pil, wrapped_text, center_x, center_y, angle,
            font_path, font_size, style, align
        )
    else:
        return render_text_on_image(
            img_pil, wrapped_text, center_x, center_y,
            font_path, font_size, style, align, anchor
        )


def _draw_speech_bubble_texts(img_pil, page_data):
    """말풍선 텍스트를 그립니다."""
    bubble_text_rects = []
    for bubble in page_data.speech_bubbles:
        element = bubble.text_element
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        style = DEFAULT_STYLES.get(element.font_style, DEFAULT_STYLES["standard"])
        element.translated_text = replace_unsupported_chars(element.translated_text, font_path)

        bubble_width = bubble.bubble_box[2] - bubble.bubble_box[0]
        bubble_height = bubble.bubble_box[3] - bubble.bubble_box[1]
        text_w = element.text_box[2] - element.text_box[0]
        text_h = element.text_box[3] - element.text_box[1]
        target_width = bubble_width * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
        target_height = bubble_height * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
        logger.debug(f"[bubble] '{element.translated_text[:15]}...' bubble=({bubble_width}x{bubble_height}) "
                     f"text_box=({text_w:.0f}x{text_h:.0f}) target=({target_width:.0f}x{target_height:.0f}) "
                     f"pred_font={element.font_size}")

        wrapped_text, font_size, vertical = _fit_text(element, target_width, target_height, font_path, style)

        if vertical:
            center_x = (element.text_box[0] + element.text_box[2]) / 2
            center_y = (element.text_box[1] + element.text_box[3]) / 2
            bbox = _render_text(img_pil, wrapped_text, font_path, font_size, center_x, center_y, 0, style)
        else:
            align, anchor, center_x, center_y = _get_alignment_for_bubble(
                bubble.attachment, element.text_box, bubble.bubble_box
            )
            bbox = _render_text(
                img_pil, wrapped_text, font_path, font_size,
                center_x, center_y, element.angle, style, align, anchor
            )

        bubble_text_rects.append(bbox)
    return bubble_text_rects


def _draw_freeform_texts(img_pil, page_data, bubble_text_rects):
    """말풍선 밖 텍스트를 그립니다."""
    for element in page_data.freeform_texts:
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        style = FREEFORM_STYLE
        element.translated_text = replace_unsupported_chars(element.translated_text, font_path)

        box_width = element.text_box[2] - element.text_box[0]
        box_height = element.text_box[3] - element.text_box[1]
        target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2))
        target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO) if box_width <= box_height else box_height

        wrapped_text, font_size, vertical = _fit_text(element, target_width, target_height, font_path, style)

        center_x = (element.text_box[0] + element.text_box[2]) / 2
        if vertical:
            center_y = (element.text_box[1] + element.text_box[3]) / 2
        else:
            _, text_h = measure_text(wrapped_text, font_path, font_size, style)
            center_y = element.text_box[1] + text_h / 2

        text_w, text_h = measure_text(wrapped_text, font_path, font_size, style)
        initial_bbox = (center_x - text_w / 2, center_y - text_h / 2,
                        center_x + text_w / 2, center_y + text_h / 2)
        img_size = (img_pil.width, img_pil.height)
        adj_x, adj_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects, img_size)

        _render_text(img_pil, wrapped_text, font_path, font_size, adj_x, adj_y,
                     element.angle if not vertical else 0, style)


def draw_text_on_image(inpainted_image, page_data: PageData):
    """Inpainted된 이미지 위에 PageData의 번역된 텍스트를 그립니다."""
    img_pil = Image.fromarray(inpainted_image)

    bubble_text_rects = _draw_speech_bubble_texts(img_pil, page_data)
    _draw_freeform_texts(img_pil, page_data, bubble_text_rects)

    return np.array(img_pil)
