import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re

from src import config
from src.data_models import PageData
from src.utils import rects_intersect


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


def _find_best_fit_font(draw, text, initial_font_size, target_width, target_height, font_path):
    """영역에 맞는 최적의 폰트 크기를 찾고 텍스트를 정렬합니다."""
    font_size = min(initial_font_size, config.MAX_FONT_SIZE)
    font = ImageFont.load_default()
    wrapped_text = text
    force_break = False

    while True:  # 잠재적인 재래핑을 위한 루프
        current_font_size = font_size
        while current_font_size >= config.MIN_FONT_SIZE:
            try:
                font = ImageFont.truetype(font_path, current_font_size)
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
                    temp_font = ImageFont.truetype(font_path, current_font_size)
                except IOError:
                    break
                temp_wrapped_text = _wrap_text(wrapped_text, temp_font, target_width)
                temp_bbox = draw.multiline_textbbox((0, 0), temp_wrapped_text, font=temp_font, align="center")
                if (temp_bbox[2] - temp_bbox[0]) > target_width or (temp_bbox[3] - temp_bbox[1]) > target_height:
                    break

                temp_text_area = (temp_bbox[2] - temp_bbox[0]) * (temp_bbox[3] - temp_bbox[1])
                if target_area > 0 and (temp_text_area / target_area) > config.FONT_AREA_FILL_RATIO:
                    last_good_font, last_good_wrapped_text = temp_font, temp_wrapped_text
                    break

                last_good_font, last_good_wrapped_text = temp_font, temp_wrapped_text
            font, wrapped_text = last_good_font, last_good_wrapped_text

    return font, wrapped_text


def _find_best_fit_font_vertical(draw, text, initial_font_size, target_height, font_path):
    """세로 쓰기 텍스트에 맞는 최적의 폰트 크기를 찾습니다."""
    font_size = min(initial_font_size, config.MAX_FONT_SIZE)
    font = ImageFont.load_default()
    text.replace("⋯", "︙")
    # ?나 !가 연속될 경우 한 글자로 취급하여 세로쓰기
    tokens = re.findall(r'[!?]+|.', text)
    vertical_text = "\n".join(tokens)

    # 1. 영역에 맞는 가장 큰 폰트 크기를 찾습니다.
    current_font_size = font_size
    while current_font_size >= config.MIN_FONT_SIZE:
        try:
            font = ImageFont.truetype(font_path, current_font_size)
        except IOError:
            font = ImageFont.load_default()
            break
        text_bbox = draw.multiline_textbbox((0, 0), vertical_text, font=font, align="center")
        if (text_bbox[3] - text_bbox[1]) <= target_height:
            break  # 맞는 크기를 찾았으므로 upscale을 고려하기 위해 중단
        current_font_size -= 2
    
    try:
        font = ImageFont.truetype(font_path, current_font_size)
    except IOError:
        font = ImageFont.load_default()

    # 2. 텍스트가 영역을 충분히 채우지 못하면 폰트 크기를 다시 키웁니다.
    if config.FONT_UPSCALE_IF_TOO_SMALL:
        text_bbox = draw.multiline_textbbox((0, 0), vertical_text, font=font, align="center")
        text_height = text_bbox[3] - text_bbox[1]
        
        if target_height > 0 and (text_height / target_height) < config.FONT_AREA_FILL_RATIO:
            last_good_font = font
            while current_font_size < config.MAX_FONT_SIZE:
                current_font_size += 1
                try:
                    temp_font = ImageFont.truetype(font_path, current_font_size)
                except IOError:
                    break
                
                temp_bbox = draw.multiline_textbbox((0, 0), vertical_text, font=temp_font, align="center")
                if (temp_bbox[3] - temp_bbox[1]) > target_height:
                    break  # 이 크기는 너무 큽니다.

                temp_text_height = temp_bbox[3] - temp_bbox[1]
                if target_height > 0 and (temp_text_height / target_height) > config.FONT_AREA_FILL_RATIO:
                    last_good_font = temp_font
                    break
                
                last_good_font = temp_font
            font = last_good_font

    return font, vertical_text


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


def draw_text_on_image(inpainted_image, page_data: PageData):
    """Inpainted된 이미지 위에 PageData의 번역된 텍스트를 그립니다."""
    img_pil = Image.fromarray(inpainted_image)
    draw = ImageDraw.Draw(img_pil)
    bubble_text_rects = []

    # 1. 말풍선 텍스트 그리기
    for bubble in page_data.speech_bubbles:
        element = bubble.text_element
        if not element.translated_text: continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        box_width = element.text_box[2] - element.text_box[0]
        box_height = element.text_box[3] - element.text_box[1]
        is_vertical = config.ENABLE_VERTICAL_TEXT and ' ' not in element.translated_text and box_width > 0 and (box_height / box_width >= config.VERTICAL_TEXT_THRESHOLD)

        if is_vertical:
            font, wrapped_text = _find_best_fit_font_vertical(draw, element.translated_text, element.font_size, box_height * (1 + config.VERTICAL_TOLERANCE_RATIO), font_path)
            center_x, center_y = (element.text_box[0] + element.text_box[2]) / 2, (element.text_box[1] + element.text_box[3]) / 2
            text_bbox = draw.multiline_textbbox((center_x, center_y), wrapped_text, font=font, anchor="mm", align="center")
            bubble_text_rects.append(text_bbox)
            draw.text((center_x, center_y), wrapped_text, font=font, fill=(0, 0, 0), anchor="mm", align="center")
        else:
            target_width = (bubble.bubble_box[2] - bubble.bubble_box[0]) * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
            target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO)
            font, wrapped_text = _find_best_fit_font(draw, element.translated_text, element.font_size, target_width, target_height, font_path)
            align, anchor, center_x, center_y = _get_alignment_for_bubble(bubble.attachment, element.text_box, bubble.bubble_box)
            
            if abs(element.angle) > config.MIN_ROTATION_ANGLE:
                text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align=align)
                txt_img = Image.new('RGBA', (int(text_bbox[2]), int(text_bbox[3])), (255, 255, 255, 0))
                txt_draw = ImageDraw.Draw(txt_img)
                txt_draw.text((0, 0), wrapped_text, font=font, fill=(0, 0, 0), align=align)
                rotated_txt = txt_img.rotate(element.angle, expand=True, resample=Image.Resampling.BICUBIC)
                paste_x, paste_y = int(center_x - rotated_txt.width / 2), int(center_y - rotated_txt.height / 2)
                img_pil.paste(rotated_txt, (paste_x, paste_y), rotated_txt)
                bubble_text_rects.append((paste_x, paste_y, paste_x + rotated_txt.width, paste_y + rotated_txt.height))
            else:
                text_bbox = draw.multiline_textbbox((center_x, center_y), wrapped_text, font=font, anchor=anchor, align=align)
                bubble_text_rects.append(text_bbox)
                draw.text((center_x, center_y), wrapped_text, font=font, fill=(0, 0, 0), anchor=anchor, align=align)

    # 2. 자유 텍스트 그리기
    for element in page_data.freeform_texts:
        if not element.translated_text: continue

        box_width = element.text_box[2] - element.text_box[0]
        box_height = element.text_box[3] - element.text_box[1]

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2))

        # 박스 가로가 세로보다 길면 높이 제약을 더 엄격하게 적용
        if box_width > box_height:
            target_height = box_height
        else:
            target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO)

        font, wrapped_text = _find_best_fit_font(draw, element.translated_text, element.font_size, target_width, target_height, font_path)

        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center", stroke_width=config.FREEFORM_STROKE_WIDTH)

        # 텍스트 블록을 박스 상단 기준으로 위치시키도록 y 좌표 계산
        center_x = (element.text_box[0] + element.text_box[2]) / 2
        center_y = element.text_box[1] + (text_bbox[3] - text_bbox[1]) / 2

        initial_bbox = (center_x - (text_bbox[2]-text_bbox[0])/2, center_y - (text_bbox[3]-text_bbox[1])/2, center_x + (text_bbox[2]-text_bbox[0])/2, center_y + (text_bbox[3]-text_bbox[1])/2)
        adj_center_x, adj_center_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects)

        # 2. 자유 텍스트 그리기
    for element in page_data.freeform_texts:
        if not element.translated_text: continue

        box_width = element.text_box[2] - element.text_box[0]
        box_height = element.text_box[3] - element.text_box[1]
        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)

        is_vertical = config.ENABLE_VERTICAL_TEXT and ' ' not in element.translated_text and box_width > 0 or (box_height / box_width >= config.VERTICAL_TEXT_THRESHOLD)

        if is_vertical:
            # --- 세로 쓰기 로직 ---
            font, wrapped_text = _find_best_fit_font_vertical(draw, element.translated_text, element.font_size, box_height * (1 + config.VERTICAL_TOLERANCE_RATIO), font_path)

            center_x, center_y = (element.text_box[0] + element.text_box[2]) / 2, (element.text_box[1] + element.text_box[3]) / 2
            text_bbox = draw.multiline_textbbox((center_x, center_y), wrapped_text, font=font, anchor="mm", align="center", stroke_width=config.FREEFORM_STROKE_WIDTH)

            initial_bbox = (center_x - (text_bbox[2]-text_bbox[0])/2, center_y - (text_bbox[3]-text_bbox[1])/2, center_x + (text_bbox[2]-text_bbox[0])/2, center_y + (text_bbox[3]-text_bbox[1])/2)
            adj_center_x, adj_center_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects)

            draw.text((adj_center_x, adj_center_y), wrapped_text, font=font, fill=config.FREEFORM_FONT_COLOR, stroke_width=config.FREEFORM_STROKE_WIDTH, stroke_fill=config.FREEFORM_STROKE_COLOR, anchor="mm", align="center")
        else:
            # --- 가로 쓰기 로직 ---
            target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2))
            if box_width > box_height:
                target_height = box_height
            else:
                target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO)

            font, wrapped_text = _find_best_fit_font(draw, element.translated_text, element.font_size, target_width, target_height, font_path)

            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center", stroke_width=config.FREEFORM_STROKE_WIDTH)

            # 텍스트 블록을 박스 상단 기준으로 위치시키도록 y 좌표 계산
            center_x = (element.text_box[0] + element.text_box[2]) / 2
            center_y = element.text_box[1] + (text_bbox[3] - text_bbox[1]) / 2
            initial_bbox = (center_x - (text_bbox[2]-text_bbox[0])/2, center_y - (text_bbox[3]-text_bbox[1])/2, center_x + (text_bbox[2]-text_bbox[0])/2, center_y + (text_bbox[3]-text_bbox[1])/2)
            adj_center_x, adj_center_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects)

            if abs(element.angle) > config.MIN_ROTATION_ANGLE:
                txt_img = Image.new('RGBA', (int(text_bbox[2]), int(text_bbox[3])), (255, 255, 255, 0))
                txt_draw = ImageDraw.Draw(txt_img)
                txt_draw.text((0, 0), wrapped_text, font=font, fill=config.FREEFORM_FONT_COLOR, stroke_width=config.FREEFORM_STROKE_WIDTH, stroke_fill=config.FREEFORM_STROKE_COLOR, align="center")
                rotated_txt = txt_img.rotate(element.angle, expand=True, resample=Image.Resampling.BICUBIC)
                paste_x, paste_y = int(adj_center_x - rotated_txt.width / 2), int(adj_center_y - rotated_txt.height / 2)
                img_pil.paste(rotated_txt, (paste_x, paste_y), rotated_txt)
            else:
                draw.text((adj_center_x, adj_center_y), wrapped_text, font=font, fill=config.FREEFORM_FONT_COLOR, stroke_width=config.FREEFORM_STROKE_WIDTH, stroke_fill=config.FREEFORM_STROKE_COLOR, anchor="mm", align="center")

    return np.array(img_pil)
