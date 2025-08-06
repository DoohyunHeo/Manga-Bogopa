import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image
import re
from tqdm import tqdm
from src import config


# [복원] 페이지 전체 Inpainting을 위한 함수
# [수정] (이미지, 마스크) 쌍을 처리하는 최종 Inpainting 함수
def erase_patches_in_batch(lama_model, patch_mask_list, target_size=512):
    if not patch_mask_list:
        return []

    print(f"총 {len(patch_mask_list)}개의 텍스트 조각을 미니 배치로 나누어 Inpainting 시작...")

    all_output_patches = []
    batch_size = 16

    for i in tqdm(range(0, len(patch_mask_list), batch_size), desc="Inpainting Batches"):
        mini_batch = patch_mask_list[i:i + batch_size]

        original_sizes = []
        batch_images = []
        batch_masks = []

        for patch_np, mask_np in mini_batch:
            patch_pil = Image.fromarray(patch_np)
            mask_pil = Image.fromarray(mask_np).convert("L")

            original_sizes.append(patch_pil.size)
            batch_images.append(patch_pil.resize((target_size, target_size), Image.Resampling.LANCZOS))
            batch_masks.append(mask_pil.resize((target_size, target_size), Image.Resampling.NEAREST))

        img_tensors = [to_tensor(img) for img in batch_images]
        mask_tensors = [to_tensor(mask) for mask in batch_masks]

        img_batch = torch.stack(img_tensors).to(lama_model.device)
        mask_batch = torch.stack(mask_tensors).to(lama_model.device)

        with torch.no_grad():
            inpainted_batch = lama_model.model(img_batch, mask_batch)

        output_patches_mini_batch = []
        for j in range(inpainted_batch.size(0)):
            inpainted_tensor = inpainted_batch[j].cpu()
            inpainted_pil = to_pil_image(inpainted_tensor)
            inpainted_pil = inpainted_pil.resize(original_sizes[j], Image.Resampling.LANCZOS)
            output_patches_mini_batch.append(np.array(inpainted_pil))

        all_output_patches.extend(output_patches_mini_batch)

    print("배치 Inpainting 완료.")
    return all_output_patches


def create_mask_from_coords(image, list_of_coords, padding=0):
    """
    좌표 리스트([x1, y1, x2, y2], ...)를 기반으로 Inpainting 마스크를 생성합니다.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for coords in list_of_coords:
        x1, y1, x2, y2 = map(int, coords)

        # 패딩 적용 및 이미지 경계 확인
        padded_x1 = max(0, x1 - padding)
        padded_y1 = max(0, y1 - padding)
        padded_x2 = min(image.shape[1], x2 + padding)
        padded_y2 = min(image.shape[0], y2 + padding)

        cv2.rectangle(mask, (padded_x1, padded_y1), (padded_x2, padded_y2), 255, -1)
    return mask


def _wrap_text(text, font, max_width):
    """단어/글자 단위로 줄 바꿈을 수행하는 함수입니다. """
    lines = []

    for paragraph in text.split('\n'):
        words = re.findall(r'(·+|[!?]+|\S+)', paragraph)
        if not words:
            continue

        current_line = words[0]
        for word in words[1:]:
            joiner = "" if re.match(r'^(·+|[!?]+)$', word) else " "

            if font.getlength(current_line + joiner + word) <= max_width:
                current_line += joiner + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

    return "\n".join(lines)


def _find_best_fit_font(draw, text, initial_font_size, target_width, target_height, font_path):
    """
    영역에 맞는 최적의 폰트 크기를 찾고 텍스트를 읽기 편하게 정렬합니다.
    """
    font_size = min(initial_font_size, config.MAX_FONT_SIZE)
    font = None
    wrapped_text = text
    force_break = False

    while True: # Loop for potential re-wrapping
        current_font_size = font_size
        # Font size reduction loop
        while current_font_size >= config.MIN_FONT_SIZE:
            try:
                font = ImageFont.truetype(font_path, current_font_size)
            except IOError:
                font = ImageFont.load_default()
                break

            wrapped_text = _wrap_text(text, font, target_width)

            try:
                text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
                text_block_height = text_bbox[3] - text_bbox[1]
                text_block_width = text_bbox[2] - text_bbox[0]
            except AttributeError:
                text_block_height = current_font_size * len(wrapped_text.split('\n'))
                text_block_width = target_width

            if text_block_width <= target_width and text_block_height <= target_height:
                break

            current_font_size -= 1

        # Check if forced line break is needed
        if not force_break and (current_font_size / initial_font_size) < config.FONT_SHRINK_THRESHOLD_RATIO:
            lines = wrapped_text.split('\n')
            if lines:
                longest_line_index = max(range(len(lines)), key=lambda i: len(lines[i]))
                longest_line = lines[longest_line_index]
                if len(longest_line) > 1:
                    # Break the longest line in the middle
                    break_point = len(longest_line) // 2
                    lines[longest_line_index] = longest_line[:break_point] + "\n" + longest_line[break_point:]
                    wrapped_text = "\n".join(lines)
                    force_break = True # Avoid infinite loop
                    font_size = min(initial_font_size, config.MAX_FONT_SIZE) # Reset font size
                    continue # Restart the whole process with the new text

        # If no forced break was needed or it was already done, exit the loop
        break

    # Font size upscale logic (remains the same)
    if config.FONT_UPSCALE_IF_TOO_SMALL and font:
        try:
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
            text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
            target_area = target_width * target_height

            if target_area > 0 and (text_area / target_area) < config.FONT_AREA_FILL_RATIO:
                last_good_font = font
                last_good_wrapped_text = wrapped_text

                while current_font_size < config.MAX_FONT_SIZE:
                    current_font_size += 1
                    try:
                        temp_font = ImageFont.truetype(font_path, current_font_size)
                    except IOError:
                        break

                    temp_wrapped_text = _wrap_text(wrapped_text, temp_font, target_width)
                    temp_bbox = draw.multiline_textbbox((0, 0), temp_wrapped_text, font=temp_font, align="center")

                    temp_block_width = temp_bbox[2] - temp_bbox[0]
                    temp_block_height = temp_bbox[3] - temp_bbox[1]

                    if temp_block_width > target_width or temp_block_height > target_height:
                        break

                    last_good_font = temp_font
                    last_good_wrapped_text = temp_wrapped_text

                font = last_good_font
                wrapped_text = last_good_wrapped_text
        except AttributeError:
            pass

    return font, wrapped_text



def _find_best_fit_font_vertical(draw, text, initial_font_size, target_height, font_path):
    """[신규] 세로 쓰기 텍스트에 맞는 최적의 폰트 크기를 찾습니다."""
    font_size = min(initial_font_size, config.MAX_FONT_SIZE)
    font = None

    # 세로 쓰기를 위해 글자 사이에 개행문자 삽입
    vertical_text = "\n".join(list(text))

    while font_size >= config.MIN_FONT_SIZE:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # 세로로 쓰인 텍스트 블록의 높이 계산
        try:
            text_bbox = draw.multiline_textbbox((0, 0), vertical_text, font=font, align="center")
            text_block_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_block_height = font_size * len(vertical_text.split('\n'))

        if text_block_height <= target_height:
            # 맞는 크기를 찾으면 해당 폰트와 텍스트 반환
            return font, vertical_text

        font_size -= 2

    # 최소 크기까지 줄여도 맞지 않으면 최소 크기 폰트 반환
    return font, vertical_text


def _get_alignment_for_bubble(attachment, text_union_box, bubble_box):
    """
    텍스트 위치를 조정하되, 말풍선 경계를 넘지 않도록 보정합니다.
    """
    text_x1, text_y1, text_x2, text_y2 = text_union_box
    bubble_x1, _, bubble_x2, _ = bubble_box

    center_y = (text_y1 + text_y2) // 2

    if attachment == 'left':
        align = 'left'
        anchor = 'lm'

        ideal_x = text_x1 - config.ATTACHED_BUBBLE_TEXT_MARGIN
        safe_x = bubble_x1 + config.BUBBLE_EDGE_SAFE_MARGIN
        center_x = max(ideal_x, safe_x)

    elif attachment == 'right':
        align = 'right'
        anchor = 'rm'
        ideal_x = text_x2 + config.ATTACHED_BUBBLE_TEXT_MARGIN
        safe_x = bubble_x2 - config.BUBBLE_EDGE_SAFE_MARGIN
        center_x = min(ideal_x, safe_x)

    else:  # 'none'
        align = 'center'
        anchor = 'mm'
        center_x = (text_x1 + text_x2) // 2

    return align, anchor, center_x, center_y


def _rects_intersect(rect1, rect2):
    """두 사각형(x1, y1, x2, y2)이 겹치는지 확인하는 함수"""
    return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[3] < rect2[1] or rect1[1] > rect2[3])


def _adjust_freeform_position(freeform_bbox, center_x, center_y, bubble_text_rects):
    """
    자유 텍스트가 말풍선 텍스트와 겹치면 상하좌우 중 최소 이동 거리로 위치를 조정하여
    새로운 (x, y) 좌표를 반환합니다.
    """
    adj_x, adj_y = center_x, center_y

    for bubble_text_rect in bubble_text_rects:
        w, h = freeform_bbox[2] - freeform_bbox[0], freeform_bbox[3] - freeform_bbox[1]
        current_bbox = (adj_x - w / 2, adj_y - h / 2, adj_x + w / 2, adj_y + h / 2)

        if _rects_intersect(current_bbox, bubble_text_rect):
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

    return adj_x, adj_y


def draw_translations(inpainted_image, page_data):
    img_pil = Image.fromarray(inpainted_image)
    draw = ImageDraw.Draw(img_pil)

    bubble_text_rects, used_freeform_indices = [], set()
    for bubble in page_data['speech_bubbles']:
        if not bubble.get('translated_text'):
            bubble_box = bubble['bubble_box']
            for i, freeform in enumerate(page_data['freeform_texts']):
                if i in used_freeform_indices: continue
                ff_box = freeform['text_box']
                center_x, center_y = (ff_box[0] + ff_box[2]) / 2, (ff_box[1] + ff_box[3]) / 2
                if (bubble_box[0] < center_x < bubble_box[2] and bubble_box[1] < center_y < bubble_box[3]):
                    bubble.update({k: v for k, v in freeform.items() if
                                   k in ['translated_text', 'font_size', 'font_style', 'text_box', 'angle']})
                    used_freeform_indices.add(i)
                    break
        if not (bubble.get('translated_text') and bubble.get('font_size')): continue

        translated_text = bubble['translated_text']
        font_path = config.FONT_MAP.get(bubble.get('font_style', 'standard'), config.DEFAULT_FONT_PATH)
        predicted_px_size = bubble['font_size']
        angle = int(bubble.get('angle', 0))
        bubble_box, text_box = bubble['bubble_box'], bubble['text_box']

        box_width, box_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        is_vertical = (config.ENABLE_VERTICAL_TEXT and ' ' not in translated_text and box_width > 0 and (
                    box_height / box_width >= config.VERTICAL_TEXT_THRESHOLD))

        if is_vertical:
            target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO)
            font, wrapped_text = _find_best_fit_font_vertical(draw, translated_text, predicted_px_size, target_height,
                                                              font_path)
            if font:
                center_x, center_y = (text_box[0] + text_box[2]) / 2, (text_box[1] + text_box[3]) / 2
                text_bbox = draw.multiline_textbbox((center_x, center_y), wrapped_text, font=font, anchor="mm",
                                                    align="center")
                bubble_text_rects.append(text_bbox)
                draw.text((center_x, center_y), wrapped_text, font=font, fill=(0, 0, 0), anchor="mm", align="center")
        else:
            target_width = (bubble_box[2] - bubble_box[0]) * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
            target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO)
            font, wrapped_text = _find_best_fit_font(draw, translated_text, predicted_px_size, target_width,
                                                     target_height, font_path)
            if font:
                align, anchor, center_x, center_y = _get_alignment_for_bubble(bubble.get('attachment', 'none'),
                                                                              text_box, bubble_box)
                if abs(angle) > config.MIN_ROTATION_ANGLE:
                    try:
                        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align=align)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]

                        txt_img = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
                        txt_draw = ImageDraw.Draw(txt_img)
                        txt_draw.text((0, 0), wrapped_text, font=font, fill=(0, 0, 0), align=align)

                        rotated_txt = txt_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
                        paste_x = int(center_x - rotated_txt.width / 2)
                        paste_y = int(center_y - rotated_txt.height / 2)
                        img_pil.paste(rotated_txt, (paste_x, paste_y), rotated_txt)
                        bubble_text_rects.append((paste_x, paste_y, paste_x + rotated_txt.width, paste_y + rotated_txt.height))
                    except Exception as e:
                        print(f"텍스트 회전 실패: {e}, 일반 텍스트로 대체합니다.")
                        text_bbox = draw.multiline_textbbox((center_x, center_y), wrapped_text, font=font, anchor=anchor, align=align)
                        bubble_text_rects.append(text_bbox)
                        draw.text((center_x, center_y), wrapped_text, font=font, fill=(0, 0, 0), anchor=anchor, align=align)
                else:
                    text_bbox = draw.multiline_textbbox((center_x, center_y), wrapped_text, font=font, anchor=anchor, align=align)
                    bubble_text_rects.append(text_bbox)
                    draw.text((center_x, center_y), wrapped_text, font=font, fill=(0, 0, 0), anchor=anchor, align=align)

    for i, freeform in enumerate(page_data['freeform_texts']):
        if i in used_freeform_indices: continue
        if not (freeform.get('translated_text') and freeform.get('font_size')): continue

        font_path = config.FONT_MAP.get(freeform.get('font_style', 'standard'), config.DEFAULT_FONT_PATH)
        predicted_px_size = freeform['font_size']
        angle = int(freeform.get('angle', 0))
        box = freeform['text_box']
        target_width = (box[2] - box[0]) * (1.0 - (config.FREEFORM_PADDING_RATIO * 2))
        target_height = (box[3] - box[1]) * (1 + config.VERTICAL_TOLERANCE_RATIO)
        font, wrapped_text = _find_best_fit_font(draw, freeform['translated_text'], predicted_px_size, target_width,
                                                 target_height, font_path)

        if font:
            # Get the size of the text block to be drawn
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center", stroke_width=config.FREEFORM_STROKE_WIDTH)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Desired position: horizontally centered, vertically top-aligned to the original box
            center_x = (box[0] + box[2]) / 2
            center_y = box[1] + text_height / 2 # Center of the new text block when top-aligned

            if abs(angle) > config.MIN_ROTATION_ANGLE:
                try:
                    txt_img = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
                    txt_draw = ImageDraw.Draw(txt_img)
                    txt_draw.text((0, 0), wrapped_text, font=font, fill=config.FREEFORM_FONT_COLOR,
                                  stroke_width=config.FREEFORM_STROKE_WIDTH, stroke_fill=config.FREEFORM_STROKE_COLOR,
                                  align="center")

                    rotated_txt = txt_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
                    
                    # The collision box is the rotated text box, centered at the desired top-aligned position
                    initial_bbox = (center_x - rotated_txt.width / 2, center_y - rotated_txt.height / 2,
                                    center_x + rotated_txt.width / 2, center_y + rotated_txt.height / 2)
                    
                    adj_center_x, adj_center_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects)
                    
                    paste_x = int(adj_center_x - rotated_txt.width / 2)
                    paste_y = int(adj_center_y - rotated_txt.height / 2)
                    
                    img_pil.paste(rotated_txt, (paste_x, paste_y), rotated_txt)

                except Exception as e:
                    print(f"자유 텍스트 회전 실패: {e}, 일반 텍스트로 대체합니다.")
                    # Fallback to non-rotated text, but still top-aligned
                    initial_bbox = (center_x - text_width / 2, center_y - text_height / 2, 
                                    center_x + text_width / 2, center_y + text_height / 2)
                    adj_center_x, adj_center_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects)
                    draw.text((adj_center_x, adj_center_y), wrapped_text, font=font, fill=config.FREEFORM_FONT_COLOR,
                              stroke_width=config.FREEFORM_STROKE_WIDTH, stroke_fill=config.FREEFORM_STROKE_COLOR,
                              anchor="mm", align="center")
            else:
                # Non-rotated text, top-aligned
                initial_bbox = (center_x - text_width / 2, center_y - text_height / 2, 
                                center_x + text_width / 2, center_y + text_height / 2)
                adj_center_x, adj_center_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects)
                draw.text((adj_center_x, adj_center_y), wrapped_text, font=font, fill=config.FREEFORM_FONT_COLOR,
                          stroke_width=config.FREEFORM_STROKE_WIDTH, stroke_fill=config.FREEFORM_STROKE_COLOR,
                          anchor="mm", align="center")

    return np.array(img_pil)