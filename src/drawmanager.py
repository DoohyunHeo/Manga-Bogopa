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
    wrapped_text = ""

    #폰트 크기 축소
    while font_size >= config.MIN_FONT_SIZE:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
            # 기본 폰트는 사이즈 조절이 안되므로 루프 중단
            break

        wrapped_text = _wrap_text(text, font, target_width)

        try:
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
            text_block_height = text_bbox[3] - text_bbox[1]
            text_block_width = text_bbox[2] - text_bbox[0]
        except AttributeError:
            # 구버전 Pillow 호환
            text_block_height = font_size * len(wrapped_text.split('\n'))
            text_block_width = target_width

        target_area = target_width * target_height
        text_area = text_block_width * text_block_height
        fill_ratio = (text_area / target_area) if target_area > 0 else 1

        # 높이, 너비, 면적 조건을 모두 만족하면 축소 중단
        if text_block_width <= target_width and text_block_height <= target_height and fill_ratio < config.MAX_AREA_FILL_RATIO:
            break

        font_size -= 2

    # 폰트 크기 확대
    if config.FONT_UPSCALE_IF_TOO_SMALL and font:
        try:
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
            text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
            target_area = target_width * target_height

            if target_area > 0 and (text_area / target_area) < config.FONT_AREA_FILL_RATIO:
                last_good_font = font
                last_good_wrapped_text = wrapped_text

                while font_size < config.MAX_FONT_SIZE:
                    font_size += 2
                    try:
                        temp_font = ImageFont.truetype(font_path, font_size)
                    except IOError:
                        break

                    temp_wrapped_text = _wrap_text(text, temp_font, target_width)
                    temp_bbox = draw.multiline_textbbox((0, 0), temp_wrapped_text, font=temp_font, align="center")

                    temp_block_width = temp_bbox[2] - temp_bbox[0]
                    temp_block_height = temp_bbox[3] - temp_bbox[1]
                    temp_area = temp_block_width * temp_block_height
                    fill_ratio = (temp_area / target_area) if target_area > 0 else 1

                    if temp_block_width > target_width or temp_block_height > target_height or fill_ratio >= config.MAX_AREA_FILL_RATIO:
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


def draw_translations(image_rgb, page_data):
    """
    말풍선 비율에 따라 가로/세로 쓰기를 자동으로 전환하여 식자합니다.
    """
    print("식자 작업을 시작합니다...")
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)

    for bubble in page_data['speech_bubbles']:
        if not (bubble.get('translated_text') and bubble.get('font_size')):
            continue

        initial_font_size = bubble['font_size']
        translated_text = bubble['translated_text']
        font_style = bubble.get('font_style', 'standard')
        font_path = config.FONT_MAP.get(font_style, config.DEFAULT_FONT_PATH)
        bubble_box = bubble['bubble_box']
        text_box = bubble['text_box']

        bubble_width = bubble_box[2] - bubble_box[0]
        bubble_height = bubble_box[3] - bubble_box[1]

        is_vertical = (config.ENABLE_VERTICAL_TEXT and
                       ' ' not in translated_text and
                       bubble_width > 0 and
                       (bubble_height / bubble_width >= config.VERTICAL_TEXT_THRESHOLD))

        if is_vertical:
            original_text_height = text_box[3] - text_box[1]
            target_height = original_text_height * (1 + config.VERTICAL_TOLERANCE_RATIO)
            font, wrapped_text = _find_best_fit_font_vertical(draw, translated_text, initial_font_size, target_height,
                                                              font_path)

            if font:
                center_x = (bubble_box[0] + bubble_box[2]) // 2
                center_y = (bubble_box[1] + bubble_box[3]) // 2
                draw.text((center_x, center_y), wrapped_text, font=font, fill=(0, 0, 0), anchor="mm", align="center")
        else:
            target_width = bubble_width * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
            original_text_height = text_box[3] - text_box[1]
            target_height = original_text_height * (1 + config.VERTICAL_TOLERANCE_RATIO)

            font, wrapped_text = _find_best_fit_font(draw, translated_text, initial_font_size, target_width,
                                                     target_height, font_path)

            if font:
                attachment = bubble.get('attachment', 'none')
                align, anchor, center_x, center_y = _get_alignment_for_bubble(attachment, text_box, bubble_box)
                draw.text((center_x, center_y), wrapped_text, font=font, fill=(0, 0, 0), anchor=anchor, align=align)

    for freeform in page_data['freeform_texts']:
        if not (freeform.get('translated_text') and freeform.get('font_size')):
            continue

        initial_font_size = freeform['font_size']
        translated_text = freeform['translated_text']
        font_style = freeform.get('font_style', 'standard')
        font_path = config.FONT_MAP.get(font_style, config.DEFAULT_FONT_PATH)
        box = freeform['text_box']

        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2))
        target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO)

        font, wrapped_text = _find_best_fit_font(draw, translated_text, initial_font_size, target_width, target_height,
                                                 font_path)

        if font:
            center_x = (box[0] + box[2]) // 2
            center_y = box[1]
            draw.text((center_x, center_y), wrapped_text, font=font, fill=config.FREEFORM_FONT_COLOR,
                      stroke_width=config.FREEFORM_STROKE_WIDTH, stroke_fill=config.FREEFORM_STROKE_COLOR,
                      anchor="ma", align="center")

    print("식자 작업 완료.")
    return np.array(img_pil)


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