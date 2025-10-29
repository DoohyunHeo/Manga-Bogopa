import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm
from typing import List

from src import config
from src.data_models import PageData


def inpaint_pages_in_batch(models, all_page_data: List[PageData]) -> List[np.ndarray]:
    """
    여러 페이지의 모든 텍스트 영역을 하나의 배치로 Inpaint합니다.
    """
    lama_model = models['inpainting']
    all_patches_to_inpaint = []
    all_patches_metadata = []
    
    # 원본 이미지를 복사하여 최종 결과물로 사용할 리스트를 초기화합니다.
    inpainted_pages = [p.image_rgb.copy() for p in all_page_data]

    tqdm.write("모든 페이지에서 Inpaint할 영역을 수집 중...")
    for page_idx, page_data in enumerate(all_page_data):
        all_coords_to_erase = []
        for bubble in page_data.speech_bubbles:
            all_coords_to_erase.append(bubble.text_element.text_box)
        for ff_text in page_data.freeform_texts:
            all_coords_to_erase.append(ff_text.text_box)

        if not all_coords_to_erase:
            continue

        full_page_mask = create_mask_from_coords(
            page_data.image_rgb, all_coords_to_erase, padding=config.INPAINT_MASK_PADDING
        )
        img_h, img_w = page_data.image_rgb.shape[:2]

        for coords in all_coords_to_erase:
            x1, y1, x2, y2 = map(int, coords)
            pad = config.INPAINT_CONTEXT_PADDING
            ctx_x1, ctx_y1 = max(0, x1 - pad), max(0, y1 - pad)
            ctx_x2, ctx_y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)

            context_patch = page_data.image_rgb[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
            patch_mask = full_page_mask[ctx_y1:ctx_y2, ctx_x1:ctx_x2]

            all_patches_to_inpaint.append((context_patch, patch_mask))
            all_patches_metadata.append({
                'page_idx': page_idx,
                'coords': (ctx_x1, ctx_y1, ctx_x2, ctx_y2)
            })

    if not all_patches_to_inpaint:
        tqdm.write("Inpaint할 텍스트가 없습니다.")
        return inpainted_pages

    # 모든 페이지의 모든 패치를 한 번에 처리
    inpainted_patches = erase_patches_in_batch(lama_model, all_patches_to_inpaint)

    tqdm.write("Inpaint된 패치를 원본 페이지에 다시 적용 중...")
    for i, patch_meta in enumerate(tqdm(all_patches_metadata, desc="Applying Patches")):
        page_idx = patch_meta['page_idx']
        ctx_x1, ctx_y1, ctx_x2, ctx_y2 = patch_meta['coords']
        inpainted_patch = inpainted_patches[i]

        h, w = ctx_y2 - ctx_y1, ctx_x2 - ctx_x1
        if inpainted_patch.shape[0] != h or inpainted_patch.shape[1] != w:
            inpainted_patch = cv2.resize(inpainted_patch, (w, h), interpolation=cv2.INTER_LANCZOS4)

        inpainted_pages[page_idx][ctx_y1:ctx_y2, ctx_x1:ctx_x2] = inpainted_patch

    return inpainted_pages


def erase_patches_in_batch(lama_model, patch_mask_list, target_size=512):
    """(이미지, 마스크) 조각 리스트를 받아 일괄적으로 텍스트를 제거합니다."""
    if not patch_mask_list:
        return []

    tqdm.write(f"총 {len(patch_mask_list)}개의 텍스트 조각을 미니 배치로 나누어 Inpainting 시작...")

    all_output_patches = []
    batch_size = 8  # 하드웨어에 맞게 조절

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

    tqdm.write("배치 Inpainting 완료.")
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


def inpaint_image_with_lama(lama_model, image_rgb, mask):
    """
    LaMa 모델을 사용하여 단일 이미지의 마스크된 영역을 Inpaint합니다.
    """
    # 이미지와 마스크를 PIL 이미지로 변환
    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask).convert("L")

    # 모델 입력에 맞게 리사이즈
    resized_image = image_pil.resize((512, 512), Image.Resampling.LANCZOS)
    resized_mask = mask_pil.resize((512, 512), Image.Resampling.NEAREST)

    # 텐서로 변환
    img_tensor = to_tensor(resized_image).unsqueeze(0).to(lama_model.device)
    mask_tensor = to_tensor(resized_mask).unsqueeze(0).to(lama_model.device)

    # Inpainting 실행
    with torch.no_grad():
        inpainted_tensor = lama_model.model(img_tensor, mask_tensor)

    # 결과 이미지를 다시 원본 크기로 리사이즈
    inpainted_pil = to_pil_image(inpainted_tensor.squeeze(0).cpu())
    inpainted_pil = inpainted_pil.resize(image_pil.size, Image.Resampling.LANCZOS)

    return np.array(inpainted_pil)