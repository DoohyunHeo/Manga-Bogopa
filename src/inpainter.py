import logging
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

from src import config
from src.data_models import PageData


def _clip_coords_to_image(image, coords):
    """텍스트 박스를 이미지 경계로 자르고, 유효하지 않으면 None을 반환합니다."""
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = coords
    x1 = max(0, min(img_w, int(np.floor(x1))))
    y1 = max(0, min(img_h, int(np.floor(y1))))
    x2 = max(0, min(img_w, int(np.ceil(x2))))
    y2 = max(0, min(img_h, int(np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# 잔여 잉크 흡수 휴리스틱 (내부 상수)
_ABSORB_REACH_PX = 12          # 글자 마스크에서 이 거리 안에 닿아 있는 잉크를 후보로
_ABSORB_MAX_AREA_RATIO = 0.35  # 텍스트 박스 면적 대비 이보다 큰 덩어리는 그림으로 간주


def _absorb_residual_ink(image_rgb, full_page_mask, text_box, bubble_box):
    """말풍선 안에서 탐지 박스 곁에 붙은 미탐지 글자 조각을 마스크로 흡수합니다.

    후리가나 줄, 첨자, 박스를 살짝 벗어난 글자 끝처럼 탐지 박스 밖에 남은
    작은 잉크 덩어리가 인페인팅 후 검은 노이즈로 남는 것을 막는다.
    - 후보: 텍스트 마스크를 _ABSORB_REACH_PX 만큼 넓힌 영역과 겹치는 잉크 성분
    - 제외: 말풍선 크롭 가장자리에 닿는 성분(테두리 선), 큰 성분(그림)
    """
    img_h, img_w = image_rgb.shape[:2]
    bx1 = max(0, int(bubble_box[0]))
    by1 = max(0, int(bubble_box[1]))
    bx2 = min(img_w, int(bubble_box[2]))
    by2 = min(img_h, int(bubble_box[3]))
    if bx2 - bx1 < 8 or by2 - by1 < 8:
        return

    crop = image_rgb[by1:by2, bx1:bx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    mask_crop = full_page_mask[by1:by2, bx1:bx2]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_ABSORB_REACH_PX * 2 + 1, _ABSORB_REACH_PX * 2 + 1)
    )
    seed = cv2.dilate(mask_crop, kernel)

    box_area = max(1.0, float((text_box[2] - text_box[0]) * (text_box[3] - text_box[1])))
    max_component_area = box_area * _ABSORB_MAX_AREA_RATIO

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    crop_h, crop_w = ink.shape
    absorbed = np.zeros_like(ink)
    absorbed_any = False
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area > max_component_area:
            continue
        # 말풍선 테두리 선은 크롭 가장자리에 닿으므로 제외
        if x <= 1 or y <= 1 or x + w >= crop_w - 1 or y + h >= crop_h - 1:
            continue
        component = labels == label
        if not (seed[component] > 0).any():
            continue
        if (mask_crop[component] > 0).all():
            continue  # 이미 전부 마스크 안
        absorbed[component] = 255
        absorbed_any = True

    if not absorbed_any:
        return

    # 흡수된 조각 주변 약간의 여유 (안티앨리어싱 가장자리) — 단, 흡수분만 팽창시키고
    # 말풍선 테두리 보호 링(바깥 2px)은 건드리지 않는다.
    absorbed = cv2.dilate(absorbed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    absorbed[:2, :] = 0
    absorbed[-2:, :] = 0
    absorbed[:, :2] = 0
    absorbed[:, -2:] = 0
    mask_crop[absorbed > 0] = 255


def _expand_box_within(text_box, pad, outer_box=None, border_keep=2):
    """텍스트 박스를 pad만큼 넓히되, outer_box(말풍선) 테두리는 침범하지 않습니다.

    탐지 박스가 글자에 빠듯하면 가장자리에 글자가 미세하게 남아 인페인팅 시
    검은 노이즈가 생기므로 넉넉하게 확장한다. 단:
    - 말풍선 테두리 안쪽 border_keep px까지만 확장 (테두리 선 보존)
    - 어떤 경우에도 원래 텍스트 박스보다 작아지지 않음 (잔상 방지 우선)
    """
    x1, y1, x2, y2 = (float(v) for v in text_box[:4])
    ex1, ey1, ex2, ey2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    if outer_box is not None:
        ox1, oy1, ox2, oy2 = (float(v) for v in outer_box[:4])
        ex1 = max(ex1, ox1 + border_keep)
        ey1 = max(ey1, oy1 + border_keep)
        ex2 = min(ex2, ox2 - border_keep)
        ey2 = min(ey2, oy2 - border_keep)
    return [min(ex1, x1), min(ey1, y1), max(ex2, x2), max(ey2, y2)]


def inpaint_pages_in_batch(models, all_page_data: List[PageData]) -> List[np.ndarray]:
    """
    여러 페이지의 모든 텍스트 영역을 하나의 배치로 Inpaint합니다.

    결과 합성은 마스크 영역(지운 글자 부분)에만 적용한다. 컨텍스트 패치의
    나머지 픽셀은 원본 그대로 유지되므로 글자 주변 그림이 열화되지 않는다.
    """
    lama_model = models['inpainting']
    all_patches_to_inpaint = []
    all_patches_metadata = []

    # 원본 이미지를 복사하여 최종 결과물로 사용할 리스트를 초기화합니다.
    inpainted_pages = [p.image_rgb.copy() for p in all_page_data]

    bubble_pad = max(0, int(getattr(config, "INPAINT_BUBBLE_MASK_PADDING", 8)))
    freeform_pad = max(0, int(config.INPAINT_MASK_PADDING))

    logger.info("모든 페이지에서 Inpaint할 영역을 수집 중...")
    for page_idx, page_data in enumerate(all_page_data):
        # (지울 박스, 합성 소유 영역) — 소유 영역 밖 픽셀은 이 패치가 절대 덮어쓰지 않는다.
        # 공유 마스크 때문에 큰 패치(축소→복원으로 화질이 떨어진 것)가 이웃 말풍선
        # 영역을 흐릿한 버전으로 덮어쓰는 사고를 막는다.
        all_coords_to_erase = []
        own_regions = []
        bubble_erase_entries = []
        for bubble in page_data.speech_bubbles:
            expanded_box = _expand_box_within(bubble.text_element.text_box, bubble_pad, bubble.bubble_box)
            bubble_erase_entries.append((expanded_box, bubble.bubble_box))
            all_coords_to_erase.append(expanded_box)
            own_regions.append([
                min(expanded_box[0], bubble.bubble_box[0]),
                min(expanded_box[1], bubble.bubble_box[1]),
                max(expanded_box[2], bubble.bubble_box[2]),
                max(expanded_box[3], bubble.bubble_box[3]),
            ])
        for ff_text in page_data.freeform_texts:
            expanded_box = _expand_box_within(ff_text.text_box, freeform_pad)
            all_coords_to_erase.append(expanded_box)
            own_regions.append(expanded_box)

        if not all_coords_to_erase:
            continue

        full_page_mask = create_mask_from_coords(
            page_data.image_rgb, all_coords_to_erase, padding=0
        )
        # 박스 곁에 남은 후리가나·첨자 같은 미탐지 글자 조각을 마스크로 흡수
        for expanded_box, bubble_box in bubble_erase_entries:
            _absorb_residual_ink(page_data.image_rgb, full_page_mask, expanded_box, bubble_box)
        img_h, img_w = page_data.image_rgb.shape[:2]

        for coords, own_region in zip(all_coords_to_erase, own_regions):
            clipped_coords = _clip_coords_to_image(page_data.image_rgb, coords)
            if clipped_coords is None:
                logger.debug(f"Skipping invalid inpaint box: {coords}")
                continue

            x1, y1, x2, y2 = clipped_coords
            # 컨텍스트는 박스 크기에 비례해 적응적으로 넓힌다. 구멍이 패치의
            # 절반을 넘으면 LaMa(FFC)가 주변 세로선 텍스처를 구멍 안으로
            # 환각(줄무늬)하므로, 큰 박스일수록 주변을 더 넓게 보여줘야 한다.
            pad = max(int(config.INPAINT_CONTEXT_PADDING), int(0.6 * max(x2 - x1, y2 - y1)))
            pad = min(pad, 256)
            ctx_x1, ctx_y1 = max(0, x1 - pad), max(0, y1 - pad)
            ctx_x2, ctx_y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
            if ctx_x2 <= ctx_x1 or ctx_y2 <= ctx_y1:
                logger.debug(f"Skipping empty inpaint context: {coords} -> {(ctx_x1, ctx_y1, ctx_x2, ctx_y2)}")
                continue

            context_patch = page_data.image_rgb[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
            patch_mask = full_page_mask[ctx_y1:ctx_y2, ctx_x1:ctx_x2]

            all_patches_to_inpaint.append((context_patch, patch_mask))
            all_patches_metadata.append({
                'page_idx': page_idx,
                'coords': (ctx_x1, ctx_y1, ctx_x2, ctx_y2),
                'own_region': own_region,
            })

    if not all_patches_to_inpaint:
        logger.info("Inpaint할 텍스트가 없습니다.")
        return inpainted_pages

    # 모든 페이지의 모든 패치를 한 번에 처리
    inpainted_patches = erase_patches_in_batch(lama_model, all_patches_to_inpaint)

    logger.info("Inpaint된 패치를 원본 페이지에 다시 적용 중...")
    for i, patch_meta in enumerate(tqdm(all_patches_metadata, desc="Applying Patches")):
        page_idx = patch_meta['page_idx']
        ctx_x1, ctx_y1, ctx_x2, ctx_y2 = patch_meta['coords']
        inpainted_patch = inpainted_patches[i]
        patch_mask = all_patches_to_inpaint[i][1]

        h, w = ctx_y2 - ctx_y1, ctx_x2 - ctx_x1
        if inpainted_patch.shape[0] != h or inpainted_patch.shape[1] != w:
            inpainted_patch = cv2.resize(inpainted_patch, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # 마스크 ∩ 자기 소유 영역 픽셀만 교체 — 글자 주변 원본 그림을 보존하고,
        # 다른 박스 영역을 (화질이 다른) 이 패치 결과로 덮어쓰지 않는다.
        region = inpainted_pages[page_idx][ctx_y1:ctx_y2, ctx_x1:ctx_x2]
        mask_bool = patch_mask > 127
        ox1, oy1, ox2, oy2 = patch_meta['own_region']
        own_bool = np.zeros_like(mask_bool)
        sy1 = max(0, int(oy1) - ctx_y1)
        sy2 = min(h, int(oy2) - ctx_y1)
        sx1 = max(0, int(ox1) - ctx_x1)
        sx2 = min(w, int(ox2) - ctx_x1)
        if sy2 > sy1 and sx2 > sx1:
            own_bool[sy1:sy2, sx1:sx2] = True
        mask_bool &= own_bool
        region[mask_bool] = inpainted_patch[mask_bool]

    return inpainted_pages


def _prepare_patch_canvas(patch_np, mask_np, target_size):
    """패치를 비율 유지한 채 target_size 정사각 캔버스에 배치합니다.

    - target_size 이하 패치: 리샘플링 없이 그대로 배치 (원본 화질 유지)
    - 초과 패치: 비율 유지 축소 후 배치
    이미지 패딩은 가장자리 복제(인페인팅 컨텍스트로 자연스러움), 마스크 패딩은 0.
    """
    h, w = patch_np.shape[:2]
    scale = min(1.0, target_size / max(h, w))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_small = cv2.resize(patch_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        new_w, new_h = w, h
        img_small, mask_small = patch_np, mask_np

    img_canvas = cv2.copyMakeBorder(
        img_small, 0, target_size - new_h, 0, target_size - new_w, cv2.BORDER_REPLICATE
    )
    mask_canvas = cv2.copyMakeBorder(
        mask_small, 0, target_size - new_h, 0, target_size - new_w,
        cv2.BORDER_CONSTANT, value=0,
    )
    return img_canvas, mask_canvas, (w, h, new_w, new_h, scale)


def erase_patches_in_batch(lama_model, patch_mask_list, target_size=512):
    """(이미지, 마스크) 조각 리스트를 받아 일괄적으로 텍스트를 제거합니다."""
    if not patch_mask_list:
        return []

    logger.info(f"총 {len(patch_mask_list)}개의 텍스트 조각을 미니 배치로 나누어 Inpainting 시작...")

    all_output_patches = []
    batch_size = config.INPAINT_BATCH_SIZE

    for i in tqdm(range(0, len(patch_mask_list), batch_size), desc="Inpainting Batches"):
        mini_batch = patch_mask_list[i:i + batch_size]

        img_canvases = []
        mask_canvases = []
        metas = []
        for patch_np, mask_np in mini_batch:
            img_canvas, mask_canvas, meta = _prepare_patch_canvas(patch_np, mask_np, target_size)
            img_canvases.append(img_canvas)
            mask_canvases.append(mask_canvas)
            metas.append(meta)

        img_batch = (
            torch.from_numpy(np.stack(img_canvases))
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
            .to(lama_model.device)
        )
        mask_batch = (
            torch.from_numpy((np.stack(mask_canvases) > 127).astype(np.float32))
            .unsqueeze(1)
            .to(lama_model.device)
        )

        with torch.inference_mode():
            inpainted_batch = lama_model.model(img_batch, mask_batch)

        out_np = (
            inpainted_batch.clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        for j, (w, h, new_w, new_h, scale) in enumerate(metas):
            crop = out_np[j][:new_h, :new_w]
            if scale < 1.0:
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
            all_output_patches.append(crop)

    logger.info("배치 Inpainting 완료.")
    return all_output_patches


def create_mask_from_coords(image, list_of_coords, padding=0):
    """
    좌표 리스트([x1, y1, x2, y2], ...)를 기반으로 Inpainting 마스크를 생성합니다.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for coords in list_of_coords:
        clipped_coords = _clip_coords_to_image(image, coords)
        if clipped_coords is None:
            continue

        x1, y1, x2, y2 = clipped_coords

        # 패딩 적용 및 이미지 경계 확인
        padded_x1 = max(0, x1 - padding)
        padded_y1 = max(0, y1 - padding)
        padded_x2 = min(image.shape[1], x2 + padding)
        padded_y2 = min(image.shape[0], y2 + padding)

        cv2.rectangle(mask, (padded_x1, padded_y1), (padded_x2, padded_y2), 255, -1)
    return mask
