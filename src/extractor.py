"""Pass 1 추출 오케스트레이션: 탐지 → 크롭 → OCR → 폰트 분석 → 페이지 구조화.

세부 구현은 책임별 모듈에 있다:
- src/detection.py      탐지 + 박스 병합
- src/font_analysis.py  폰트 모델 추론 (스타일/기울기/크기)
- src/page_structure.py 말풍선 매칭 + PageData 구조화
"""
import logging
import re

import cv2
import numpy as np
from PIL import Image

# pass1_stage 등 기존 호출부 호환을 위해 façade로 재노출한다.
from src.detection import detect_objects, merge_text_boxes  # noqa: F401
from src.font_analysis import predict_font_properties, strip_furigana_column
from src.page_structure import structure_page_data  # noqa: F401
from src.data_models import TextElement

logger = logging.getLogger(__name__)


def _prepare_crops(text_items, batch_images_rgb, batch_paths):
    """Prepare crop images for OCR and font analysis.

    OCR uses the original crop (manga-ocr was trained with furigana present).
    Font-size/style model uses a furigana-stripped variant so that a narrow
    phonetic-reading column doesn't skew size predictions.
    """
    crops_for_ocr = []
    for item in text_items:
        image_rgb = batch_images_rgb[item["page_idx"]]
        coords = item["box"].astype(int)
        original_crop_pil = Image.fromarray(image_rgb[coords[1]:coords[3], coords[0]:coords[2]])
        crops_for_ocr.append(original_crop_pil)
        item["crop"] = strip_furigana_column(original_crop_pil)
        # 크기 측정은 원시 크롭 사용: 후리가나 제거 크롭은 다컬럼 본문에서
        # 본문 컬럼이 잘릴 수 있고, 측정기는 후리가나 컬럼을 자체 필터링한다.
        item["crop_raw"] = original_crop_pil

    return crops_for_ocr


def extract_text_properties(models, batch_images_rgb, text_items, batch_paths):
    """Run OCR and font-property prediction for text boxes."""
    if not text_items:
        return []

    crops_for_ocr = _prepare_crops(text_items, batch_images_rgb, batch_paths)

    logger.info(f"Running batch OCR for {len(crops_for_ocr)} text crops...")
    all_ocr_results = models["ocr"](crops_for_ocr)

    page_heights = [img.shape[0] for img in batch_images_rgb] if batch_images_rgb else [1600]
    all_props = predict_font_properties(
        models.get("font_appearance_classifier"),
        models.get("font_classifier"),
        text_items,
        page_heights=page_heights,
    )

    processed_text_elements = []
    filtered_count = 0
    for i, item in enumerate(text_items):
        ocr_text = all_ocr_results[i]
        if not ocr_text:
            continue
        if not _is_valid_text(ocr_text, item["box"]):
            filtered_count += 1
            continue
        if _is_probably_artwork(item.get("crop_raw") or item.get("crop")):
            # 컬러 일러스트를 텍스트로 오인하면 그림을 지우고 글자를 식자하는
            # 사고가 난다. 만화 글자는 컬러 페이지에서도 저채도이므로
            # 고채도 크롭은 그림으로 판정해 버린다.
            logger.info(f"Filtered high-saturation (artwork-like) text box: '{ocr_text[:12]}'")
            filtered_count += 1
            continue
        element = TextElement(
            text_box=item["box"].tolist(),
            original_text=ocr_text,
            **all_props[i],
        )
        processed_text_elements.append({
            "element": element,
            "page_idx": item["page_idx"],
            "class_name": item["class_name"],
        })

    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} low-quality OCR text items.")

    return processed_text_elements


def _is_probably_artwork(crop_pil):
    """고채도 크롭 = 컬러 일러스트 오탐 판정 (만화 글자는 컬러 페이지에서도 저채도)."""
    if crop_pil is None:
        return False
    rgb = np.asarray(crop_pil.convert("RGB"))
    if rgb.size == 0:
        return False
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    return float(saturation.mean()) > 130.0


def _is_valid_text(text, box):
    """Validate OCR output."""
    text = text.strip()
    if len(text) == 0:
        return False
    if re.fullmatch(r"[\d\s\.\-\+\*/=@#%&:;,!?\(\)\[\]{}<>\"\'`~^|\\/_]+", text):
        return False
    if len(text) == 1 and not _is_cjk_or_kana(text[0]):
        return False
    x1, y1, x2, y2 = box[:4]
    if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
        return False
    if not any(_is_cjk_or_kana(c) or c.isalpha() for c in text):
        return False
    return True


def _is_cjk_or_kana(char):
    cp = ord(char)
    return (
        (0x3040 <= cp <= 0x309F)
        or (0x30A0 <= cp <= 0x30FF)
        or (0x4E00 <= cp <= 0x9FFF)
        or (0xAC00 <= cp <= 0xD7A3)
        or (0x3400 <= cp <= 0x4DBF)
        or (0xFF00 <= cp <= 0xFFEF)
    )


def process_image_batch(models, batch_images_rgb, batch_paths):
    """Process a batch of pages into PageData objects."""
    all_text_items, all_bubbles_by_page = detect_objects(models["detection"], batch_images_rgb)
    merged_text_items = merge_text_boxes(all_text_items)
    processed_text_elements = extract_text_properties(models, batch_images_rgb, merged_text_items, batch_paths)
    batch_page_data = structure_page_data(batch_paths, batch_images_rgb, all_bubbles_by_page, processed_text_elements)
    return batch_page_data
