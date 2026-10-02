"""Pass 1 추출 오케스트레이션: 탐지 → 크롭 → OCR → 글자 크기 재기 → 페이지 구조화.

세부 구현은 책임별 모듈에 있다:
- src/detection.py      탐지 + 박스 병합
- src/font_analysis.py  크기·기울기 잉크 측정
- src/page_structure.py 말풍선 매칭 + PageData 구조화
"""
import logging
import re
import time

import cv2
import numpy as np
from PIL import Image

# pass1_stage 등 기존 호출부 호환을 위해 façade로 재노출한다.
from src.detection import detect_objects, merge_text_boxes  # noqa: F401
from src.font_analysis import measure_font_properties
from src.furigana import read_furigana
from src.page_structure import structure_page_data  # noqa: F401
from src.data_models import TextElement
from src.text_filters import is_doubtful_supplement

logger = logging.getLogger(__name__)


def _prepare_crops(text_items, batch_images_rgb):
    """글자 상자마다 원본 크롭(item["crop"])을 만든다 — 글자 읽기와 크기·기울기 측정이 같이 쓴다.

    manga-ocr는 후리가나가 있는 그대로 배웠고, 크기 측정기는 후리가나 컬럼을 스스로 거른다.
    """
    crops = []
    for item in text_items:
        image_rgb = batch_images_rgb[item["page_idx"]]
        coords = item["box"].astype(int)
        item["crop"] = Image.fromarray(image_rgb[coords[1]:coords[3], coords[0]:coords[2]])
        crops.append(item["crop"])
    return crops


def extract_text_properties(models, batch_images_rgb, text_items, stats=None, clock=None):
    """Run OCR and the ink size/angle measurement for text boxes.

    stats(dict)가 오면 걸러낸 이유별 개수를 더한다: 빈 판독(ocr_empty), 글자가 아닌 판독
    (ocr_invalid), 그림으로 판단(artwork). clock(StageClock)이 오면 읽기·글자 크기 재기 시간을 더한다.
    """
    stats = stats if stats is not None else {}
    if not text_items:
        return []

    crops_for_ocr = _prepare_crops(text_items, batch_images_rgb)

    logger.info(f"Running batch OCR for {len(crops_for_ocr)} text crops...")
    started = time.perf_counter()
    all_ocr_results = models["ocr"](crops_for_ocr)
    readings = getattr(models["ocr"], "last_readings", None) or [None] * len(text_items)
    # 후리가나는 읽기 모델이 빼고 읽으므로 따로 읽어 번역 요청에 싣는다 (인명 읽기 — furigana 참고)
    ruby = read_furigana(getattr(models["ocr"], "manga", models["ocr"]), crops_for_ocr, all_ocr_results)
    if clock is not None:
        clock.add("글자 읽기", time.perf_counter() - started)

    for item, ocr_text in zip(text_items, all_ocr_results):
        item["ocr_text"] = ocr_text  # 크기 측정이 글자 수를 본다 (글자 1~3개 크롭의 바탕 부스러기 거르기)
    started = time.perf_counter()
    all_props = measure_font_properties(text_items)
    if clock is not None:
        clock.add("글자 크기 재기", time.perf_counter() - started)

    processed_text_elements = []
    filtered_count = 0
    for i, item in enumerate(text_items):
        ocr_text = all_ocr_results[i]
        if not ocr_text:
            stats["ocr_empty"] = stats.get("ocr_empty", 0) + 1
            continue
        if not _is_valid_text(ocr_text, item["box"]):
            stats["ocr_invalid"] = stats.get("ocr_invalid", 0) + 1
            filtered_count += 1
            continue
        if item.get("supplement") and is_doubtful_supplement(ocr_text, item.get("score", 1.0), readings[i]):
            # 기본 모델이 못 본 자리를 보완 모델이 낮은 점수로 잡고 가나 몇 자로만 읽혔다 — 그림 선일 공산이 크다
            logger.info(f"Filtered doubtful supplementary text box: '{ocr_text[:12]}' (score {item.get('score', 0):.2f})")
            stats["artwork"] = stats.get("artwork", 0) + 1
            filtered_count += 1
            continue
        if _is_probably_artwork(item.get("crop")):
            # 컬러 일러스트를 텍스트로 오인하면 그림을 지우고 글자를 식자하는
            # 사고가 난다. 만화 글자는 컬러 페이지에서도 저채도이므로
            # 고채도 크롭은 그림으로 판정해 버린다.
            logger.info(f"Filtered high-saturation (artwork-like) text box: '{ocr_text[:12]}'")
            stats["artwork"] = stats.get("artwork", 0) + 1
            filtered_count += 1
            continue
        element = TextElement(
            text_box=item["box"].tolist(),
            original_text=ocr_text,
            furigana=ruby[i],
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
