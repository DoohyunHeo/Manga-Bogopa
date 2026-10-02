"""1단계 글자 속성 산출: 크기·글자 비율·기울기는 잉크 기하 측정 (glyph_metrics).

글씨체는 여기서 기본값(말풍선 안 standard, 말풍선 밖 narration)으로 두고, 식자 직전 책 단위 판정
(src/font_relative.py)이 원문 글씨 모양과 획 굵기로 정한다.
세로 일본어 크롭의 후리가나 컬럼 제거(strip_furigana_column)는 굵기 측정(font_relative.measure_weight) 전용이다.
"""
import logging
import re
from contextlib import nullcontext

import cv2
import numpy as np
import torch

from src import config
from src.glyph_metrics import count_glyphs, estimate_glyph_height, estimate_text_angle

logger = logging.getLogger(__name__)

# Internal heuristic constants (not user-tunable).
FALLBACK_FONT_SIZE = 20
_CHAR_RATIO_FONT_SIZE_GAIN = 1.18
# 막대 부호 — 장음·줄표·물결·느낌표·세로줄 (estimate_glyph_height의 bars)
_BAR_MARKS = re.compile("[ーｰ─━―—‐~〜～!！|｜]")
# 가나·한자 — 대사의 글자 수(estimate_glyph_height의 glyphs)는 이것만 센다
_KANA_KANJI = re.compile("[ぁ-ゖゝゞァ-ヺヽヾｦ-ｯｱ-ﾝ㐀-䶿一-鿿々〆]")


def _cuda_autocast_context():
    if config.DEVICE != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def strip_furigana_column(crop_pil):
    """Remove a narrow side column (likely furigana) from a vertical manga crop.

    Conservative — returns the original crop if any of the guards fail:
    - not a vertical-oriented crop
    - too small to analyze
    - only one dark band detected
    - the widest band already dominates the crop
    - no sufficiently wide gap between bands

    Rationale: for vertical Japanese manga text, furigana sits in a narrow column
    to the right of the main kanji column. Cropping it away gives the stroke-weight
    measurement (font_relative.measure_weight) a cleaner signal. OCR input and the
    size measurement use the original crop.
    """
    gray = np.array(crop_pil.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    if w >= h or h < 30 or w < 12:
        return crop_pil

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    col_density = binary.sum(axis=0, dtype=np.int32) / 255.0

    smooth_kernel = max(3, w // 10)
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1
    kernel = np.ones(smooth_kernel) / smooth_kernel
    smoothed = np.convolve(col_density, kernel, mode="same")

    peak = smoothed.max()
    if peak < 1.0:
        return crop_pil
    threshold = peak * 0.15
    is_dark = smoothed > threshold

    bands = []
    start = None
    for idx, flag in enumerate(is_dark):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            bands.append((start, idx))
            start = None
    if start is not None:
        bands.append((start, len(is_dark)))

    if len(bands) < 2:
        return crop_pil

    widest = max(bands, key=lambda b: b[1] - b[0])
    widest_w = widest[1] - widest[0]
    if widest_w >= w * 0.8:
        return crop_pil

    min_gap = max(2, int(w * float(config.VERTICAL_FURIGANA_MIN_GAP_RATIO)))
    # Require at least one gap around the widest band that's >= min_gap wide.
    sorted_bands = sorted(bands)
    widest_index = sorted_bands.index(widest)
    left_gap = widest[0] - sorted_bands[widest_index - 1][1] if widest_index > 0 else widest[0]
    right_gap = (
        sorted_bands[widest_index + 1][0] - widest[1]
        if widest_index < len(sorted_bands) - 1
        else w - widest[1]
    )
    if max(left_gap, right_gap) < min_gap:
        return crop_pil

    pad = 2
    left = max(0, widest[0] - pad)
    right = min(w, widest[1] + pad)
    if right - left < 10:
        return crop_pil
    return crop_pil.crop((left, 0, right, h))


def _measure_font_size(crop, text, dialogue=False):
    """잉크 기하 측정으로 글자 크기를 결정합니다 — (크기, 잉크 높이 ÷ 크롭 높이).

    dialogue(말풍선 대사)면 글자 수는 가나·한자만 세고(없으면 count_glyphs), 막대 부호 조각을 빼고, 세로가 긴 크롭은
    세로쓰기로 잰다 — 말풍선 밖 글자(효과음·간판)는 글자 인식이 읽은 글과 잉크 조각이 덜 맞고 가로 글자도 많아 그대로 잰다.
    측정 불가(극소 크롭·잉크 없음) 시 크롭 높이의 70%를 휴리스틱으로 쓴다.
    """
    if crop is None:
        return FALLBACK_FONT_SIZE, None

    if dialogue and text:
        glyphs, bars = len(_KANA_KANJI.findall(text)) or count_glyphs(text), len(_BAR_MARKS.findall(text))
    else:
        glyphs, bars = (count_glyphs(text) if text else None), 0
    measured_height = estimate_glyph_height(crop, glyphs, bars=bars, upright=dialogue)
    crop_height = max(1, crop.height)
    if measured_height > 0:
        font_size = max(1, int(round(measured_height * _CHAR_RATIO_FONT_SIZE_GAIN)))
        return font_size, measured_height / crop_height

    logger.debug("[size-fallback] 측정 불가 -> 크롭높이x0.7 휴리스틱 (crop %sx%s)",
                 crop.width, crop.height)
    return max(1, int(round(crop_height * 0.7))), None


def _resolve_style_name(font_model, predicted_index):
    style_mapping = getattr(font_model, "style_mapping", {}) or {}
    if not style_mapping:
        return "standard"

    index_int = int(predicted_index)
    if index_int in style_mapping:
        return style_mapping[index_int]

    index_str = str(index_int)
    if index_str in style_mapping:
        return style_mapping[index_str]

    return "standard"


# 최종 글씨체를 고른 이유 (font_style_reason) — 모델 판정과 뒤에 덧붙인 규칙을 따로 볼 수 있게 한다
STYLE_REASONS = (
    "standard",         # 1단계 기본값 (말풍선 안 standard, 말풍선 밖 narration) — 책 단위 판정 전
    # 아래는 식자 직전 책 단위 판정 (src/font_relative.py)·식자 (src/page_drawer.py) — 대사집에는 남지 않는다
    "shape_model",      # 모양 6분류 모델 1위를 그대로 씀
    "shape_threshold_bold",  # 장식 글씨지만 확신이 모자라 굵은 보통체로
    "mincho_threshold", # 가는명조지만 확신이 모자라 평문체로
    "handwriting_threshold",  # 손글씨지만 확신이 모자라 평문체로
    "mincho_setting",  # 가는명조지만 설정 '나레이션체를 평문체로'로 평문체
    "bold_weight",      # 평문체인데 책 기준보다 획이 굵어 외침 글꼴로
    "screen_group",     # 한 화면에 쌓인 채팅 줄을 가장 많은 글씨체로 맞춤
)


def _freeform_fallback_style(class_name):
    """free_text items fall back to narration (which IS their "standard")."""
    return "narration" if class_name == "free_text" else "standard"


def measure_font_properties(text_items):
    """조각마다 크기·글자 비율·기울기를 잉크로 잰다 (TextElement 필드명과 1:1 대응하는 dict 목록).

    기울기를 잴 수 없는 크롭(글자 1~2개)은 0. 글씨체는 기본값으로 둔다 — 책 단위 판정이 다시 정한다.
    """
    all_props = []
    for item in text_items:
        crop = item.get("crop")
        font_size, font_char_ratio = _measure_font_size(crop, item.get("ocr_text"),
                                                        dialogue=item.get("class_name", "text") == "text")
        all_props.append({
            "font_size": font_size,
            "angle": int(round(estimate_text_angle(crop))) if crop is not None else 0,
            "font_style": _freeform_fallback_style(item.get("class_name", "text")),
            "font_char_ratio": font_char_ratio,
            "font_style_reason": "standard",
        })
    return all_props
