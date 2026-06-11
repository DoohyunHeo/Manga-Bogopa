"""폰트 속성 산출: 글씨체는 모델 분류, 크기·기울기는 잉크 기하 측정.

크기 회귀 모델은 측정 대비 오차가 커 제거되었다 (glyph_metrics 참고).

- TTA(테스트 타임 증강) 평균으로 판정 안정화
- 신뢰도 기반 폴백 (확신 없으면 standard/narration)
- 세로 일본어 크롭의 후리가나 컬럼 제거 (크기 예측 정밀화, 폰트 모델 입력 전용)
"""
import logging
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

from src import config
from src.glyph_metrics import estimate_glyph_height, estimate_text_angle
from src.utils import Letterbox

logger = logging.getLogger(__name__)

# Internal model/heuristic constants (not user-tunable).
FALLBACK_FONT_SIZE = 20
_FONT_MODEL_INPUT_SIZE = (224, 224)
_CHAR_RATIO_FONT_SIZE_GAIN = 1.18
_STYLE_FALLBACK_MAX_ANGLE = 15.0
_STYLE_SPECIAL_MIN_CONFIDENCE = {
    "standard": 0.0,
    "pop": 0.46,
    "shouting": 0.42,
    "handwriting": 0.40,
    "angry": 0.38,
    "cute": 0.38,
    "scared": 0.38,
    "embarrassment": 0.36,
    "narration": 0.34,
}


def _cuda_autocast_context():
    if config.DEVICE != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def strip_furigana_column(crop_pil):
    """Remove a narrow side column (likely furigana) from a vertical manga crop.

    Conservative — returns the original crop if any of the guards fail:
    - feature disabled
    - not a vertical-oriented crop
    - too small to analyze
    - only one dark band detected
    - the widest band already dominates the crop
    - no sufficiently wide gap between bands

    Rationale: for vertical Japanese manga text, furigana sits in a narrow column
    to the right of the main kanji column. Cropping it away gives the font-size
    model a cleaner signal. OCR input stays untouched.
    """
    if not getattr(config, "VERTICAL_FURIGANA_STRIP_ENABLED", True):
        return crop_pil

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

    min_gap = max(2, int(w * float(getattr(config, "VERTICAL_FURIGANA_MIN_GAP_RATIO", 0.08))))
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


def _measure_font_size(item):
    """잉크 기하 측정으로 글자 크기를 결정합니다.

    측정은 원시 크롭 우선 — 후리가나 제거 크롭은 다컬럼 본문에서 본문
    컬럼이 잘릴 수 있다. 측정 불가(극소 크롭·잉크 없음) 시 크롭 높이의
    70%를 휴리스틱으로 쓴다.
    """
    crop = item.get("crop_raw") or item.get("crop")
    if crop is None:
        return FALLBACK_FONT_SIZE, None

    measured_height = estimate_glyph_height(crop)
    crop_height = max(1, crop.height)
    if measured_height > 0:
        font_size = max(1, int(round(measured_height * _CHAR_RATIO_FONT_SIZE_GAIN)))
        return font_size, measured_height / crop_height

    logger.debug("[size-fallback] 측정 불가 -> 크롭높이x0.7 휴리스틱 (crop %sx%s)",
                 crop.width, crop.height)
    return max(1, int(round(crop_height * 0.7))), None


def _build_font_model_tta_views(crop: Image.Image):
    base = crop.convert("RGB")
    if not getattr(config, "FONT_MODEL_TTA_ENABLED", True):
        return [base]

    variants = max(1, int(getattr(config, "FONT_MODEL_TTA_VARIANTS", 3)))
    views = [base]

    if variants >= 2:
        contrast_view = ImageOps.autocontrast(base)
        contrast_view = ImageEnhance.Contrast(contrast_view).enhance(1.08)
        views.append(contrast_view)

    if variants >= 3:
        sharp_view = ImageEnhance.Sharpness(base).enhance(1.15)
        views.append(sharp_view)

    return views[:variants]


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


def _choose_style_name(font_model, style_probs_row, angle_deg, expressive_logits_row=None, class_name="text"):
    """Select the final font style for a text item.

    Handles both the generic confidence-based fallback (for any class) and the
    free_text specialization where "standard" is treated as narration.
    """
    style_probs_cpu = style_probs_row.detach().cpu()
    top_k = min(2, int(style_probs_cpu.shape[0]))
    top_values, top_indices = torch.topk(style_probs_cpu, k=top_k)
    top_confidence = float(top_values[0])
    second_confidence = float(top_values[1]) if top_k > 1 else 0.0
    confidence_margin = top_confidence - second_confidence

    predicted_style = _resolve_style_name(font_model, int(top_indices[0]))
    expressive_confidence = None
    if expressive_logits_row is not None and getattr(font_model, "has_expressive_head", False):
        expressive_confidence = float(torch.sigmoid(expressive_logits_row.detach().cpu()))

    fallback_target = _freeform_fallback_style(class_name)
    fallback_enabled = getattr(config, "FONT_STYLE_FALLBACK_ENABLED", True)

    if predicted_style == "standard":
        return fallback_target, top_confidence, expressive_confidence
    if not fallback_enabled:
        # Generic fallback is off, but freeform confidence rule still applies.
        if _freeform_needs_fallback(class_name, top_confidence):
            return fallback_target, top_confidence, expressive_confidence
        return predicted_style, top_confidence, expressive_confidence

    low_confidence_threshold = float(getattr(config, "FONT_STYLE_LOW_CONFIDENCE_THRESHOLD", 0.24))
    low_margin_threshold = float(getattr(config, "FONT_STYLE_LOW_MARGIN_THRESHOLD", 0.04))
    expressive_prob_threshold = float(getattr(config, "FONT_STYLE_EXPRESSIVE_PROB_THRESHOLD", 0.55))
    style_specific_threshold = float(_STYLE_SPECIAL_MIN_CONFIDENCE.get(predicted_style, low_confidence_threshold))

    triggered_fallback = (
        (expressive_confidence is not None and expressive_confidence < expressive_prob_threshold)
        or top_confidence < low_confidence_threshold
        or top_confidence < style_specific_threshold
        or (confidence_margin < low_margin_threshold and abs(float(angle_deg)) <= _STYLE_FALLBACK_MAX_ANGLE)
        or _freeform_needs_fallback(class_name, top_confidence)
    )

    if triggered_fallback:
        return fallback_target, top_confidence, expressive_confidence
    return predicted_style, top_confidence, expressive_confidence


def _freeform_fallback_style(class_name):
    """free_text items fall back to narration (which IS their "standard")."""
    return "narration" if class_name == "free_text" else "standard"


def _freeform_needs_fallback(class_name, top_confidence):
    """Freeform items have a stricter confidence floor to avoid weak style picks."""
    if class_name != "free_text":
        return False
    min_confidence = float(getattr(config, "FREEFORM_STYLE_MIN_CONFIDENCE", 0.70))
    return top_confidence < min_confidence


def predict_font_properties(font_appearance_model, legacy_font_model, text_items, page_heights=None):
    """Predict font style/angle (모델) + size (잉크 기하 측정 — 크기 모델 폐기)."""
    style_angle_model = font_appearance_model or legacy_font_model
    if not style_angle_model:
        props = []
        for item in text_items:
            font_size, font_char_ratio = _measure_font_size(item)
            crop = item.get("crop_raw") or item.get("crop")
            measured_angle = estimate_text_angle(crop) if crop is not None else None
            props.append({
                "font_size": font_size,
                "angle": int(round(measured_angle)) if measured_angle is not None else 0,
                "font_style": "standard",
                "font_char_ratio": font_char_ratio,
                "font_stroke_ratio": None,
                "font_style_confidence": None,
                "expressive_confidence": None,
            })
        return props

    logger.info(
        "Analyzing %s text crops with font model: style_angle=%s (size는 잉크 측정)",
        len(text_items),
        getattr(style_angle_model, "checkpoint_path", None),
    )
    transform = transforms.Compose([
        Letterbox(_FONT_MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_props = []
    font_batch_size = config.FONT_MODEL_BATCH_SIZE
    for i in tqdm(range(0, len(text_items), font_batch_size), desc="Font Model"):
        batch_items = text_items[i:i + font_batch_size]
        batch_views = [_build_font_model_tta_views(item["crop"]) for item in batch_items]
        num_views = len(batch_views[0]) if batch_views else 1

        with torch.inference_mode():
            style_prob_sum = None
            angle_sin_sum = None
            angle_cos_sum = None
            expressive_logit_sum = None

            for view_idx in range(num_views):
                image_tensors = torch.stack([
                    transform(views[view_idx]) for views in batch_views
                ]).to(config.DEVICE)

                style_outputs = None
                if style_angle_model is not None:
                    with _cuda_autocast_context():
                        style_outputs = style_angle_model(image_tensors)

                    style_probs = torch.softmax(style_outputs["style"], dim=1)
                    angle_radians = torch.deg2rad(style_outputs["angle"].float())
                    angle_sin = torch.sin(angle_radians)
                    angle_cos = torch.cos(angle_radians)
                    expressive_logits = style_outputs.get("expressive")

                    style_prob_sum = style_probs if style_prob_sum is None else style_prob_sum + style_probs
                    angle_sin_sum = angle_sin if angle_sin_sum is None else angle_sin_sum + angle_sin
                    angle_cos_sum = angle_cos if angle_cos_sum is None else angle_cos_sum + angle_cos
                    if expressive_logits is not None:
                        expressive_logit_sum = expressive_logits if expressive_logit_sum is None else expressive_logit_sum + expressive_logits


            pred_style_indices = None
            pred_angles = None
            avg_style_probs = None
            if style_prob_sum is not None and angle_sin_sum is not None and angle_cos_sum is not None:
                avg_style_probs = style_prob_sum / num_views
                pred_style_indices = torch.argmax(avg_style_probs, dim=1).cpu().numpy()
                pred_angles = torch.rad2deg(
                    torch.atan2(angle_sin_sum / num_views, angle_cos_sum / num_views)
                ).cpu().numpy()

        for j in range(len(batch_items)):
            batch_class_name = batch_items[j].get("class_name", "text")
            style_name = _freeform_fallback_style(batch_class_name)
            style_confidence = None
            expressive_confidence = None
            if pred_style_indices is not None:
                averaged_expressive = (expressive_logit_sum / num_views)[j] if expressive_logit_sum is not None else None
                style_name, style_confidence, expressive_confidence = _choose_style_name(
                    style_angle_model,
                    avg_style_probs[j],
                    pred_angles[j],
                    averaged_expressive,
                    class_name=batch_class_name,
                )

            font_stroke_ratio = None
            font_size, font_char_ratio = _measure_font_size(batch_items[j])
            # 각도: 투영 선명도 측정 우선, 구조가 부족한 크롭(글자 1~2개)만
            # 모델 예측으로 폴백.
            crop_for_angle = batch_items[j].get("crop_raw") or batch_items[j].get("crop")
            measured_angle = estimate_text_angle(crop_for_angle) if crop_for_angle is not None else None
            if measured_angle is not None:
                final_angle = int(round(measured_angle))
            else:
                final_angle = int(round(pred_angles[j])) if pred_angles is not None else 0

            all_props.append({
                "font_size": font_size,
                "angle": final_angle,
                "font_style": style_name,
                "font_char_ratio": font_char_ratio,
                "font_stroke_ratio": font_stroke_ratio,
                "font_style_confidence": style_confidence,
                "expressive_confidence": expressive_confidence,
            })

    return all_props
