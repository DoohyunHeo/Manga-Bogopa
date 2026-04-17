"""Font-size fitting loops for horizontal and vertical text layout.

Public entry points:
    find_best_fit_font(text, initial_font_size, target_w, target_h, font_path, style, ...)
    find_best_fit_font_vertical(text, initial_font_size, target_w, target_h, font_path, style, ...)

Both return `(wrapped_text, font_size)`. The horizontal fitter searches over
wrap strategies and width ratios; the vertical fitter iterates sizes top-down.

Caps: predicted × MODEL_FONT_SIZE_CEILING_RATIO (hard upper bound),
      predicted × MODEL_FONT_SIZE_FLOOR_RATIO (soft lower bound via penalty).
"""
import logging
import math
import re

import numpy as np

from src import config
from src.text_renderer import measure_character_body_height_ratio, measure_line, measure_text
from src.text_wrapping import (
    LINE_HEAD_FORBIDDEN,
    LINE_TAIL_FORBIDDEN,
    TALL_BUBBLE_RATIO,
    build_wrap_candidates,
    visible_len,
)

logger = logging.getLogger(__name__)


def is_font_size_correction_enabled():
    return bool(getattr(config, "FONT_SIZE_CORRECTION_ENABLED", True))


def get_char_ratio_target(element):
    """Return the per-element char-height ratio target, or None if disabled/absent."""
    if not getattr(config, "FONT_CHAR_FIT_ENABLED", True):
        return None
    char_ratio = getattr(element, "font_char_ratio", None)
    if char_ratio is None:
        return None
    try:
        char_ratio = float(char_ratio)
    except (TypeError, ValueError):
        return None
    return char_ratio if char_ratio > 0 else None


def _get_font_search_start(initial_font_size, target_height):
    return max(
        config.MIN_FONT_SIZE,
        min(
            config.MAX_FONT_SIZE,
            max(int(round(target_height)), initial_font_size, config.MIN_READABLE_TEXT_SIZE + 2),
        ),
    )


def _get_model_font_upper_bound(initial_font_size):
    predicted_size = max(1, int(round(initial_font_size)))
    growth_ratio = max(1.0, float(config.MODEL_FONT_SIZE_CEILING_RATIO))
    capped_size = int(math.floor(predicted_size * growth_ratio))
    return max(
        config.MIN_FONT_SIZE,
        min(config.MAX_FONT_SIZE, max(predicted_size, capped_size)),
    )


def _find_best_font_size(text, font_path, font_size, target_width, target_height, wrap_fn, style, min_size=None):
    wrapped_text = text
    current_size = font_size
    minimum_size = max(config.MIN_FONT_SIZE, min_size or config.MIN_FONT_SIZE)
    best_candidate = None

    while current_size >= minimum_size:
        wrapped_text = wrap_fn(text, font_path, current_size, style, target_width)
        text_w, text_h = measure_text(wrapped_text, font_path, current_size, style)
        overflow = max(0.0, text_w - target_width) + max(0.0, text_h - target_height)
        if best_candidate is None or overflow < best_candidate[0]:
            best_candidate = (overflow, wrapped_text, current_size)
        if text_w <= target_width and text_h <= target_height:
            return wrapped_text, current_size, True
        current_size -= 1

    if best_candidate is not None:
        _, wrapped_text, current_size = best_candidate
        return wrapped_text, current_size, False

    fallback_text = wrap_fn(text, font_path, minimum_size, style, target_width)
    return fallback_text, minimum_size, False


def _score_wrapped_candidate(
    wrapped_text,
    font_path,
    font_size,
    style,
    target_width,
    target_height,
    bubble_ratio,
    char_ratio_target=None,
    char_ratio_reference_height=None,
):
    text_w, text_h = measure_text(wrapped_text, font_path, font_size, style)
    lines = [line for line in wrapped_text.split('\n') if line.strip()]
    if not lines:
        return float("-inf")

    line_widths = [measure_line(line, font_path, font_size, style) for line in lines]
    fit_overflow = max(0.0, text_w - target_width) + max(0.0, text_h - target_height)
    fill_ratio = (text_w * text_h) / max(target_width * target_height, 1)
    vertical_fill = text_h / max(target_height, 1)
    horizontal_fill = max(line_widths) / max(target_width, 1)
    target_fill_floor = 0.52 if bubble_ratio <= 0.9 else 0.46
    target_fill = min(0.76, max(config.FONT_AREA_FILL_RATIO, target_fill_floor))
    score = font_size * 4.0
    score -= fit_overflow * 2.5

    if fill_ratio < target_fill:
        score -= (target_fill - fill_ratio) * 26.0
    else:
        score -= (fill_ratio - target_fill) * 12.0

    if font_size < config.MIN_READABLE_TEXT_SIZE:
        score -= (config.MIN_READABLE_TEXT_SIZE - font_size) * 5.0

    if bubble_ratio <= 0.9:
        score -= max(0.0, 0.72 - vertical_fill) * 16.0
        score -= max(0.0, 0.78 - horizontal_fill) * 10.0
        if len(lines) == 1 and visible_len(lines[0]) >= 12:
            score -= 10.0
    else:
        score -= max(0.0, 0.58 - vertical_fill) * 8.0

    if len(lines) > 1:
        edge_avg = (line_widths[0] + line_widths[-1]) / 2
        middle_max = max(line_widths[1:-1], default=max(line_widths))
        if bubble_ratio >= TALL_BUBBLE_RATIO:
            score -= max(0.0, edge_avg - middle_max) / max(target_width, 1) * 12.0
        else:
            score -= np.std(line_widths) / max(target_width, 1) * 4.0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if idx > 0 and stripped[0] in LINE_HEAD_FORBIDDEN:
            score -= 8.0
        if idx < len(lines) - 1 and stripped[-1] in LINE_TAIL_FORBIDDEN:
            score -= 8.0
        if idx in (0, len(lines) - 1) and visible_len(stripped) <= 2 and len(lines) >= 3:
            score -= 6.0

    if char_ratio_target is not None and getattr(config, "FONT_CHAR_FIT_ENABLED", True):
        reference_height = (
            max(float(char_ratio_reference_height), 1.0)
            if char_ratio_reference_height is not None
            else target_height
        )
        measured_char_ratio = measure_character_body_height_ratio(
            wrapped_text,
            font_path,
            font_size,
            style,
            reference_height=reference_height,
        )
        char_rel_error = abs(measured_char_ratio - char_ratio_target) / max(char_ratio_target, 1e-4)
        score -= char_rel_error * float(getattr(config, "FONT_CHAR_SCORE_WEIGHT", 42.0))

    return score


def _evaluate_fit_candidates(
    text,
    start_size,
    target_width,
    target_height,
    font_path,
    style,
    bubble_ratio,
    width_ratios,
    wrap_fns,
    minimum_size,
    preferred_minimum_size,
    char_ratio_target,
    char_ratio_reference_height,
):
    best_fit_candidate = None
    best_relaxed_candidate = None

    for width_ratio in width_ratios:
        candidate_width = max(1.0, target_width * width_ratio)
        for wrap_fn in wrap_fns:
            wrapped_text, found_size, fits = _find_best_font_size(
                text,
                font_path,
                start_size,
                candidate_width,
                target_height,
                wrap_fn,
                style,
                min_size=minimum_size,
            )
            score = _score_wrapped_candidate(
                wrapped_text,
                font_path,
                found_size,
                style,
                target_width,
                target_height,
                bubble_ratio,
                char_ratio_target=char_ratio_target,
                char_ratio_reference_height=char_ratio_reference_height,
            )
            if found_size < preferred_minimum_size:
                score -= (preferred_minimum_size - found_size) * 12.0

            candidate = (score, wrapped_text, found_size, wrap_fn.__name__, width_ratio, fits)
            if fits:
                if best_fit_candidate is None or candidate[0] > best_fit_candidate[0]:
                    best_fit_candidate = candidate
            elif best_relaxed_candidate is None or candidate[0] > best_relaxed_candidate[0]:
                best_relaxed_candidate = candidate

    return best_fit_candidate, best_relaxed_candidate


def _width_ratios_for(bubble_ratio):
    width_ratios = [1.0, 0.9]
    if bubble_ratio <= 0.9:
        width_ratios.extend([0.82, 0.72])
    elif bubble_ratio >= TALL_BUBBLE_RATIO:
        width_ratios.extend([0.78, 0.66, 0.56])
    elif bubble_ratio >= 1.2:
        width_ratios.extend([0.84, 0.74])
    return list(dict.fromkeys(width_ratios))


def _select_fixed_size_wrap(
    text,
    initial_font_size,
    target_width,
    target_height,
    font_path,
    style,
    char_ratio_target=None,
    char_ratio_reference_height=None,
):
    fixed_size = max(1, int(round(initial_font_size)))
    bubble_ratio = target_height / max(target_width, 1)
    width_ratios = _width_ratios_for(bubble_ratio)
    wrap_fns = build_wrap_candidates(text, bubble_ratio)
    best_candidate = None

    for width_ratio in width_ratios:
        candidate_width = max(1.0, target_width * width_ratio)
        for wrap_fn in wrap_fns:
            wrapped_text = wrap_fn(text, font_path, fixed_size, style, candidate_width)
            score = _score_wrapped_candidate(
                wrapped_text,
                font_path,
                fixed_size,
                style,
                target_width,
                target_height,
                bubble_ratio,
                char_ratio_target=char_ratio_target,
                char_ratio_reference_height=char_ratio_reference_height,
            )
            candidate = (score, wrapped_text, wrap_fn.__name__, width_ratio)
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate

    if best_candidate is None:
        return text, fixed_size

    score, wrapped_text, wrap_name, width_ratio = best_candidate
    logger.debug(
        f"[font-fit-raw] text='{text[:20]}...' target=({target_width:.0f}x{target_height:.0f}) "
        f"fixed={fixed_size}, wrap={wrap_name}, width_ratio={width_ratio:.2f}, score={score:.2f}"
    )
    return wrapped_text, fixed_size


def find_best_fit_font(
    text,
    initial_font_size,
    target_width,
    target_height,
    font_path,
    style,
    char_ratio_target=None,
    char_ratio_reference_height=None,
):
    if not is_font_size_correction_enabled() and char_ratio_target is None:
        return _select_fixed_size_wrap(
            text,
            initial_font_size,
            target_width,
            target_height,
            font_path,
            style,
            char_ratio_target=char_ratio_target,
            char_ratio_reference_height=char_ratio_reference_height,
        )

    max_allowed_size = _get_model_font_upper_bound(initial_font_size)
    start_size = min(_get_font_search_start(initial_font_size, target_height), max_allowed_size)
    preferred_minimum_size = max(
        config.MIN_FONT_SIZE,
        min(config.MAX_FONT_SIZE, math.ceil(initial_font_size * config.MODEL_FONT_SIZE_FLOOR_RATIO)),
    )
    bubble_ratio = target_height / max(target_width, 1)
    width_ratios = _width_ratios_for(bubble_ratio)
    wrap_fns = build_wrap_candidates(text, bubble_ratio)

    best_candidate, relaxed_candidate = _evaluate_fit_candidates(
        text,
        start_size,
        target_width,
        target_height,
        font_path,
        style,
        bubble_ratio,
        width_ratios,
        wrap_fns,
        preferred_minimum_size,
        preferred_minimum_size,
        char_ratio_target,
        char_ratio_reference_height,
    )

    if best_candidate is None:
        best_candidate, fallback_candidate = _evaluate_fit_candidates(
            text,
            start_size,
            target_width,
            target_height,
            font_path,
            style,
            bubble_ratio,
            width_ratios,
            wrap_fns,
            config.MIN_FONT_SIZE,
            preferred_minimum_size,
            char_ratio_target,
            char_ratio_reference_height,
        )
        if best_candidate is None:
            best_candidate = fallback_candidate or relaxed_candidate

    if best_candidate is None:
        return text, max(config.MIN_FONT_SIZE, min(initial_font_size, config.MAX_FONT_SIZE))

    score, wrapped_text, found_size, wrap_name, width_ratio, fits = best_candidate
    logger.debug(
        f"[font-fit] text='{text[:20]}...' target=({target_width:.0f}x{target_height:.0f}) "
        f"initial={start_size} -> best={found_size}, wrap={wrap_name}, width_ratio={width_ratio:.2f}, "
        f"score={score:.2f}, fits={fits}, floor={preferred_minimum_size}, cap={max_allowed_size}"
    )
    return wrapped_text, found_size


def find_best_fit_font_vertical(
    text,
    initial_font_size,
    target_width,
    target_height,
    font_path,
    style,
    char_ratio_target=None,
    char_ratio_reference_height=None,
):
    if not is_font_size_correction_enabled() and char_ratio_target is None:
        fixed_size = max(1, int(round(initial_font_size)))
        text = text.replace("â‹¯", "ï¸™")
        tokens = re.findall(r'[!?]+|.', text)
        vertical_text = "\n".join(tokens)
        logger.debug(
            f"[font-fit-raw-vertical] text='{text[:20]}...' target=({target_width:.0f}x{target_height:.0f}) "
            f"fixed={fixed_size}"
        )
        return vertical_text, fixed_size

    max_allowed_size = _get_model_font_upper_bound(initial_font_size)
    min_allowed_size = max(
        config.MIN_FONT_SIZE,
        min(max_allowed_size, math.ceil(initial_font_size * config.MODEL_FONT_SIZE_FLOOR_RATIO)),
    )
    text = text.replace("â‹¯", "ï¸™")
    tokens = re.findall(r'[!?]+|.', text)
    vertical_text = "\n".join(tokens)

    best_candidate = None

    for current_size in range(max_allowed_size, config.MIN_FONT_SIZE - 1, -1):
        text_w, text_h = measure_text(vertical_text, font_path, current_size, style)
        if text_h > target_height or text_w > target_width:
            continue

        score = current_size * 4.0
        fill_ratio = (text_w * text_h) / max(target_width * target_height, 1.0)
        if fill_ratio < config.FONT_AREA_FILL_RATIO:
            score -= (config.FONT_AREA_FILL_RATIO - fill_ratio) * 18.0
        if current_size < min_allowed_size:
            score -= (min_allowed_size - current_size) * 12.0

        if char_ratio_target is not None:
            reference_height = (
                max(float(char_ratio_reference_height), 1.0)
                if char_ratio_reference_height is not None
                else target_height
            )
            measured_char_ratio = measure_character_body_height_ratio(
                vertical_text,
                font_path,
                current_size,
                style,
                reference_height=reference_height,
            )
            char_rel_error = abs(measured_char_ratio - char_ratio_target) / max(char_ratio_target, 1e-4)
            score -= char_rel_error * float(getattr(config, "FONT_CHAR_SCORE_WEIGHT", 42.0))

        if best_candidate is None or score > best_candidate[0]:
            best_candidate = (score, current_size)

    if best_candidate is not None:
        return vertical_text, best_candidate[1]

    return vertical_text, max(config.MIN_FONT_SIZE, min(initial_font_size, max_allowed_size))
