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

import numpy as np

from src import config
from src.text_renderer import (
    measure_character_body_height_ratio,
    measure_line,
    measure_text,
    measure_vertical_block,
)
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
    # 시작점 = 원문 측정 크기 (가독성 하한만 보장). 측정값이 곧 원문 글자
    # 크기이므로 그대로 시작해야 번역문이 원문과 같은 크기로 식자된다.
    return max(
        config.MIN_FONT_SIZE,
        min(
            config.MAX_FONT_SIZE,
            max(int(round(initial_font_size)), config.MIN_READABLE_TEXT_SIZE + 2),
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
    """가장 큰 '들어가는' 크기를 이분 탐색으로 찾습니다.

    크기가 작아질수록 줄당 단어가 늘어 줄 수·폭·높이가 같이 줄어들므로
    fits(size)는 단조: 한 번 들어가면 더 작은 크기도 들어간다. 선형 1px
    감소 스캔 대비 측정 횟수를 O(범위) → O(log 범위)로 줄인다.
    """
    minimum_size = max(config.MIN_FONT_SIZE, min_size or config.MIN_FONT_SIZE)
    start_size = max(int(font_size), minimum_size)

    def _attempt(size):
        wrapped = wrap_fn(text, font_path, size, style, target_width)
        text_w, text_h = measure_text(wrapped, font_path, size, style)
        overflow = max(0.0, text_w - target_width) + max(0.0, text_h - target_height)
        return wrapped, overflow

    wrapped_hi, overflow_hi = _attempt(start_size)
    if overflow_hi <= 0.0:
        return wrapped_hi, start_size, True

    wrapped_lo, overflow_lo = _attempt(minimum_size)
    if overflow_lo > 0.0:
        # 최소 크기로도 안 들어감 → 오버플로가 가장 작은 최소 크기를 반환.
        return wrapped_lo, minimum_size, False

    lo, hi = minimum_size, start_size  # 불변식: fits(lo)=True, fits(hi)=False
    best_wrapped = wrapped_lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        wrapped_mid, overflow_mid = _attempt(mid)
        if overflow_mid <= 0.0:
            lo, best_wrapped = mid, wrapped_mid
        else:
            hi = mid
    return best_wrapped, lo, True


def _count_midword_breaks(source_text, wrapped_text):
    """줄 경계 중 원문의 공백(어절 경계)과 일치하지 않는 곳의 수.

    공백을 제거한 누적 글자 수로 비교하므로 줄바꿈 과정에서 공백이
    소비/복원되어도 정확하다.
    """
    word_break_positions = set()
    n = 0
    for ch in source_text:
        if ch in (' ', '\n'):
            word_break_positions.add(n)
        else:
            n += 1
    breaks = 0
    cum = 0
    lines = [line for line in wrapped_text.split('\n') if line.strip()]
    for line in lines[:-1]:
        cum += len(line.replace(' ', ''))
        if cum not in word_break_positions:
            breaks += 1
    return breaks


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
    predicted_size=None,
    source_text=None,
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
    # 크기 보상은 원문 추정 크기에서 클램프: 원문보다 작아지는 건 강하게 막되,
    # 원문 '이상'으로 키우는 건 보상하지 않는다 (식자 정석 = 원문과 같은 크기.
    # 클램프가 없으면 글자 단위 줄바꿈 도입 후 모든 텍스트가 상한 1.2배까지
    # 자라는 전역 인플레이션이 생긴다).
    rewarded_size = min(font_size, predicted_size) if predicted_size else font_size
    score = rewarded_size * 4.0
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

    # 2줄 레이아웃의 외톨이 끝줄(3+1 분할): 끝줄이 첫 줄의 절반에 한참
    # 못 미치면 균등 분할(2+2) 후보가 이기도록 패널티
    if len(lines) == 2:
        first_len, last_len = visible_len(lines[0]), visible_len(lines[1])
        if last_len <= 2 and last_len * 2 < first_len:
            score -= 6.0

    # 단어 중간 줄바꿈은 '마지막 수단': 줄바꿈당 -6점이면 크기 이득 ~2px(+8점)
    # 이상일 때만 채택된다. 패널티가 없으면 채움(fill) 항이 줄 수 많은 쪽을
    # 미세하게 선호해, 크기 이득이 없어도 단어를 쪼개는 부작용이 생긴다.
    if source_text:
        score -= _count_midword_breaks(source_text, wrapped_text) * 6.0

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
    predicted_size=None,
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
                predicted_size=predicted_size,
                source_text=text,
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
                source_text=text,
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
        predicted_size=initial_font_size,
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
            predicted_size=initial_font_size,
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
    max_columns=None,
):
    """Returns (text, font_size, fits).

    세로쓰기는 항상 단일 단(1-column)이 기본이다. fits=False면 어떤 크기로도
    한 단에 들어가지 않았다는 뜻 (말풍선은 이 신호로 가로쓰기로 되돌린다).
    """
    max_columns = 1 if max_columns is None else max(1, int(max_columns))

    if not is_font_size_correction_enabled() and char_ratio_target is None:
        fixed_size = max(1, int(round(initial_font_size)))
        text_w, text_h, n_columns = measure_vertical_block(
            text, font_path, fixed_size, style,
            max_column_height=target_height,
            max_columns=max_columns,
        )
        fits = n_columns > 0 and text_w <= target_width and text_h <= target_height
        logger.debug(
            f"[font-fit-raw-vertical] text='{text[:20]}...' target=({target_width:.0f}x{target_height:.0f}) "
            f"fixed={fixed_size} fits={fits}"
        )
        return text, fixed_size, fits

    max_allowed_size = _get_model_font_upper_bound(initial_font_size)
    min_allowed_size = max(
        config.MIN_FONT_SIZE,
        min(max_allowed_size, math.ceil(initial_font_size * config.MODEL_FONT_SIZE_FLOOR_RATIO)),
    )

    best_candidate = None

    # 가로 피팅과 동일: 시작점 = 원문 측정 크기 (가독성 하한 보장, 상한 캡)
    vertical_start_size = max(
        config.MIN_FONT_SIZE,
        min(max_allowed_size, max(int(round(initial_font_size)), config.MIN_READABLE_TEXT_SIZE + 2)),
    )

    for current_size in range(vertical_start_size, config.MIN_FONT_SIZE - 1, -1):
        text_w, text_h, n_columns = measure_vertical_block(
            text, font_path, current_size, style,
            max_column_height=target_height,
            max_columns=max_columns,
        )
        if n_columns == 0:
            continue
        if text_h > target_height or text_w > target_width:
            continue

        # 가로 피팅과 동일하게 원문 추정 크기에서 보상 클램프 (전역 인플레 방지)
        score = min(current_size, max(1, int(round(initial_font_size)))) * 4.0
        fill_ratio = (text_w * text_h) / max(target_width * target_height, 1.0)
        if fill_ratio < config.FONT_AREA_FILL_RATIO:
            score -= (config.FONT_AREA_FILL_RATIO - fill_ratio) * 18.0
        if current_size < min_allowed_size:
            score -= (min_allowed_size - current_size) * 12.0
        # 같은 값이면 단 수가 적은 쪽(읽기 쉬운 쪽)을 선호
        score -= (n_columns - 1) * 2.0

        if char_ratio_target is not None:
            reference_height = (
                max(float(char_ratio_reference_height), 1.0)
                if char_ratio_reference_height is not None
                else target_height
            )
            measured_char_ratio = measure_character_body_height_ratio(
                text,
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
        return text, best_candidate[1], True

    return text, max(config.MIN_FONT_SIZE, min(initial_font_size, max_allowed_size)), False
