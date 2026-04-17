import logging
import math
import re
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from src import config
from src.data_models import Attachment
from src.text_renderer import (
    DEFAULT_STYLES,
    FREEFORM_STYLE,
    TextStyle,
    measure_character_body_height_ratio,
    measure_line,
    measure_text,
)
from src.utils import rects_intersect

logger = logging.getLogger(__name__)

LINE_HEAD_FORBIDDEN = set(")]}ã€‰ã€‹ã€ã€ã€‘ã€ã€‚ï¼Œï¼ï¼Ÿâ€¦â‹¯:;")
LINE_TAIL_FORBIDDEN = set("([<{ã€ˆã€Šã€Œã€Žã€")

# Internal layout heuristics (not user-tunable; changing these alters wrap strategy and oversample).
_VERTICAL_TOLERANCE_RATIO = 0.05
_DEFAULT_TEXT_OVERSAMPLE = 2
_SMALL_TEXT_OVERSAMPLE = 3
_TALL_BUBBLE_RATIO = 1.8
_TALL_BUBBLE_MIN_CHARS = 8


@dataclass(frozen=True)
class TextRenderPlan:
    text: str
    font_path: str
    font_size: int
    style: TextStyle
    center_x: float
    center_y: float
    angle: float = 0.0
    align: str = "center"
    anchor: str = "mm"
    vertical: bool = False
    initial_bbox: Optional[tuple] = None


def _text_density(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _wrap_text_korean(text, font_path, font_size, style, max_width):
    lines = []
    for paragraph in text.split('\n'):
        words = paragraph.split()
        if not words:
            continue

        current_line = ""
        for word in words:
            candidate = word if not current_line else f"{current_line} {word}"
            if measure_line(candidate, font_path, font_size, style) <= max_width:
                current_line = candidate
                continue

            if current_line:
                lines.append(current_line)
                current_line = ""

            current_line = word

        if current_line:
            lines.append(current_line)

    return "\n".join(lines)


def _line_layout_penalty(line_text, line_width, max_width, target_ratio, is_last_line):
    stripped = line_text.strip()
    if not stripped:
        return 100.0

    width_ratio = line_width / max(max_width, 1)
    desired_ratio = 0.68 if is_last_line else target_ratio
    penalty = 0.0

    if width_ratio < desired_ratio:
        penalty += (desired_ratio - width_ratio) * 18.0
    else:
        penalty += (width_ratio - desired_ratio) * 7.0

    if stripped[0] in LINE_HEAD_FORBIDDEN:
        penalty += 8.0
    if stripped[-1] in LINE_TAIL_FORBIDDEN:
        penalty += 8.0
    if is_last_line and _visible_len(stripped) <= 2:
        penalty += 5.0

    return penalty


def _wrap_text_balanced(text, font_path, font_size, style, max_width, target_lines):
    wrapped_paragraphs = []

    for paragraph in text.split('\n'):
        words = paragraph.split()
        if not words:
            continue

        if len(words) <= 1 or target_lines <= 1:
            wrapped_paragraphs.append(_wrap_text_korean(paragraph, font_path, font_size, style, max_width))
            continue

        tokens = words
        line_count = min(target_lines, len(tokens))
        target_ratio = 0.86 if line_count <= 2 else 0.8
        token_count = len(tokens)
        widths = {}

        for start in range(token_count):
            line_text = ""
            for end in range(start, token_count):
                line_text = tokens[end] if not line_text else f"{line_text} {tokens[end]}"
                widths[(start, end + 1)] = (line_text, measure_line(line_text, font_path, font_size, style))

        inf = float("inf")
        dp = [[inf] * (line_count + 1) for _ in range(token_count + 1)]
        prev = [[None] * (line_count + 1) for _ in range(token_count + 1)]
        dp[0][0] = 0.0

        for start in range(token_count):
            for used_lines in range(line_count):
                if dp[start][used_lines] == inf:
                    continue

                remaining_lines = line_count - used_lines
                max_end = token_count - (remaining_lines - 1)
                for end in range(start + 1, max_end + 1):
                    line_text, line_width = widths[(start, end)]
                    penalty = _line_layout_penalty(
                        line_text,
                        line_width,
                        max_width,
                        target_ratio,
                        used_lines == line_count - 1,
                    )
                    total = dp[start][used_lines] + penalty
                    if total < dp[end][used_lines + 1]:
                        dp[end][used_lines + 1] = total
                        prev[end][used_lines + 1] = start

        if dp[token_count][line_count] == inf:
            wrapped_paragraphs.append(_wrap_text_korean(paragraph, font_path, font_size, style, max_width))
            continue

        lines = []
        end = token_count
        used_lines = line_count
        while used_lines > 0:
            start = prev[end][used_lines]
            if start is None:
                break
            line_text, _ = widths[(start, end)]
            lines.append(line_text)
            end = start
            used_lines -= 1

        lines.reverse()
        wrapped_paragraphs.append("\n".join(lines) if lines else _wrap_text_korean(paragraph, font_path, font_size, style, max_width))

    return "\n".join(wrapped_paragraphs)


def _make_balanced_wrap(target_lines):
    def _wrap(text, font_path, font_size, style, max_width):
        return _wrap_text_balanced(text, font_path, font_size, style, max_width, target_lines)

    _wrap.__name__ = f"balanced_{target_lines}"
    return _wrap


def resolve_bubble_style(element, bubble_box, target_width, target_height):
    base_style = DEFAULT_STYLES.get(element.font_style, DEFAULT_STYLES["standard"])
    bubble_width = max(1, bubble_box[2] - bubble_box[0])
    bubble_height = max(1, bubble_box[3] - bubble_box[1])
    bubble_ratio = bubble_height / bubble_width
    density = _text_density(element.translated_text)

    style = replace(base_style, oversample_scale=max(base_style.oversample_scale, _DEFAULT_TEXT_OVERSAMPLE))
    target_scale = style.horizontal_scale
    target_letter_spacing = style.letter_spacing
    target_line_spacing = style.line_spacing
    embolden = style.embolden
    oversample = style.oversample_scale

    if bubble_ratio >= _TALL_BUBBLE_RATIO and density >= _TALL_BUBBLE_MIN_CHARS:
        target_scale = min(target_scale, 0.90)
        target_letter_spacing = min(target_letter_spacing, -0.8)
        target_line_spacing = max(target_line_spacing, 1.22)
        oversample = max(oversample, _DEFAULT_TEXT_OVERSAMPLE)

    if min(target_width, target_height) <= 70 or element.font_size <= config.MIN_READABLE_TEXT_SIZE:
        target_scale = max(target_scale, 0.92)
        target_letter_spacing = max(target_letter_spacing, -0.25)
        target_line_spacing = max(target_line_spacing, 1.16)
        embolden = True
        oversample = max(oversample, _SMALL_TEXT_OVERSAMPLE)

    return replace(
        style,
        horizontal_scale=target_scale,
        letter_spacing=target_letter_spacing,
        line_spacing=target_line_spacing,
        embolden=embolden,
        oversample_scale=oversample,
    )


def resolve_freeform_style(element, box_width, box_height):
    style = FREEFORM_STYLE
    if min(box_width, box_height) <= 60 or element.font_size <= config.MIN_READABLE_TEXT_SIZE:
        return replace(
            style,
            horizontal_scale=max(style.horizontal_scale, 0.95),
            letter_spacing=max(style.letter_spacing, -0.1),
            embolden=True,
            oversample_scale=max(style.oversample_scale, _SMALL_TEXT_OVERSAMPLE),
        )
    return style


def _adjust_freeform_position(freeform_bbox, anchor_x, anchor_y, bubble_text_rects, img_size=None):
    """Nudge a freeform text to avoid overlap with bubble texts.

    `freeform_bbox` is the initial bounding box produced by the anchor-aware
    layout; the function preserves the anchor→bbox offset while moving the
    anchor, so left/right-aligned texts keep their alignment after adjustment.
    """
    adj_x, adj_y = anchor_x, anchor_y
    dx = freeform_bbox[0] - anchor_x
    dy = freeform_bbox[1] - anchor_y
    w = freeform_bbox[2] - freeform_bbox[0]
    h = freeform_bbox[3] - freeform_bbox[1]

    for _ in range(3):
        moved = False
        for bubble_text_rect in bubble_text_rects:
            current_bbox = (adj_x + dx, adj_y + dy, adj_x + dx + w, adj_y + dy + h)
            if rects_intersect(current_bbox, bubble_text_rect):
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
                moved = True
        if not moved:
            break

    if img_size:
        img_w, img_h = img_size
        adj_x = max(-dx, min(adj_x, img_w - dx - w))
        adj_y = max(-dy, min(adj_y, img_h - dy - h))

    return adj_x, adj_y


def _wrap_text(text, font_path, font_size, style, max_width):
    lines = []
    for paragraph in text.split('\n'):
        words = re.findall(r'(Â·+|[!?]+|\S+)', paragraph)
        if not words:
            continue
        current_line = words[0]
        for word in words[1:]:
            joiner = "" if re.match(r'^(Â·+|[!?â‹¯]+)$', word) else " "
            if measure_line(current_line + joiner + word, font_path, font_size, style) <= max_width:
                current_line += joiner + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return "\n".join(lines)


def _is_vertical(element, box_width, box_height):
    if not config.ENABLE_VERTICAL_TEXT or box_width <= 0:
        return False

    aspect = box_height / box_width
    # Extreme aspect: force vertical regardless of predicted font size or whitespace.
    force_threshold = float(getattr(config, "VERTICAL_FORCE_ASPECT_RATIO", 6.0))
    if aspect >= force_threshold:
        return True

    return (
        box_height > box_width * 1.2
        and element.font_size >= config.MIN_READABLE_TEXT_SIZE
        and ' ' not in element.translated_text
        and aspect >= config.VERTICAL_TEXT_THRESHOLD
    )


def _aggressive_wrap(text, font_path, font_size, style, max_width):
    raw_lines = text.replace('\n', ' ').split(' ')
    lines = []
    current = ""
    for word in raw_lines:
        if not word:
            continue
        if not current:
            current = word
        elif measure_line(current + " " + word, font_path, font_size, style) <= max_width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


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
        if len(lines) == 1 and _visible_len(lines[0]) >= 12:
            score -= 10.0
    else:
        score -= max(0.0, 0.58 - vertical_fill) * 8.0

    if len(lines) > 1:
        edge_avg = (line_widths[0] + line_widths[-1]) / 2
        middle_max = max(line_widths[1:-1], default=max(line_widths))
        if bubble_ratio >= _TALL_BUBBLE_RATIO:
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
        if idx in (0, len(lines) - 1) and _visible_len(stripped) <= 2 and len(lines) >= 3:
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


def _get_font_search_start(initial_font_size, target_height):
    return max(
        config.MIN_FONT_SIZE,
        min(
            config.MAX_FONT_SIZE,
            max(int(round(target_height)), initial_font_size, config.MIN_READABLE_TEXT_SIZE + 2),
        ),
    )


def _is_font_size_correction_enabled():
    return bool(getattr(config, "FONT_SIZE_CORRECTION_ENABLED", True))


def _get_model_font_upper_bound(initial_font_size):
    predicted_size = max(1, int(round(initial_font_size)))
    growth_ratio = max(1.0, float(config.MODEL_FONT_SIZE_CEILING_RATIO))
    capped_size = int(math.floor(predicted_size * growth_ratio))
    return max(
        config.MIN_FONT_SIZE,
        min(config.MAX_FONT_SIZE, max(predicted_size, capped_size)),
    )


def _build_wrap_candidates(text, bubble_ratio):
    candidates = [
        _wrap_text_korean,
        _wrap_text,
        _aggressive_wrap,
    ]

    density = _text_density(text)
    if " " in text and bubble_ratio <= 1.0 and density >= 8:
        line_targets = [2]
        if density >= 14 or bubble_ratio <= 0.75:
            line_targets.append(3)
        if density >= 24 and bubble_ratio <= 0.58:
            line_targets.append(4)
        candidates = [_make_balanced_wrap(target_lines) for target_lines in line_targets] + candidates

    return candidates


def _get_char_ratio_target(element):
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
    width_ratios = [1.0, 0.9]
    if bubble_ratio <= 0.9:
        width_ratios.extend([0.82, 0.72])
    elif bubble_ratio >= _TALL_BUBBLE_RATIO:
        width_ratios.extend([0.78, 0.66, 0.56])
    elif bubble_ratio >= 1.2:
        width_ratios.extend([0.84, 0.74])

    width_ratios = list(dict.fromkeys(width_ratios))
    wrap_fns = _build_wrap_candidates(text, bubble_ratio)
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


def _find_best_fit_font(
    text,
    initial_font_size,
    target_width,
    target_height,
    font_path,
    style,
    char_ratio_target=None,
    char_ratio_reference_height=None,
):
    if not _is_font_size_correction_enabled() and char_ratio_target is None:
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
    width_ratios = [1.0, 0.9]
    if bubble_ratio <= 0.9:
        width_ratios.extend([0.82, 0.72])
    elif bubble_ratio >= _TALL_BUBBLE_RATIO:
        width_ratios.extend([0.78, 0.66, 0.56])
    elif bubble_ratio >= 1.2:
        width_ratios.extend([0.84, 0.74])

    width_ratios = list(dict.fromkeys(width_ratios))
    wrap_fns = _build_wrap_candidates(text, bubble_ratio)
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


def _find_best_fit_font_vertical(
    text,
    initial_font_size,
    target_width,
    target_height,
    font_path,
    style,
    char_ratio_target=None,
    char_ratio_reference_height=None,
):
    if not _is_font_size_correction_enabled() and char_ratio_target is None:
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


def _fit_text(element, target_width, target_height, font_path, style):
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    vertical = _is_vertical(element, box_width, box_height)
    char_ratio_target = _get_char_ratio_target(element)
    char_ratio_reference_height = max(box_height, 1)

    if vertical:
        wrapped_text, font_size = _find_best_fit_font_vertical(
            element.translated_text, element.font_size,
            box_width,
            box_height * (1 + _VERTICAL_TOLERANCE_RATIO),
            font_path,
            style,
            char_ratio_target=char_ratio_target,
            char_ratio_reference_height=char_ratio_reference_height,
        )
        return wrapped_text, font_size, vertical

    wrapped_text, font_size = _find_best_fit_font(
        element.translated_text, element.font_size,
        target_width,
        target_height,
        font_path,
        style,
        char_ratio_target=char_ratio_target,
        char_ratio_reference_height=char_ratio_reference_height,
    )

    if _should_switch_to_vertical(element, font_size):
        logger.debug(
            f"[vertical-switch] text='{(element.translated_text or '')[:20]}...' "
            f"horizontal={font_size} < threshold({config.FONT_SHRINK_THRESHOLD_RATIO}) "
            f"× predicted({element.font_size}); retrying vertical from predicted size"
        )
        vertical_wrapped, vertical_size = _find_best_fit_font_vertical(
            element.translated_text, element.font_size,
            box_width,
            box_height * (1 + _VERTICAL_TOLERANCE_RATIO),
            font_path,
            style,
            char_ratio_target=char_ratio_target,
            char_ratio_reference_height=char_ratio_reference_height,
        )
        return vertical_wrapped, vertical_size, True

    return wrapped_text, font_size, vertical


def _should_switch_to_vertical(element, fitted_size):
    """Fall back to vertical when horizontal fit shrank too far below the model's prediction."""
    if not config.ENABLE_VERTICAL_TEXT:
        return False
    if not element.translated_text or ' ' in element.translated_text:
        return False
    predicted = max(1, int(element.font_size))
    threshold = float(getattr(config, "FONT_SHRINK_THRESHOLD_RATIO", 0.75))
    return fitted_size < predicted * threshold


def plan_bubble_text(element, alignment, target_width, target_height, font_path, style):
    wrapped_text, font_size, vertical = _fit_text(element, target_width, target_height, font_path, style)

    if vertical:
        center_x = (element.text_box[0] + element.text_box[2]) / 2
        center_y = (element.text_box[1] + element.text_box[3]) / 2
        return TextRenderPlan(
            text=wrapped_text,
            font_path=font_path,
            font_size=font_size,
            style=style,
            center_x=center_x,
            center_y=center_y,
            angle=0,
            align="center",
            anchor="mm",
            vertical=True,
        )

    align, anchor, center_x, center_y = alignment
    return TextRenderPlan(
        text=wrapped_text,
        font_path=font_path,
        font_size=font_size,
        style=style,
        center_x=center_x,
        center_y=center_y,
        angle=0,
        align=align,
        anchor=anchor,
        vertical=False,
    )


def plan_freeform_text(element, bubble_text_rects, img_size, font_path, style):
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2))
    target_height = box_height * (1 + _VERTICAL_TOLERANCE_RATIO) if box_width <= box_height else box_height
    wrapped_text, font_size, vertical = _fit_text(element, target_width, target_height, font_path, style)

    text_w, text_h = measure_text(wrapped_text, font_path, font_size, style)
    x1, y1, x2, y2 = element.text_box
    if vertical:
        center_y = (y1 + y2) / 2
    else:
        center_y = y1 + text_h / 2

    attachment = getattr(element, "attachment", Attachment.NONE)
    # Vertical layout keeps its own geometry; horizontal layout honors attachment.
    if vertical or attachment == Attachment.NONE:
        align, anchor = "center", "mm"
        anchor_x = (x1 + x2) / 2
        initial_bbox = (
            anchor_x - text_w / 2,
            center_y - text_h / 2,
            anchor_x + text_w / 2,
            center_y + text_h / 2,
        )
    elif attachment == Attachment.LEFT:
        align, anchor = "left", "lm"
        anchor_x = x1 - config.FREEFORM_ATTACHMENT_TEXT_MARGIN
        initial_bbox = (anchor_x, center_y - text_h / 2, anchor_x + text_w, center_y + text_h / 2)
    else:  # Attachment.RIGHT
        align, anchor = "right", "rm"
        anchor_x = x2 + config.FREEFORM_ATTACHMENT_TEXT_MARGIN
        initial_bbox = (anchor_x - text_w, center_y - text_h / 2, anchor_x, center_y + text_h / 2)

    adj_x, adj_y = _adjust_freeform_position(initial_bbox, anchor_x, center_y, bubble_text_rects, img_size)

    return TextRenderPlan(
        text=wrapped_text,
        font_path=font_path,
        font_size=font_size,
        style=style,
        center_x=adj_x,
        center_y=adj_y,
        angle=element.angle if not vertical else 0,
        align=align,
        anchor=anchor,
        vertical=vertical,
        initial_bbox=initial_bbox,
    )
