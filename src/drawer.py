import logging
import math
import re
from dataclasses import replace

import numpy as np
from PIL import Image

from src import config
from src.data_models import PageData
from src.text_renderer import (
    DEFAULT_STYLES, FREEFORM_STYLE,
    measure_line, measure_text, render_text_on_image, render_rotated_text_on_image,
    replace_unsupported_chars,
)
from src.utils import rects_intersect

logger = logging.getLogger(__name__)

LINE_HEAD_FORBIDDEN = set(")]}〉》」』】、。，！？…⋯:;")
LINE_TAIL_FORBIDDEN = set("([<{〈《「『【")


def _text_density(text: str) -> int:
    """줄바꿈/공백을 제외한 문자 수를 반환합니다."""
    return len(re.sub(r"\s+", "", text or ""))


def _visible_len(text: str) -> int:
    """공백을 제외한 가시 문자 수를 반환합니다."""
    return len(re.sub(r"\s+", "", text or ""))


def _wrap_text_korean(text, font_path, font_size, style, max_width):
    """한국어 말풍선에 맞춘 줄 바꿈 함수입니다."""
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
    """한 줄의 폭 활용도와 금지 행두/행말을 기준으로 패널티를 계산합니다."""
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
    """넓은 말풍선에서 의도적인 개행으로 더 큰 폰트를 쓰기 위한 balanced wrap."""
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
    """고정 줄 수 balanced wrap 함수를 생성합니다."""
    def _wrap(text, font_path, font_size, style, max_width):
        return _wrap_text_balanced(text, font_path, font_size, style, max_width, target_lines)

    _wrap.__name__ = f"balanced_{target_lines}"
    return _wrap


def _resolve_bubble_style(element, bubble_box, target_width, target_height):
    """커뮤니티 식질 규칙을 반영해 말풍선용 스타일을 보정합니다."""
    base_style = DEFAULT_STYLES.get(element.font_style, DEFAULT_STYLES["standard"])
    bubble_width = max(1, bubble_box[2] - bubble_box[0])
    bubble_height = max(1, bubble_box[3] - bubble_box[1])
    bubble_ratio = bubble_height / bubble_width
    density = _text_density(element.translated_text)

    style = replace(base_style, oversample_scale=max(base_style.oversample_scale, config.DEFAULT_TEXT_OVERSAMPLE))
    target_scale = style.horizontal_scale
    target_letter_spacing = style.letter_spacing
    target_line_spacing = style.line_spacing
    embolden = style.embolden
    oversample = style.oversample_scale

    # 세로로 긴 말풍선은 한국어를 세로쓰기하지 않고, 장평과 줄 밀도로 세로형 인상을 만듭니다.
    if bubble_ratio >= config.TALL_BUBBLE_RATIO and density >= config.TALL_BUBBLE_MIN_CHARS:
        target_scale = min(target_scale, 0.90)
        target_letter_spacing = min(target_letter_spacing, -0.8)
        target_line_spacing = max(target_line_spacing, 1.22)
        oversample = max(oversample, config.DEFAULT_TEXT_OVERSAMPLE)

    # 작은 말풍선은 과도한 압축보다 판독성을 우선합니다.
    if min(target_width, target_height) <= 70 or element.font_size <= config.MIN_READABLE_TEXT_SIZE:
        target_scale = max(target_scale, 0.92)
        target_letter_spacing = max(target_letter_spacing, -0.25)
        target_line_spacing = max(target_line_spacing, 1.16)
        embolden = True
        oversample = max(oversample, config.SMALL_TEXT_OVERSAMPLE)

    return replace(
        style,
        horizontal_scale=target_scale,
        letter_spacing=target_letter_spacing,
        line_spacing=target_line_spacing,
        embolden=embolden,
        oversample_scale=oversample,
    )


def _resolve_freeform_style(element, box_width, box_height):
    """말풍선 밖 텍스트는 stroke를 유지하되, 작은 글씨일수록 더 선명하게 그립니다."""
    style = FREEFORM_STYLE
    if min(box_width, box_height) <= 60 or element.font_size <= config.MIN_READABLE_TEXT_SIZE:
        return replace(
            style,
            horizontal_scale=max(style.horizontal_scale, 0.95),
            letter_spacing=max(style.letter_spacing, -0.1),
            embolden=True,
            oversample_scale=max(style.oversample_scale, config.SMALL_TEXT_OVERSAMPLE),
        )
    return style


def _wrap_text(text, font_path, font_size, style, max_width):
    """단어/글자 단위로 줄 바꿈을 수행하는 함수입니다."""
    lines = []
    for paragraph in text.split('\n'):
        words = re.findall(r'(·+|[!?]+|\S+)', paragraph)
        if not words:
            continue
        current_line = words[0]
        for word in words[1:]:
            joiner = "" if re.match(r'^(·+|[!?⋯]+)$', word) else " "
            if measure_line(current_line + joiner + word, font_path, font_size, style) <= max_width:
                current_line += joiner + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return "\n".join(lines)


def _is_vertical(element, box_width, box_height):
    """텍스트가 세로 쓰기인지 판단합니다."""
    return (config.ENABLE_VERTICAL_TEXT
            and ' ' not in element.translated_text
            and box_width > 0
            and (box_height / box_width >= config.VERTICAL_TEXT_THRESHOLD))


def _aggressive_wrap(text, font_path, font_size, style, max_width):
    """모든 공백에서 줄바꿈하되, 여전히 넘치는 줄은 가장 가까운 공백에서 한번 더 쪼갭니다.
    단어 자체는 절대 쪼개지 않습니다."""
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
    """주어진 wrap 함수로 텍스트를 감싸면서 맞는 폰트 크기를 찾습니다."""
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


def _score_wrapped_candidate(wrapped_text, font_path, font_size, style, target_width, target_height, bubble_ratio):
    """커뮤니티 식질 규칙을 기반으로 래핑 후보를 점수화합니다."""
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
        if bubble_ratio >= config.TALL_BUBBLE_RATIO:
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

    return score


def _get_font_search_start(initial_font_size, target_height):
    """OCR 추정치가 낮더라도 말풍선 높이 기준으로 충분한 탐색을 시작합니다."""
    return max(
        config.MIN_FONT_SIZE,
        min(
            config.MAX_FONT_SIZE,
            max(int(round(target_height)), initial_font_size, config.MIN_READABLE_TEXT_SIZE + 2),
        ),
    )


def _get_model_font_upper_bound(initial_font_size):
    """모델 추정 폰트 크기 대비 허용 가능한 최대 확대 상한을 반환합니다."""
    predicted_size = max(1, int(round(initial_font_size)))
    growth_ratio = max(1.0, float(config.MODEL_FONT_SIZE_CEILING_RATIO))
    capped_size = int(math.floor(predicted_size * growth_ratio))
    return max(
        config.MIN_FONT_SIZE,
        min(config.MAX_FONT_SIZE, max(predicted_size, capped_size)),
    )


def _build_wrap_candidates(text, bubble_ratio):
    """말풍선 형태에 맞춰 일반/강제/balanced wrap 후보를 구성합니다."""
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
):
    """주어진 하한선에서 래핑 후보를 평가하고, fit 후보와 완화 후보를 분리합니다."""
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
                wrapped_text, font_path, found_size, style, target_width, target_height, bubble_ratio
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


def _find_best_fit_font(text, initial_font_size, target_width, target_height, font_path, style):
    """영역에 맞는 최적의 폰트 크기를 찾고 텍스트를 정렬합니다.

    전략:
    1) 여러 래핑 전략 후보를 생성
    2) 각 후보의 최대 적합 폰트 크기를 찾음
    3) 한국어 줄 배치와 말풍선 실루엣을 점수화해 최적 후보를 선택
    """
    max_allowed_size = _get_model_font_upper_bound(initial_font_size)
    start_size = min(_get_font_search_start(initial_font_size, target_height), max_allowed_size)
    bubble_ratio = target_height / max(target_width, 1)
    width_ratios = [1.0, 0.9]
    if bubble_ratio <= 0.9:
        width_ratios.extend([0.82, 0.72])
    elif bubble_ratio >= config.TALL_BUBBLE_RATIO:
        width_ratios.extend([0.78, 0.66, 0.56])
    elif bubble_ratio >= 1.2:
        width_ratios.extend([0.84, 0.74])

    width_ratios = list(dict.fromkeys(width_ratios))
    wrap_fns = _build_wrap_candidates(text, bubble_ratio)
    preferred_minimum_size = max(
        config.MIN_FONT_SIZE,
        min(config.MAX_FONT_SIZE, math.ceil(initial_font_size * config.MODEL_FONT_SIZE_FLOOR_RATIO)),
    )

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
        )
        if best_candidate is None:
            best_candidate = fallback_candidate or relaxed_candidate

    if best_candidate is None:
        return text, max(config.MIN_FONT_SIZE, min(initial_font_size, config.MAX_FONT_SIZE))

    score, wrapped_text, found_size, wrap_name, width_ratio, fits = best_candidate
    logger.debug(
        f"[font-fit] text='{text[:20]}...' target=({target_width:.0f}x{target_height:.0f}) "
        f"initial={start_size} → best={found_size}, wrap={wrap_name}, width_ratio={width_ratio:.2f}, "
        f"score={score:.2f}, fits={fits}, floor={preferred_minimum_size}, cap={max_allowed_size}"
    )
    return wrapped_text, found_size


def _find_best_fit_font_vertical(text, initial_font_size, target_width, target_height, font_path, style):
    """세로 쓰기 텍스트에 맞는 최적의 폰트 크기를 찾습니다."""
    max_allowed_size = _get_model_font_upper_bound(initial_font_size)
    font_size = min(initial_font_size, max_allowed_size)
    text = text.replace("⋯", "︙")
    tokens = re.findall(r'[!?]+|.', text)
    vertical_text = "\n".join(tokens)

    current_size = font_size
    while current_size >= config.MIN_FONT_SIZE:
        text_w, text_h = measure_text(vertical_text, font_path, current_size, style)
        if text_h <= target_height and text_w <= target_width:
            break
        current_size -= 1

    # 영역 대비 텍스트가 작으면 업스케일
    text_w, text_h = measure_text(vertical_text, font_path, current_size, style)
    if target_height > 0 and (text_h / target_height) < config.FONT_AREA_FILL_RATIO:
        last_good_size = current_size
        while current_size < max_allowed_size:
            current_size += 1
            temp_w, temp_h = measure_text(vertical_text, font_path, current_size, style)
            if temp_h > target_height or temp_w > target_width:
                break
            last_good_size = current_size
        current_size = last_good_size

    return vertical_text, current_size


def _fit_text(element, target_width, target_height, font_path, style):
    """수직/수평을 판단하여 최적 폰트 크기와 텍스트를 반환합니다."""
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    vertical = _is_vertical(element, box_width, box_height)

    if vertical:
        wrapped_text, font_size = _find_best_fit_font_vertical(
            element.translated_text, element.font_size,
            box_width, box_height * (1 + config.VERTICAL_TOLERANCE_RATIO), font_path, style
        )
    else:
        wrapped_text, font_size = _find_best_fit_font(
            element.translated_text, element.font_size,
            target_width, target_height, font_path, style
        )

    return wrapped_text, font_size, vertical


def _get_alignment_for_bubble(attachment, text_box, bubble_box):
    """말풍선 내 텍스트 위치를 정렬합니다."""
    text_x1, text_y1, text_x2, text_y2 = text_box
    bubble_x1, _, bubble_x2, _ = bubble_box
    center_y = (text_y1 + text_y2) // 2

    if attachment == 'left':
        align, anchor = 'left', 'lm'
        center_x = max(text_x1 - config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x1 + config.BUBBLE_EDGE_SAFE_MARGIN)
    elif attachment == 'right':
        align, anchor = 'right', 'rm'
        center_x = min(text_x2 + config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x2 - config.BUBBLE_EDGE_SAFE_MARGIN)
    else:  # 'none'
        align, anchor = 'center', 'mm'
        center_x = (text_x1 + text_x2) // 2
    return align, anchor, center_x, center_y


def _adjust_freeform_position(freeform_bbox, center_x, center_y, bubble_text_rects, img_size=None):
    """말풍선 밖 텍스트가 다른 텍스트와 겹치지 않도록 위치를 조정합니다."""
    adj_x, adj_y = center_x, center_y
    w, h = freeform_bbox[2] - freeform_bbox[0], freeform_bbox[3] - freeform_bbox[1]

    for _ in range(3):  # 최대 3회 재검사
        moved = False
        for bubble_text_rect in bubble_text_rects:
            current_bbox = (adj_x - w / 2, adj_y - h / 2, adj_x + w / 2, adj_y + h / 2)
            if rects_intersect(current_bbox, bubble_text_rect):
                moves = {
                    'up': current_bbox[3] - bubble_text_rect[1],
                    'down': bubble_text_rect[3] - current_bbox[1],
                    'left': current_bbox[2] - bubble_text_rect[0],
                    'right': bubble_text_rect[2] - current_bbox[0]
                }
                min_move_dir = min(moves, key=moves.get)
                if min_move_dir == 'up': adj_y -= moves['up']
                elif min_move_dir == 'down': adj_y += moves['down']
                elif min_move_dir == 'left': adj_x -= moves['left']
                elif min_move_dir == 'right': adj_x += moves['right']
                moved = True
        if not moved:
            break

    # 이미지 경계 클램핑
    if img_size:
        img_w, img_h = img_size
        adj_x = max(w / 2, min(adj_x, img_w - w / 2))
        adj_y = max(h / 2, min(adj_y, img_h - h / 2))

    return adj_x, adj_y


def _render_text(img_pil, wrapped_text, font_path, font_size, center_x, center_y, angle, style,
                 align='center', anchor='mm'):
    """텍스트를 렌더링하고 바운딩 박스를 반환합니다. 회전이 필요하면 회전 처리합니다."""
    if abs(angle) > config.MIN_ROTATION_ANGLE:
        return render_rotated_text_on_image(
            img_pil, wrapped_text, center_x, center_y, angle,
            font_path, font_size, style, align
        )
    else:
        return render_text_on_image(
            img_pil, wrapped_text, center_x, center_y,
            font_path, font_size, style, align, anchor
        )


def _draw_speech_bubble_texts(img_pil, page_data):
    """말풍선 텍스트를 그립니다."""
    bubble_text_rects = []
    for bubble in page_data.speech_bubbles:
        element = bubble.text_element
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        element.translated_text = replace_unsupported_chars(element.translated_text, font_path)

        bubble_width = bubble.bubble_box[2] - bubble.bubble_box[0]
        bubble_height = bubble.bubble_box[3] - bubble.bubble_box[1]
        text_w = element.text_box[2] - element.text_box[0]
        text_h = element.text_box[3] - element.text_box[1]
        target_width = bubble_width * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
        target_height = bubble_height * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
        logger.debug(f"[bubble] '{element.translated_text[:15]}...' bubble=({bubble_width}x{bubble_height}) "
                     f"text_box=({text_w:.0f}x{text_h:.0f}) target=({target_width:.0f}x{target_height:.0f}) "
                     f"pred_font={element.font_size}")

        style = _resolve_bubble_style(element, bubble.bubble_box, target_width, target_height)
        wrapped_text, font_size, vertical = _fit_text(element, target_width, target_height, font_path, style)

        if vertical:
            center_x = (element.text_box[0] + element.text_box[2]) / 2
            center_y = (element.text_box[1] + element.text_box[3]) / 2
            bbox = _render_text(img_pil, wrapped_text, font_path, font_size, center_x, center_y, 0, style)
        else:
            align, anchor, center_x, center_y = _get_alignment_for_bubble(
                bubble.attachment, element.text_box, bubble.bubble_box
            )
            bbox = _render_text(
                img_pil, wrapped_text, font_path, font_size,
                center_x, center_y, element.angle, style, align, anchor
            )

        bubble_text_rects.append(bbox)
    return bubble_text_rects


def _draw_freeform_texts(img_pil, page_data, bubble_text_rects):
    """말풍선 밖 텍스트를 그립니다."""
    for element in page_data.freeform_texts:
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        element.translated_text = replace_unsupported_chars(element.translated_text, font_path)

        box_width = element.text_box[2] - element.text_box[0]
        box_height = element.text_box[3] - element.text_box[1]
        target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2))
        target_height = box_height * (1 + config.VERTICAL_TOLERANCE_RATIO) if box_width <= box_height else box_height
        style = _resolve_freeform_style(element, box_width, box_height)

        wrapped_text, font_size, vertical = _fit_text(element, target_width, target_height, font_path, style)

        center_x = (element.text_box[0] + element.text_box[2]) / 2
        if vertical:
            center_y = (element.text_box[1] + element.text_box[3]) / 2
        else:
            _, text_h = measure_text(wrapped_text, font_path, font_size, style)
            center_y = element.text_box[1] + text_h / 2

        text_w, text_h = measure_text(wrapped_text, font_path, font_size, style)
        initial_bbox = (center_x - text_w / 2, center_y - text_h / 2,
                        center_x + text_w / 2, center_y + text_h / 2)
        img_size = (img_pil.width, img_pil.height)
        adj_x, adj_y = _adjust_freeform_position(initial_bbox, center_x, center_y, bubble_text_rects, img_size)

        _render_text(img_pil, wrapped_text, font_path, font_size, adj_x, adj_y,
                     element.angle if not vertical else 0, style)


def draw_text_on_image(inpainted_image, page_data: PageData):
    """Inpainted된 이미지 위에 PageData의 번역된 텍스트를 그립니다."""
    img_pil = Image.fromarray(inpainted_image)

    bubble_text_rects = _draw_speech_bubble_texts(img_pil, page_data)
    _draw_freeform_texts(img_pil, page_data, bubble_text_rects)

    return np.array(img_pil)
