"""Layout planning for translated text.

This module is the thin assembler that decides:
- which style preset to use (resolve_bubble_style / resolve_freeform_style)
- whether to use vertical or horizontal layout (_is_vertical + _should_switch_to_vertical)
- which anchor/align to use based on attachment
- final TextRenderPlan for the renderer

Heavy lifting (wrapping, font-size fitting) lives in text_wrapping.py and
text_fitting.py.
"""
import logging
from dataclasses import dataclass, replace
from typing import Optional

from src import config
from src.data_models import Attachment
from src.text_fitting import (
    find_best_fit_font,
    find_best_fit_font_vertical,
    get_char_ratio_target,
)
from src.text_renderer import DEFAULT_STYLES, FREEFORM_STYLE, TextStyle, measure_text, measure_vertical_text
from src.text_wrapping import TALL_BUBBLE_MIN_CHARS, TALL_BUBBLE_RATIO, text_density, visible_len
from src.utils import rects_intersect

logger = logging.getLogger(__name__)

# Internal layout heuristics (not user-tunable).
_VERTICAL_TOLERANCE_RATIO = 0.05
_DEFAULT_TEXT_OVERSAMPLE = 2
_SMALL_TEXT_OVERSAMPLE = 3


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
    # 세로쓰기 단 나눔 기준 높이 — 피팅과 렌더링이 같은 값을 써야 배치가 일치한다.
    # (세로쓰기는 모든 텍스트에서 단일 단만 사용)
    vertical_column_height: Optional[float] = None
    initial_bbox: Optional[tuple] = None


def resolve_bubble_style(element, bubble_box, target_width, target_height):
    base_style = DEFAULT_STYLES.get(element.font_style, DEFAULT_STYLES["standard"])
    bubble_width = max(1, bubble_box[2] - bubble_box[0])
    bubble_height = max(1, bubble_box[3] - bubble_box[1])
    bubble_ratio = bubble_height / bubble_width
    density = text_density(element.translated_text)

    style = replace(base_style, oversample_scale=max(base_style.oversample_scale, _DEFAULT_TEXT_OVERSAMPLE))
    target_scale = style.horizontal_scale
    target_letter_spacing = style.letter_spacing
    target_line_spacing = style.line_spacing
    embolden = style.embolden
    oversample = style.oversample_scale

    if bubble_ratio >= TALL_BUBBLE_RATIO and density >= TALL_BUBBLE_MIN_CHARS:
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


def _is_vertical(element, box_width, box_height, is_bubble=False):
    if not config.ENABLE_VERTICAL_TEXT or box_width <= 0:
        return False

    aspect = box_height / box_width
    # Extreme aspect: force vertical regardless of predicted font size or whitespace.
    force_threshold = float(getattr(config, "VERTICAL_FORCE_ASPECT_RATIO", 6.0))
    if aspect >= force_threshold:
        return True

    # 프리텍스트는 매우 길쭉한 박스(위 강제 문턱)가 아니면 가로쓰기가 원칙.
    # 좁은 박스는 줄 폭 보장(최소 3자)으로 해결하고, 가로가 심하게
    # 줄어들면 _should_switch_to_vertical이 구제한다.
    if not is_bubble:
        return False

    return (
        box_height > box_width * 1.2
        and element.font_size >= config.MIN_READABLE_TEXT_SIZE
        and ' ' not in element.translated_text
        and aspect >= config.VERTICAL_TEXT_THRESHOLD
    )


def _should_switch_to_vertical(element, fitted_size, is_bubble=False):
    """Fall back to vertical when horizontal fit shrank too far below the model's prediction."""
    if not config.ENABLE_VERTICAL_TEXT:
        return False
    if not element.translated_text:
        return False
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    # 프리텍스트는 길쭉한 박스가 아니면 가로 유지: 크기가 줄어도 종횡비 4
    # 미만이면 세로로 전환하지 않는다 (가로형 박스의 세로 식자 방지).
    if not is_bubble and box_height < box_width * 4:
        return False
    if ' ' in element.translated_text:
        # 띄어쓰기가 있는 문장은 원래 가로 유지가 원칙이지만, 박스가 명백히
        # 세로형이면 (다단 세로쓰기가 가능하므로) 세로 전환을 허용한다.
        if box_height <= box_width * 2:
            return False
    predicted = max(1, int(element.font_size))
    threshold = float(getattr(config, "FONT_SHRINK_THRESHOLD_RATIO", 0.75))
    return fitted_size < predicted * threshold


def _fit_text(element, target_width, target_height, font_path, style, is_bubble=False):
    """Returns (wrapped_text, font_size, vertical, vertical_column_height).

    세로쓰기는 항상 단일 단(1-column)만 사용한다. 말풍선 텍스트는 한 단으로
    끝나지 않으면 세로쓰기로 가지 않고 가로쓰기를 유지한다.
    """
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    vertical = _is_vertical(element, box_width, box_height, is_bubble=is_bubble)
    char_ratio_target = get_char_ratio_target(element)
    char_ratio_reference_height = max(box_height, 1)
    vertical_column_height = box_height * (1 + _VERTICAL_TOLERANCE_RATIO)

    if vertical:
        wrapped_text, font_size, fits = find_best_fit_font_vertical(
            element.translated_text, element.font_size,
            box_width,
            vertical_column_height,
            font_path,
            style,
            char_ratio_target=char_ratio_target,
            char_ratio_reference_height=char_ratio_reference_height,
        )
        if fits or not is_bubble:
            return wrapped_text, font_size, True, vertical_column_height
        logger.debug(
            f"[vertical-reject] bubble text='{(element.translated_text or '')[:20]}...' "
            f"does not fit one column; keeping horizontal"
        )

    wrapped_text, font_size = find_best_fit_font(
        element.translated_text, element.font_size,
        target_width,
        target_height,
        font_path,
        style,
        char_ratio_target=char_ratio_target,
        char_ratio_reference_height=char_ratio_reference_height,
    )

    if _should_switch_to_vertical(element, font_size, is_bubble=is_bubble):
        vertical_wrapped, vertical_size, vertical_fits = find_best_fit_font_vertical(
            element.translated_text, element.font_size,
            box_width,
            vertical_column_height,
            font_path,
            style,
            char_ratio_target=char_ratio_target,
            char_ratio_reference_height=char_ratio_reference_height,
        )
        # 말풍선: 한 단에 들어가고 가로보다 실제로 커질 때만 전환.
        # 말풍선 밖: 기존처럼 항상 전환 (좁은 효과음 박스는 세로가 자연스러움).
        allow_switch = (vertical_fits and vertical_size > font_size) if is_bubble else True
        if allow_switch:
            logger.debug(
                f"[vertical-switch] text='{(element.translated_text or '')[:20]}...' "
                f"horizontal={font_size} < threshold({config.FONT_SHRINK_THRESHOLD_RATIO}) "
                f"× predicted({element.font_size}); vertical={vertical_size} (fits={vertical_fits})"
            )
            return vertical_wrapped, vertical_size, True, vertical_column_height

    return wrapped_text, font_size, False, None


def _horizontal_clearance(page_gray, text_box, max_reach):
    """박스 좌/우로 '막히기 전까지'의 여유 픽셀 수를 잽니다.

    막히는 조건: ① 컷 경계 — 박스 세로 범위의 70% 이상을 덮는
    어두운 세로줄(=만화의 다른 칸 시작) ② 이미지 끝.
    가로쓰기 확장이 옆 칸을 침범하거나 페이지 밖으로 나가는 것을 막는다.
    """
    height, width = page_gray.shape
    x1, y1, x2, y2 = [int(v) for v in text_box]
    y1c, y2c = max(0, y1), min(height, max(y1 + 1, y2))
    band_height = max(1, y2c - y1c)

    def scan(start, step):
        distance = 0
        x = start
        while 0 <= x < width and distance < max_reach:
            column = page_gray[y1c:y2c, x]
            if int((column < 96).sum()) >= band_height * 0.7:
                break  # 컷 테두리/다른 칸
            distance += 1
            x += step
        return distance

    return scan(x1 - 1, -1), scan(x2, 1)


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


def plan_bubble_text(element, alignment, target_width, target_height, font_path, style, bubble_box=None):
    wrapped_text, font_size, vertical, column_height = _fit_text(
        element, target_width, target_height, font_path, style, is_bubble=True
    )

    if vertical:
        # 세로쓰기도 말풍선 중심 기준 (원문 박스는 한쪽으로 치우친 경우가 많음)
        anchor_box = bubble_box if bubble_box is not None else element.text_box
        center_x = (anchor_box[0] + anchor_box[2]) / 2
        center_y = (anchor_box[1] + anchor_box[3]) / 2
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
            vertical_column_height=column_height,
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


def plan_freeform_text(element, bubble_text_rects, img_size, font_path, style, page_gray=None):
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    # 가로쓰기는 탐지 박스가 빠듯한 경우가 많아 일정 비율 넘침을 허용한다
    # (중앙 기준으로 양쪽에 고르게 퍼짐).
    if box_width >= box_height:
        overflow_scale = 1.0 + max(0.0, float(getattr(config, "FREEFORM_BOX_OVERFLOW_RATIO", 0.2)))
    else:
        overflow_scale = 1.0
    target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2)) * overflow_scale
    # 세로형 박스(원본 세로 컬럼 자리)도 가로쓰기가 원칙이므로 줄 폭이
    # 최소 3자는 담도록 보장한다 — 없으면 짧은 어절이 1~2자씩 조각난다.
    # 박스의 2.4배까지만 (옆 그림 침범 가드).
    min_capacity = min(element.font_size * 3.4, box_width * 2.4)
    target_width = max(target_width, min_capacity)

    # 컷 경계/이미지 끝 가드: 가로 확장 전 양옆을 스캔해서
    # - 양쪽 열림 → 중앙 기준 확장 (양쪽 여유의 최솟값까지)
    # - 한쪽만 막힘 → 막힌 쪽으로 정렬(기존 attachment 패턴)하고 열린 쪽으로만 확장
    # - 양쪽 막힘 → 확장 포기 (자기 박스 폭 안에서만)
    attachment = getattr(element, "attachment", Attachment.NONE)
    if page_gray is not None and target_width > box_width:
        margin = 6
        max_reach = int(target_width - box_width) + margin
        left_clear, right_clear = _horizontal_clearance(page_gray, element.text_box, max_reach)
        left_open = left_clear >= margin + 2
        right_open = right_clear >= margin + 2
        if attachment == Attachment.NONE:
            if left_open and right_open:
                width_cap = box_width + 2 * max(0, min(left_clear, right_clear) - margin)
            elif right_open:
                attachment = Attachment.LEFT  # 왼쪽이 막힘 → 왼쪽 정렬, 오른쪽으로 확장
                width_cap = box_width + max(0, right_clear - margin)
            elif left_open:
                attachment = Attachment.RIGHT
                width_cap = box_width + max(0, left_clear - margin)
            else:
                width_cap = box_width
        elif attachment == Attachment.LEFT:
            width_cap = box_width + max(0, right_clear - margin)
        else:  # Attachment.RIGHT
            width_cap = box_width + max(0, left_clear - margin)
        target_width = min(target_width, max(width_cap, box_width * (1.0 - config.FREEFORM_PADDING_RATIO * 2)))

    if box_width <= box_height:
        target_height = box_height * (1 + _VERTICAL_TOLERANCE_RATIO)
    else:
        target_height = box_height * overflow_scale
    wrapped_text, font_size, vertical, column_height = _fit_text(
        element, target_width, target_height, font_path, style, is_bubble=False
    )

    # Vertical text renders as a single-column stack; its actual w/h differs
    # from the horizontal measure result. 피팅과 같은 기준으로 측정해야 한다.
    if vertical:
        text_w, text_h = measure_vertical_text(
            wrapped_text, font_path, font_size, style,
            max_column_height=column_height,
            max_columns=1,
        )
    else:
        text_w, text_h = measure_text(wrapped_text, font_path, font_size, style)

    x1, y1, x2, y2 = element.text_box
    if vertical:
        center_y = (y1 + y2) / 2
    else:
        box_h = y2 - y1
        if text_h > box_h:
            # 번역문이 박스보다 길면 위아래로 균등하게 넘치게 (top-anchor면
            # 아래로만 넘쳐 페이지 하단 안내문 등이 가장자리에 밀착/잘림)
            center_y = (y1 + y2) / 2
        else:
            center_y = y1 + text_h / 2

    # attachment는 위에서 컷 경계 스캔으로 보강(추론)된 값을 그대로 사용한다.
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
        vertical_column_height=column_height,
        initial_bbox=initial_bbox,
    )
