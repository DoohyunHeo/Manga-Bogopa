"""Layout planning for translated text.

This module is the thin assembler that decides:
- which style preset to use (resolve_bubble_style / resolve_freeform_style)
- whether to use vertical or horizontal layout (말풍선: page_drawer가 bubble_vertical_rescue로, 말풍선 밖:
  _is_vertical + _should_switch_to_vertical)
- which anchor/align to use based on attachment
- final TextRenderPlan for the renderer

Heavy lifting (wrapping, font-size fitting) lives in text_wrapping.py and
text_fitting.py.
"""
import functools
import logging
import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from src import config
from src.data_models import Attachment
from src.text_fitting import (
    FitRequest,
    find_best_fit_font,
    fits_target,
    find_best_fit_font_vertical,
    get_char_ratio_target,
    rank_wraps_for_size,
    score_wrap_for_size,
    select_wrap_for_size,
)
from src.text_renderer import (
    DEFAULT_STYLES,
    FREEFORM_STYLE,
    TextStyle,
    compose_text_layer,
    measure_text,
    measure_vertical_text,
)
from src.text_wrapping import TALL_BUBBLE_MIN_CHARS, TALL_BUBBLE_RATIO, count_midword_breaks, text_density, visible_len
from src.utils import rects_intersect

logger = logging.getLogger(__name__)

# Internal layout heuristics (not user-tunable).
_VERTICAL_TOLERANCE_RATIO = 0.05
_FREEFORM_MAX_WIDTH_RATIO = 1.1
_FREEFORM_END_MARGIN = 0.35  # 해설 상자 좌우 끝에 붙지 않게
# 말풍선 밖 세로 한 단 글자(상자 세로 ÷ 가로 이 값 이상)는 가로로 여러 줄이 되면 1단 세로도 계산해, 가로 크기의
# 0.9배 이상으로 들어가면 세로로 — 원문 모양을 따르고 옆 말풍선을 덜 침범한다 (기울어진 세로 캡션 등)
_FREEFORM_COLUMN_ASPECT = 2.5
_FREEFORM_COLUMN_MIN_SIZE_RATIO = 0.9
# 가로로 바꾼 글이 원래 세로 상자 폭의 이 배를 넘으면(그림·옆 말풍선을 덮는다) 크기가 좀 작아져도 1단 세로로 —
# 가로 크기의 이 배 이상, 읽을 수 있는 크기 이상일 때만
_FREEFORM_COLUMN_MAX_SPREAD = 1.2
_FREEFORM_COLUMN_SPREAD_MIN_SIZE_RATIO = 0.6
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
    # (말풍선 세로쓰기는 한 단 — 여러 단은 말풍선 밖 긴 세로 캡션만 vertical_columns로 쓴다)
    vertical_column_height: Optional[float] = None
    initial_bbox: Optional[tuple] = None
    # 세로쓰기 단 수 — 1이 기본(세로는 한 단). 말풍선 밖 긴 세로 캡션을 살릴 때만 원문처럼 여러 단 (page_drawer)
    vertical_columns: int = 1


@dataclass(frozen=True)
class FreeformGeometry:
    """프리텍스트의 목표 영역과 (컷 경계 스캔으로 보강된) 붙음 방향.

    plan_freeform_text가 계산하던 값을 떼어낸 것 — 식자 검수가 같은 기하로
    줄바꿈/크기 대안을 다시 배치할 수 있게 한다.
    """
    target_width: float
    target_height: float
    attachment: Attachment
    wall_gap: Optional[tuple] = None  # 원문 상자 왼쪽·오른쪽 끝에서 칸 테두리까지 빈 픽셀 (잴 그림이 없으면 None)


# 같은 가족 굵은 파일이 보통 파일보다 획이 이 배만큼 굵지 않으면 굵은 파일에 합성 굵게를 더한다 — 강조가 보통 대사와
# 구별은 되되 가볍게. Pretendard Regular→SemiBold(1.38배)는 그대로, 한 단계만 굵은 Medium 같은 짝(1.2배 아래)에만
# 더한다
_BOLD_STROKE_MIN_RATIO = 1.25
_STROKE_SAMPLE = "한글식자기준본문"


@functools.lru_cache(maxsize=16)
def _stroke_px(font_path):
    """그 글꼴로 기준 문자열을 48px에 그렸을 때의 평균 획 폭 (잉크 넓이 × 2 ÷ 가장자리 픽셀 수)."""
    style = replace(DEFAULT_STYLES["standard"], embolden=False, stroke_width=0)
    layer, _, _, _ = compose_text_layer(_STROKE_SAMPLE, 300, 60, font_path, 48, style)
    ink = np.asarray(layer)[..., 3] > 127
    inner = ink.copy()
    inner[1:, :] &= ink[:-1, :]
    inner[:-1, :] &= ink[1:, :]
    inner[:, 1:] &= ink[:, :-1]
    inner[:, :-1] &= ink[:, 1:]
    edge = int(ink.sum() - inner.sum())
    return 2.0 * float(ink.sum()) / edge if edge else 0.0


def needs_synthetic_bold() -> bool:
    """굵은 대사(shouting)에 합성 굵게를 더할지 — 보통 글꼴과 같은 가족의 굵은 파일이 없거나, 있어도 보통 파일보다 획이
    _BOLD_STROKE_MIN_RATIO배 굵지 않을 때. 강조가 보통 대사와 구별되어 보여야 한다."""
    bold = config.BOLD_FONT
    if bold is None or bold.synthetic:
        return True
    try:
        base = _stroke_px(config.FONT_MAP.get("standard", config.DEFAULT_FONT_PATH))
        return base > 0 and _stroke_px(bold.path) < _BOLD_STROKE_MIN_RATIO * base
    except Exception:  # noqa: BLE001 — 글꼴을 못 읽으면 굵은 파일만으로 그린다
        return False


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
    embolden = style.embolden if element.font_style != "shouting" else needs_synthetic_bold()
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
        oversample = max(oversample, _SMALL_TEXT_OVERSAMPLE)
        # 합성 굵게는 글자가 작아 획이 뭉개질 때만 — 좁은 말풍선이라는 것만으로는 굵게 그리지 않는다 (작은 곁말도 보통 굵기)
        embolden = embolden or element.font_size <= config.MIN_READABLE_TEXT_SIZE

    return replace(
        style,
        horizontal_scale=target_scale,
        letter_spacing=target_letter_spacing,
        line_spacing=target_line_spacing,
        embolden=embolden,
        oversample_scale=oversample,
    )


_LOOK_MAX_OUTLINE_PX = 8        # 원문 외곽선을 따라 그릴 최대 두께 (글자 바깥으로 보이는 폭)
_LOOK_MAX_OUTLINE_RATIO = 0.12  # 글자 크기 대비 최대 두께 — 두꺼우면 말줄임표가 뭉치고 스티커처럼 보인다


def apply_text_look(style, look, freeform, font_size=None):
    """원문 글자 모양 실측(look)을 스타일에 반영한다. 반영할 게 없으면 None — 부르는 쪽이 기본 규칙을 쓴다.

    흰 글자면 글자색과 테두리색을 맞바꾸고, 원문에 외곽선이 있으면 그 두께로 두른다(skia 테두리는 윤곽선 양쪽에
    그려지므로 두 배). 검은 글자에 외곽선이 없는 말풍선 글자는 그대로 둔다. 극성을 '글자 영역에 많은 색'으로만
    짐작한 경우(basis="ink")는 실제 쪽에서 틀린 적이 있어 쓰지 않는다.
    """
    if not look or not look.get("polarity"):
        return None
    if look.get("basis") not in ("plain", "outline"):
        return None
    dark = look["polarity"] == "dark"
    outline = bool(look.get("outline"))
    if not freeform and dark and not outline:
        return None
    if freeform:
        ink = tuple(int(v) for v in config.FREEFORM_FONT_COLOR)
        rim = tuple(int(v) for v in config.FREEFORM_STROKE_COLOR)
    else:
        ink, rim = style.color, (255, 255, 255)
    if not dark:
        ink, rim = rim, ink
    width = style.stroke_width
    if outline:
        visible = min(float(look.get("outline_px") or 0.0), _LOOK_MAX_OUTLINE_PX)
        if font_size:
            visible = min(visible, _LOOK_MAX_OUTLINE_RATIO * float(font_size))
        width = max(style.stroke_width, 2.0 * visible)
    elif not freeform:
        width = 0.0  # 검은 말풍선 속 흰 글자 — 원문처럼 테두리 없이
    return replace(style, color=ink, stroke_color=rim, stroke_width=width)


def resolve_freeform_style(element, box_width, box_height):
    # 글자색·테두리색·테두리 두께는 코드 고정값(FREEFORM_FONT_COLOR·FREEFORM_STROKE_COLOR·FREEFORM_STROKE_WIDTH)
    # 합성 굵게는 말풍선과 같게 굵은 대사(굵은 파일이 모자랄 때)와 획이 뭉개지는 작은 글자에만 — 해설·곁말은 보통 굵기로
    embolden = ((element.font_style == "shouting" and needs_synthetic_bold())
                or element.font_size <= config.MIN_READABLE_TEXT_SIZE)
    style = replace(
        FREEFORM_STYLE,
        color=tuple(int(v) for v in config.FREEFORM_FONT_COLOR),
        stroke_color=tuple(int(v) for v in config.FREEFORM_STROKE_COLOR),
        stroke_width=float(max(0, int(config.FREEFORM_STROKE_WIDTH))),
        embolden=embolden,
    )
    if min(box_width, box_height) <= 60 or element.font_size <= config.MIN_READABLE_TEXT_SIZE:
        return replace(
            style,
            horizontal_scale=max(style.horizontal_scale, 0.95),
            letter_spacing=max(style.letter_spacing, -0.1),
            oversample_scale=max(style.oversample_scale, _SMALL_TEXT_OVERSAMPLE),
        )
    return style


def _is_vertical(box_width, box_height):
    """말풍선 밖 글자를 처음부터 세로로 쓸지."""
    if not config.ENABLE_VERTICAL_TEXT or box_width <= 0:
        return False

    aspect = box_height / box_width
    # Extreme aspect: force vertical regardless of the measured source size or whitespace.
    # 프리텍스트는 매우 길쭉한 박스(이 강제 문턱)가 아니면 가로쓰기가 원칙.
    # 좁은 박스는 줄 폭 보장(최소 3자)으로 해결하고, 가로가 심하게
    # 줄어들면 _should_switch_to_vertical이 구제한다.
    force_threshold = float(config.VERTICAL_FORCE_ASPECT_RATIO)
    return aspect >= force_threshold


def _should_switch_to_vertical(element, fitted_size):
    """Fall back to vertical when horizontal fit shrank too far below the measured source size."""
    if not config.ENABLE_VERTICAL_TEXT:
        return False
    if not element.translated_text:
        return False
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    # 프리텍스트는 길쭉한 박스가 아니면 가로 유지: 크기가 줄어도 종횡비 4
    # 미만이면 세로로 전환하지 않는다 (가로형 박스의 세로 식자 방지).
    if box_height < box_width * 4:
        return False
    if ' ' in element.translated_text:
        # 띄어쓰기가 있는 문장은 원래 가로 유지가 원칙이지만, 박스가 명백히
        # 세로형이면 (다단 세로쓰기가 가능하므로) 세로 전환을 허용한다.
        if box_height <= box_width * 2:
            return False
    predicted = max(1, int(element.font_size))
    threshold = float(config.FONT_SHRINK_THRESHOLD_RATIO)
    return fitted_size < predicted * threshold


def _horizontal_fit_request(element, target_width, target_height, font_path, style, in_bubble=True):
    box_height = element.text_box[3] - element.text_box[1]
    return FitRequest(
        text=element.translated_text,
        font_path=font_path,
        style=style,
        target_width=target_width,
        target_height=target_height,
        initial_font_size=element.font_size,
        char_ratio_target=get_char_ratio_target(element),
        char_ratio_reference_height=max(box_height, 1),
        in_bubble=in_bubble,
        room=getattr(element, "room", None) if in_bubble else None,  # 말풍선 모양 (page_drawer가 붙인다)
    )


def rewrap_horizontal_at_size(element, target_width, target_height, font_path, style, font_size, in_bubble=True):
    """고정 크기에서 가장 좋은 가로 줄바꿈을 골라 (wrapped_text, fits)를 돌려준다.

    글자·공백은 그대로, 줄바꿈 위치만 달라진다. fits는 측정 블록이 목표 영역
    안에 들어가는지 여부 (페이지 통일이 키운 크기를 채택할지 판단하는 데 쓴다).
    """
    req = _horizontal_fit_request(element, target_width, target_height, font_path, style, in_bubble)
    wrapped = select_wrap_for_size(req, font_size)
    return wrapped, fits_target(req, wrapped, font_size)


def horizontal_wrap_alternatives(element, target_width, target_height, font_path, style, font_size, limit,
                                 in_bubble=True):
    """같은 크기에서 시도할 수 있는 서로 다른 줄바꿈 텍스트를 점수순으로 최대 limit개."""
    if limit <= 0:
        return []
    req = _horizontal_fit_request(element, target_width, target_height, font_path, style, in_bubble)
    return [text for _, text in rank_wraps_for_size(req, font_size)][:limit]


# 외톨이 끝줄 대안은 피팅 점수가 기준(현재 줄바꿈)보다 이 이상 낮으면 채택하지 않는다.
# (스코어러의 외톨이 벌점 6점보다 작게 잡아, 채움·균형을 크게 해치는 대안은 걸러진다)
_ORPHAN_SCORE_TOLERANCE = 4.0
# 끝줄이 이 글자 수 이하일 때만 외톨이로 본다. '가자'·'먹어' 같은 두 글자 완결 단어는
# 한 줄로 서도 자연스러워서, 한 글자만 남은 끝줄로 한정한다.
_ORPHAN_MAX_CHARS = 1


def has_orphan_last_line(wrapped_text):
    """끝줄이 한 글자만 홀로 남은 줄바꿈인지 (2줄 이상일 때)."""
    lines = [line for line in (wrapped_text or "").split("\n") if line.strip()]
    return len(lines) >= 2 and visible_len(lines[-1]) <= _ORPHAN_MAX_CHARS


def orphan_free_wrap_alternatives(element, target_width, target_height, font_path, style,
                                  font_size, current_text, limit, in_bubble=True):
    """외톨이 끝줄을 없애는 같은 크기 줄바꿈 대안 (점수순, 목표 영역 안에 들어가는 것만).

    원문 번역에 의도된 줄바꿈('\\n')이 있으면 손대지 않는다. 현재 줄바꿈이
    외톨이 끝줄이 아니면 빈 목록.
    """
    if limit <= 0 or not has_orphan_last_line(current_text):
        return []
    if "\n" in (element.translated_text or ""):
        return []
    req = _horizontal_fit_request(element, target_width, target_height, font_path, style, in_bubble)
    base_score = score_wrap_for_size(req, current_text, font_size)
    out = []
    for score, text in rank_wraps_for_size(req, font_size):
        if text == current_text or has_orphan_last_line(text):
            continue
        if score < base_score - _ORPHAN_SCORE_TOLERANCE:
            break  # 점수 내림차순이므로 이후는 모두 기준 미달
        if fits_target(req, text, font_size):
            out.append(text)
        if len(out) >= limit:
            break
    return out


# 화면 속 채팅·슈퍼챗 한 줄 — 원문이 가로 한 줄(상자 폭이 높이의 3배 이상, 높이가 글자 크기의 1.8배 미만)이고 흰 글자면
# 한국어도 한 줄로, 왼쪽 정렬로 쓴다 (여러 줄로 흩으면 띠 밖으로 나간다). 줄여서 한 줄에 들어갈 때만(원문 크기의 0.6배 이상)
_CHAT_MIN_ASPECT = 3.0
_CHAT_MAX_LINE_RATIO = 1.8
_CHAT_MIN_SIZE_RATIO = 0.6
_CHAT_END_MARGIN = 0.35


def is_chat_line(element):
    x1, y1, x2, y2 = element.text_box[:4]
    look = getattr(element, "look", None) or {}
    text = element.translated_text or ""
    return (look.get("polarity") == "light" and "\n" not in text and element.font_size
            and (x2 - x1) >= _CHAT_MIN_ASPECT * (y2 - y1) and (y2 - y1) < _CHAT_MAX_LINE_RATIO * element.font_size)


def _single_line_fit(element, target_width, font_path, style):
    """한 줄로 들어가는 가장 큰 크기 (원문 크기·최대 글자 크기 이하), 너무 작아지면 None.

    폭은 원래 상자 폭에서 끝 여백(글자 크기의 _CHAT_END_MARGIN배)을 뺀 만큼까지만 — 띠 끝에 붙지 않게.
    """
    box_width = element.text_box[2] - element.text_box[0]
    size = int(min(element.font_size, config.MAX_FONT_SIZE))

    def room(px):
        return min(target_width, box_width) - _CHAT_END_MARGIN * px

    width, _ = measure_text(element.translated_text, font_path, size, style)
    if width > room(size):
        size = int(size * room(size) / max(width, 1.0))
        while size > 1 and measure_text(element.translated_text, font_path, size, style)[0] > room(size):
            size -= 1
    return size if size >= max(config.MIN_READABLE_TEXT_SIZE, _CHAT_MIN_SIZE_RATIO * element.font_size) else None


# 말풍선 세로쓰기는 구제용으로만 — 공식 한국어판처럼 말풍선은 가로가 기본이고, 세로는 가로로 쓰면 글자를 한참 줄여야
# 할 때뿐이다. page_drawer가 가로와 한 단 세로(말풍선 목표 영역 또는 원문 글자 상자
# 모양)를 각각 실제로 그릴 계획 — 같은 맞춤기·같은 키우기라 같은 상한·하한 — 으로 만들어 그 크기로 견준다: 가로가 세로의
# _BUBBLE_VERTICAL_RESCUE_RATIO배에 못 미치거나, 가로가 읽을 수 있는 크기(MIN_READABLE_TEXT_SIZE)·원문 크기의
# _FLOOR_RESCUE_OF_SOURCE배 아래이면서 세로가 _FLOOR_RESCUE_GAIN배 이상 클 때만 세로로 쓴다. 좁은 말풍선은 원문보다
# 조금 작아도 가로로 쓴다 — 정책 하한(원문 0.8배)에 몇 px 못 미친다고 세로로 넘기지 않는다
_BUBBLE_VERTICAL_RESCUE_RATIO = 0.6
_FLOOR_RESCUE_GAIN = 1.1
_FLOOR_RESCUE_OF_SOURCE = 0.65
# 띄어쓰기 없는 짧은 말(글자 이만큼 이하)이 가로로는 그 크기에서 한 줄에 안 들어가 낱말 중간에서 끊기는 좁은 말풍선은,
# 세로가 크기를 줄이지 않으면 한 글자씩 세로로 쓴다 — 짧은 이름·외침을 낱말 중간에서 끊지 않게. 긴 말은 가로이고,
# 한 줄에 들어가는데 모양 점수로 나눈 말('으/악.')은 한 줄 가로가 맞다
_SHORT_WORD_MAX_LETTERS = 6


def _splits_off_a_letter(word, horizontal_text):
    """줄바꿈이 낱말에서 글자 하나를 따로 떼어 놓았는지('로보/코!!'·'진/짜/로/요!!') — 로마자 낱말은 어디서 끊어도
    낱말이 깨진다('ROBO/CO?!'). 반반으로 나뉜 '배후령/이잖아'·'제법/이네'는 명사 뒤 꼬리 앞 자연스러운 가로 끊김이다."""
    lines = [line for line in (horizontal_text or "").split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    if word.isascii():
        return True
    return min(sum(1 for ch in line if ch.isalnum()) for line in lines) <= 1


def short_word_split(element, horizontal_text):
    """가로 계획이 띄어쓰기 없는 짧은 말을 낱말 중간에서 끊어 글자를 떼어 놓았으면 그 말, 아니면 None."""
    word = (element.translated_text or "").strip()
    if not word or any(ch.isspace() for ch in word) or sum(1 for ch in word if ch.isalnum()) > _SHORT_WORD_MAX_LETTERS:
        return None
    if not count_midword_breaks(word, horizontal_text or "") or not _splits_off_a_letter(word, horizontal_text):
        return None
    return word


def short_word_forced_break(element, horizontal_text, target_width, target_height, font_path, style, font_size):
    """가로 계획이 짧은 말을 끊어 글자를 떼어 놓았고(short_word_split), 그 크기로는 한 줄에 안 들어가 끊을 수밖에
    없었는지."""
    word = short_word_split(element, horizontal_text)
    if word is None:
        return False
    req = _horizontal_fit_request(element, target_width, target_height, font_path, style, True)
    return not fits_target(req, word, font_size)


def bubble_vertical_rescue(element, horizontal_size, vertical_size, short_word_forced=False):
    """말풍선 세로 구제 시험 — 두 크기는 가로·세로로 실제로 그릴 최종 크기, element는 키우기 전 원래 요소,
    short_word_forced는 가로가 짧은 말을 어쩔 수 없이 낱말 중간에서 끊었는지 (short_word_forced_break)."""
    if not config.ENABLE_VERTICAL_TEXT:
        return False
    if short_word_forced and vertical_size >= horizontal_size:
        return True
    if vertical_size <= horizontal_size:
        return False
    floor = max(config.MIN_READABLE_TEXT_SIZE, _FLOOR_RESCUE_OF_SOURCE * float(element.font_size or 0))
    return (horizontal_size < _BUBBLE_VERTICAL_RESCUE_RATIO * vertical_size
            or (horizontal_size < floor and vertical_size >= _FLOOR_RESCUE_GAIN * horizontal_size))


def _bubble_vertical_fit(element, target_width, target_height, font_path, style):
    """한 단 세로 (글, 크기, 단 높이) — 말풍선 목표 영역과 원문 글자 상자 모양 중 크게 들어가는 쪽, 안 들어가면 None."""
    if not (element.translated_text or "").strip():
        return None
    request = replace(_horizontal_fit_request(element, target_width, target_height, font_path, style, True),
                      room=None)  # 말풍선 모양 따라 쓰기는 가로 줄만
    x1, y1, x2, y2 = element.text_box[:4]
    best = None
    for width, height in ((target_width, target_height), (x2 - x1, (y2 - y1) * (1 + _VERTICAL_TOLERANCE_RATIO))):
        text, size, fits = find_best_fit_font_vertical(replace(request, target_width=width, target_height=height))
        if fits and (best is None or size > best[1]):
            best = (text, size, height)
    return best


def _fit_bubble_text(element, target_width, target_height, font_path, style, vertical=False):
    """말풍선: 가로, vertical이면 한 단 세로 (안 들어가면 가로). 어느 쪽을 쓸지는 page_drawer가 정한다."""
    if vertical:
        fit = _bubble_vertical_fit(element, target_width, target_height, font_path, style)
        if fit is not None:
            return fit[0], fit[1], True, fit[2]
    wrapped_text, font_size = find_best_fit_font(
        _horizontal_fit_request(element, target_width, target_height, font_path, style, True))
    return wrapped_text, font_size, False, None


def _fit_text(element, target_width, target_height, font_path, style, is_bubble=False,
              bubble_vertical=False):
    """Returns (wrapped_text, font_size, vertical, vertical_column_height).

    세로쓰기는 항상 단일 단(1-column)만 사용한다. 말풍선은 _fit_bubble_text로 — 가로, bubble_vertical이면
    한 단 세로 (어느 쪽을 쓸지는 page_drawer가 두 최종 계획을 구제 시험으로 견줘 정한다).
    """
    if is_bubble:
        return _fit_bubble_text(element, target_width, target_height, font_path, style, bubble_vertical)
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    vertical = _is_vertical(box_width, box_height)
    vertical_column_height = box_height * (1 + _VERTICAL_TOLERANCE_RATIO)

    horizontal_req = _horizontal_fit_request(element, target_width, target_height, font_path, style, in_bubble=False)
    # 세로쓰기는 탐지 박스 폭 × (단 나눔 기준 높이)를 목표 영역으로 쓴다.
    vertical_req = replace(
        horizontal_req,
        target_width=box_width,
        target_height=vertical_column_height,
    )

    if vertical:
        wrapped_text, font_size, _ = find_best_fit_font_vertical(vertical_req)
        return wrapped_text, font_size, True, vertical_column_height

    if is_chat_line(element):
        single = _single_line_fit(element, target_width, font_path, style)
        if single is not None:
            return element.translated_text, single, False, None

    wrapped_text, font_size = find_best_fit_font(horizontal_req)

    if (config.ENABLE_VERTICAL_TEXT and box_width > 0
            and box_height >= box_width * _FREEFORM_COLUMN_ASPECT):
        spread = measure_text(wrapped_text, font_path, int(font_size), style)[0] > box_width * _FREEFORM_COLUMN_MAX_SPREAD
        if spread or "\n" in wrapped_text:
            vertical_text, vertical_size, vertical_fits = find_best_fit_font_vertical(vertical_req)
            floor = _FREEFORM_COLUMN_SPREAD_MIN_SIZE_RATIO if spread else _FREEFORM_COLUMN_MIN_SIZE_RATIO
            if vertical_fits and vertical_size >= max(font_size * floor, config.MIN_READABLE_TEXT_SIZE):
                return vertical_text, vertical_size, True, vertical_column_height

    if _should_switch_to_vertical(element, font_size):
        # 늘 세로로 바꾼다 (좁은 효과음 박스는 세로가 자연스러움)
        vertical_wrapped, vertical_size, vertical_fits = find_best_fit_font_vertical(vertical_req)
        logger.debug(
            f"[vertical-switch] text='{(element.translated_text or '')[:20]}...' "
            f"horizontal={font_size} < threshold({config.FONT_SHRINK_THRESHOLD_RATIO}) "
            f"× source({element.font_size}); vertical={vertical_size} (fits={vertical_fits})"
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


def horizontal_candidate(element, target_width, target_height, font_path, style):
    """말풍선 목표 영역 안 가로 배치 (wrapped_text, size) — 식자 쪽에서 세로와 맞출 때 쓴다."""
    return find_best_fit_font(_horizontal_fit_request(element, target_width, target_height, font_path, style))


def vertical_candidate(element, target_width, target_height, font_path, style, max_columns=None):
    """목표 영역 안 세로 (text, size, fits) — 기본은 1단, max_columns를 주면 그 단 수까지 (말밖 긴 세로 캡션)."""
    req = replace(_horizontal_fit_request(element, target_width, target_height, font_path, style),
                  target_width=target_width, target_height=target_height, room=None)
    return find_best_fit_font_vertical(req, max_columns=max_columns)


def plan_bubble_text(element, alignment, target_width, target_height, font_path, style, bubble_box=None,
                     vertical=False):
    """vertical이면 한 단 세로로 (안 들어가면 가로) — 말풍선 방향은 page_drawer가 두 계획을 견줘 정한다."""
    wrapped_text, font_size, vertical, column_height = _fit_text(
        element, target_width, target_height, font_path, style, is_bubble=True, bubble_vertical=vertical
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


def resolve_freeform_geometry(element, page_gray=None) -> FreeformGeometry:
    """프리텍스트의 목표 영역(폭·높이)과 붙음 방향을 정한다 (피팅 전 단계)."""
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    # 가로쓰기는 탐지 박스가 빠듯한 경우가 많아 일정 비율 넘침을 허용한다
    # (중앙 기준으로 양쪽에 고르게 퍼짐).
    if box_width >= box_height:
        overflow_scale = 1.0 + max(0.0, float(config.FREEFORM_BOX_OVERFLOW_RATIO))
    else:
        overflow_scale = 1.0
    target_width = box_width * (1.0 - (config.FREEFORM_PADDING_RATIO * 2)) * overflow_scale
    # 세로형 박스(원본 세로 컬럼 자리)도 가로쓰기가 원칙이므로 줄 폭이
    # 최소 3자는 담도록 보장한다 — 없으면 짧은 어절이 1~2자씩 조각난다.
    # 박스의 2.4배까지만 (옆 그림 침범 가드).
    min_capacity = min(element.font_size * 3.4, box_width * 2.4)
    target_width = max(target_width, min_capacity)
    if box_width > box_height:
        # 가로 상자는 폭도 원래 상자의 1.1배 안으로 — 높이 제한(아래)과 함께 못 맞추면 크기를 줄인다
        target_width = min(target_width, box_width * _FREEFORM_MAX_WIDTH_RATIO
                           - 2 * _FREEFORM_END_MARGIN * float(element.font_size or 0))

    # 컷 경계/이미지 끝 가드: 가로 확장 전 양옆을 스캔해서
    # - 양쪽 열림 → 중앙 기준 확장 (양쪽 여유의 최솟값까지)
    # - 한쪽만 막힘 → 막힌 쪽으로 정렬(기존 attachment 패턴)하고 열린 쪽으로만 확장
    # - 양쪽 막힘 → 확장 포기 (자기 박스 폭 안에서만)
    attachment = getattr(element, "attachment", Attachment.NONE)
    if is_chat_line(element):
        attachment = Attachment.LEFT  # 화면 속 채팅 한 줄은 원문처럼 왼쪽에서 시작
    if page_gray is not None and target_width > box_width:
        margin = 6
        max_reach = int(target_width - box_width) + margin
        left_clear, right_clear = _horizontal_clearance(page_gray, element.text_box, max_reach)
        left_open = left_clear >= margin + 2
        right_open = right_clear >= margin + 2
        # 붙음 판정이 있어도 붙었다는 쪽에 글자 크기의 절반 넘게 빈 자리가 있으면 붙은 게 아니다 — 그림 속 선을 컷 경계로
        # 잘못 보고 붙이면 원문 자리를 벗어난다
        free_reach = 0.5 * float(element.font_size or 0)
        if (attachment == Attachment.LEFT and left_clear >= free_reach and right_open) or (
                attachment == Attachment.RIGHT and right_clear >= free_reach and left_open):
            attachment = Attachment.NONE
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
        # 가로 상자는 옆으로만 넘침을 허용하고 높이는 원래 상자 안으로 — 위아래로 넘치면 옆 칸 글자와 겹친다
        target_height = box_height
    wall_gap = None
    if page_gray is not None:
        wall_gap = _horizontal_clearance(page_gray, element.text_box, int(_WALL_RIM_PX[1]) + 2)
    return FreeformGeometry(target_width=target_width, target_height=target_height, attachment=attachment,
                            wall_gap=wall_gap)


# 붙은 쪽 끝은 원문 상자 끝에 맞추고, 칸 테두리가 가까우면 흰 테두리가 보일 수 있는 가장 넓은 폭(글자 크기의 0.12배를
# 3~8px로 — page_drawer._rim_floor 최대 3px, _follow_rim 상한 0.12배·8px)에 1px을 더 남기고 안쪽으로 당긴다
_WALL_RIM_RATIO = 0.12
_WALL_RIM_PX = (3.0, 8.0)


def _attached_margin(geometry, side, font_size):
    """붙은 쪽(0 = 왼쪽, 1 = 오른쪽) 끝을 원문 상자 밖으로 낼 폭 — 0이거나, 칸 테두리가 가까우면 음수(안쪽으로 당김)."""
    gap = getattr(geometry, "wall_gap", None)
    if gap is None:
        return 0.0
    rim = min(_WALL_RIM_PX[1], max(_WALL_RIM_PX[0], _WALL_RIM_RATIO * float(font_size)))
    return min(0.0, float(gap[side]) - rim - 1.0)


def plan_freeform_text(element, bubble_text_rects, img_size, font_path, style, page_gray=None):
    geometry = resolve_freeform_geometry(element, page_gray=page_gray)
    wrapped_text, font_size, vertical, column_height = _fit_text(
        element, geometry.target_width, geometry.target_height, font_path, style, is_bubble=False
    )
    return place_freeform_plan(
        element, geometry, wrapped_text, font_size, vertical, column_height,
        font_path, style, bubble_text_rects, img_size,
    )


def place_freeform_plan(element, geometry, wrapped_text, font_size, vertical, column_height,
                        font_path, style, bubble_text_rects, img_size, columns=1):
    """이미 정해진 줄바꿈/크기/방향으로 프리텍스트 계획을 배치한다.

    plan_freeform_text의 배치 단계이자, 식자 검수가 줄바꿈·크기 대안을 같은
    규칙(붙음 정렬·겹침 회피·페이지 클램프)으로 다시 놓을 때 쓰는 진입점.
    """
    attachment = geometry.attachment

    # Vertical text renders as a single-column stack; its actual w/h differs
    # from the horizontal measure result. 피팅과 같은 기준으로 측정해야 한다.
    if vertical:
        text_w, text_h = measure_vertical_text(
            wrapped_text, font_path, font_size, style,
            max_column_height=column_height,
            max_columns=columns,
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
        anchor_x = x1 - _attached_margin(geometry, 0, font_size)
        initial_bbox = (anchor_x, center_y - text_h / 2, anchor_x + text_w, center_y + text_h / 2)
    else:  # Attachment.RIGHT
        align, anchor = "right", "rm"
        anchor_x = x2 + _attached_margin(geometry, 1, font_size)
        initial_bbox = (anchor_x - text_w, center_y - text_h / 2, anchor_x, center_y + text_h / 2)

    if not vertical and abs(element.angle) > config.MIN_ROTATION_ANGLE:
        # 기울어진 글은 렌더러가 글 덩어리 가운데를 중심으로 돌려 그린다(compose_rotated_text_layer) — 왼쪽·오른쪽
        # 끝 기준점을 그대로 넘기면 붙은 쪽 절반이 컷 경계·페이지 끝 너머로 그려졌다. 돌린 덩어리가 같은 쪽 끝에
        # 붙도록 가운데 기준점으로 옮기고, 겹침 피하기·페이지 안 맞추기도 돌린 덩어리 크기로 한다
        rad = math.radians(abs(element.angle))
        block_w = text_w * math.cos(rad) + text_h * math.sin(rad)
        block_h = text_w * math.sin(rad) + text_h * math.cos(rad)
        if align == "left":
            left = anchor_x
        elif align == "right":
            left = anchor_x - block_w
        else:
            left = anchor_x - block_w / 2
        anchor = "mm"
        anchor_x = left + block_w / 2
        initial_bbox = (left, center_y - block_h / 2, left + block_w, center_y + block_h / 2)

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
        vertical_columns=columns if vertical else 1,
    )
