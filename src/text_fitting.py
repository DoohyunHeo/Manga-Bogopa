"""Font-size fitting loops for horizontal and vertical text layout.

Public entry points:
    find_best_fit_font(req: FitRequest) -> (wrapped_text, font_size)
    find_best_fit_font_vertical(req: FitRequest) -> (text, font_size, fits)

The horizontal fitter searches over wrap strategies and width ratios; the
vertical fitter iterates sizes top-down.

Caps: source × MODEL_FONT_SIZE_CEILING_RATIO (hard upper bound),
      source × MODEL_FONT_SIZE_FLOOR_RATIO (soft lower bound via penalty),
where source is the measured source font size (FitRequest.initial_font_size).
"""
import logging
import math
import re
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np

from src import config
from src.text_renderer import (
    measure_character_body_height_ratio,
    measure_line,
    measure_text,
    measure_vertical_block,
)
from src.text_wrapping import (
    FORBIDDEN_LINE_PENALTY,
    LINE_HEAD_FORBIDDEN,
    LINE_TAIL_FORBIDDEN,
    PHRASE_END_MARKS,
    TALL_BUBBLE_RATIO,
    build_wrap_candidates,
    is_determiner,
    count_lone_middle_lines,
    count_midword_breaks,
    count_tail_breaks,
    count_unnatural_breaks,
    is_layout_equivalent,
    visible_len,
    wrap_text_korean,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitRequest:
    """한 텍스트 요소의 피팅 입력 묶음.

    가로/세로 피팅과 스코어링 전반을 관통하는 공통 인자를 한 번에 전달한다.
    target_*는 텍스트가 들어가야 할 영역 크기 — 가로/세로 피팅이 서로 다른
    영역을 쓸 수 있다 (text_layout._fit_text 참고).
    """
    text: str
    font_path: str
    style: object  # TextStyle (text_renderer) — measure_* 함수에 그대로 전달
    target_width: float
    target_height: float
    initial_font_size: float
    char_ratio_target: Optional[float] = None
    char_ratio_reference_height: Optional[float] = None
    in_bubble: bool = True  # 말풍선 밖 글자는 세로 채움 점수를 쓰지 않고, 좁은 말풍선 줄 줄이기도 하지 않는다
    room: Optional["BubbleRoom"] = None  # 말풍선 모양 — 있으면 줄마다 그 높이의 폭으로 들어가는지 본다

    @property
    def bubble_ratio(self) -> float:
        return self.target_height / max(self.target_width, 1)


# 말풍선 모양 따라 쓰기 — 줄마다 쓸 수 있는 폭을 그 줄이 놓일 높이에서 말풍선 안쪽이 실제로 넓은 만큼으로 잡는다. 둥근
# 말풍선은 가운데 줄이 넓고 위아래 줄이 좁다 (한국어판의 마름모꼴 배치). 거리 지도는 page_drawer._bubble_interior가
# 지운 쪽에서 재고, 테두리에서 글자 크기의 margin_ratio배를 뗀 곳까지 쓴다. 글자 덩어리는 (cx, cy)를 가운데로 둔다.
# 한글 잉크는 줄 칸 높이의 가운데 _SHAPE_INK_BAND에만 찍힌다 — 줄 칸 전체에서 가장 좁은 곳을 잡으면 가시 끝·곡선이 칸
# 위아래에만 걸려도 폭을 좁게 잡아 가시 말풍선·납작한 말풍선에서 글자가 작아진다
# 줄 폭은 잉크 띠 높이들의 폭 가운데 하위 _SHAPE_ROW_PERCENTILE% — 가장 좁은 곳을 쓰면 가시 말풍선에서 가시 끝이 한 높이만
# 파고들어도 줄 전체가 좁아진다. 실제 잉크가 테두리에 닿는지는 식자 검수(말풍선 안쪽 흰 모양)가 픽셀로 가린다
_SHAPE_MAX_LINES = 8
_SHAPE_INK_BAND = 0.65
_SHAPE_ROW_PERCENTILE = 15


@dataclass(frozen=True, eq=False)
class BubbleRoom:
    dist: np.ndarray      # 안쪽 흰 영역의 테두리까지 거리
    x0: int               # 거리 지도의 쪽 좌표
    y0: int
    cx: float             # 글자 덩어리 가운데
    cy: float
    margin_ratio: float
    _extents: dict = field(default_factory=dict, repr=False)

    def _reach(self, margin: float):
        """행마다 cx에서 왼쪽·오른쪽으로 테두리 여백 안에 머무는 거리 (px) — 여백 값마다 한 번 잰다."""
        key = max(1, int(round(margin)))
        if key not in self._extents:
            ok = self.dist >= key
            c = int(round(self.cx)) - self.x0
            if not 0 <= c < ok.shape[1]:
                zero = np.zeros(ok.shape[0])
                self._extents[key] = (zero, zero)
            else:
                left, right = ok[:, :c + 1][:, ::-1], ok[:, c:]
                self._extents[key] = (np.where(left.all(axis=1), left.shape[1], np.argmin(left, axis=1)).astype(float),
                                      np.where(right.all(axis=1), right.shape[1], np.argmin(right, axis=1)).astype(float))
        return self._extents[key]

    def widths(self, n_lines: int, block_height: float, font_size: float):
        """덩어리를 n_lines줄·높이 block_height로 cy에 두었을 때 줄마다 쓸 수 있는 폭 (줄 칸 가운데 잉크 띠의 좁은 쪽)."""
        left, right = self._reach(self.margin_ratio * float(font_size))
        top = self.cy - block_height / 2.0 - self.y0
        slot = block_height / max(1, n_lines)
        pad = slot * (1.0 - _SHAPE_INK_BAND) / 2.0
        out = []
        for i in range(n_lines):
            a, b = int(math.floor(top + i * slot + pad)), int(math.ceil(top + (i + 1) * slot - pad))
            if a < 0 or b > len(left) or b <= a:
                out.append(0.0)
                continue
            out.append(2.0 * float(np.percentile(np.minimum(left[a:b], right[a:b]), _SHAPE_ROW_PERCENTILE)))
        return out


def _line_overflow(req: FitRequest, wrapped_text: str, font_size, text_h: float) -> float:
    """말풍선 모양에서 줄마다 쓸 수 있는 폭을 넘은 픽셀의 합."""
    lines = [line for line in wrapped_text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    rooms = req.room.widths(len(lines), text_h, font_size)
    return sum(max(0.0, measure_line(line, req.font_path, font_size, req.style) - room) for line, room in zip(lines, rooms))


def fit_overflow_px(req: FitRequest, wrapped_text: str, font_size, text_w: float, text_h: float,
                    width: Optional[float] = None) -> float:
    """목표 영역을 넘은 픽셀 — 말풍선 모양이 있으면 줄마다 그 높이의 폭, 없으면 사각형(폭 width, 기본은 목표 폭)."""
    over_h = max(0.0, text_h - req.target_height)
    if req.room is not None:
        return _line_overflow(req, wrapped_text, font_size, text_h) + over_h
    return max(0.0, text_w - (req.target_width if width is None else width)) + over_h


def fits_target(req: FitRequest, wrapped_text: str, font_size) -> bool:
    """그 크기의 줄바꿈이 목표 영역(말풍선 모양이 있으면 그 모양)에 들어가는지."""
    text_w, text_h = measure_text(wrapped_text, req.font_path, int(font_size), req.style)
    return fit_overflow_px(req, wrapped_text, int(font_size), text_w, text_h) <= 0.0


def _shape_wrap(req: FitRequest):
    """말풍선 모양 따라 줄바꿈 — 줄 수를 늘려 가며 줄마다 그 높이의 폭까지 어절을 채워, 어절이 모두 들어가는 가장 적은
    줄 수. 원문에 정한 줄바꿈이 있거나 끝내 안 들어가면 보통 줄바꿈으로."""
    def shape_wrap(text, font_path, font_size, style, max_width):
        words = text.split()
        if not words or "\n" in text.strip():
            return wrap_text_korean(text, font_path, font_size, style, max_width)
        for n in range(1, min(len(words), _SHAPE_MAX_LINES) + 1):
            block_h = measure_text("\n".join(["가"] * n), font_path, font_size, style)[1]
            lines, i = [], 0
            for room in req.room.widths(n, block_h, font_size):
                line = words[i]
                i += 1
                while i < len(words) and measure_line(f"{line} {words[i]}", font_path, font_size, style) <= room:
                    line = f"{line} {words[i]}"
                    i += 1
                lines.append(line)
                if i >= len(words):
                    break
            if i >= len(words):
                return "\n".join(lines)
        return wrap_text_korean(text, font_path, font_size, style, max_width)
    return shape_wrap


def _wrap_candidates(req: FitRequest):
    """줄바꿈 전략들 — 말풍선 모양이 있으면 모양 따라 줄바꿈을 더한다."""
    candidates = build_wrap_candidates(req.text, req.bubble_ratio)
    return candidates + [_shape_wrap(req)] if req.room is not None else candidates


class FitCandidate(NamedTuple):
    """피팅 탐색이 낳은 후보 하나 — 스코어 비교와 로그에 쓰인다."""
    score: float
    wrapped_text: str
    font_size: int
    wrap_name: str
    width_ratio: float
    fits: bool


def get_char_ratio_target(element):
    """Return the per-element char-height ratio target, or None if absent."""
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


def _find_best_font_size(req: FitRequest, start_size, candidate_width, wrap_fn, min_size=None):
    """가장 큰 '들어가는' 크기를 이분 탐색으로 찾습니다.

    크기가 작아질수록 줄당 단어가 늘어 줄 수·폭·높이가 같이 줄어들므로
    fits(size)는 단조: 한 번 들어가면 더 작은 크기도 들어간다. 선형 1px
    감소 스캔 대비 측정 횟수를 O(범위) → O(log 범위)로 줄인다.
    """
    minimum_size = max(config.MIN_FONT_SIZE, min_size or config.MIN_FONT_SIZE)
    start_size = max(int(start_size), minimum_size)

    def _attempt(size):
        wrapped = wrap_fn(req.text, req.font_path, size, req.style, candidate_width)
        text_w, text_h = measure_text(wrapped, req.font_path, size, req.style)
        return wrapped, fit_overflow_px(req, wrapped, size, text_w, text_h, width=candidate_width)

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


# 원문에서 잰 글자 비율과 렌더 측정치의 상대 오차에 곱하는 감점
FONT_CHAR_SCORE_WEIGHT = 42.0


def _char_ratio_penalty(req: FitRequest, text_for_measure, font_size):
    """원문에서 잰 글자 비율과 렌더 측정치의 상대 오차 페널티 (가로/세로 공통)."""
    reference_height = (
        max(float(req.char_ratio_reference_height), 1.0)
        if req.char_ratio_reference_height is not None
        else req.target_height
    )
    measured_char_ratio = measure_character_body_height_ratio(
        text_for_measure,
        req.font_path,
        font_size,
        req.style,
        reference_height=reference_height,
    )
    char_rel_error = abs(measured_char_ratio - req.char_ratio_target) / max(req.char_ratio_target, 1e-4)
    return char_rel_error * FONT_CHAR_SCORE_WEIGHT


MIDWORD_BREAK_PENALTY = 40.0
# 어절 안 활용 꼬리 앞 줄바꿈('공부/해요'·'따라/가요')은 크기 3px만큼만 깎는다 — 한국어판은
# 좁은 말풍선에서 이 자리를 끊어 가로 두 줄로 크게 쓴다 — 40점을 주면 한 줄로 작게 넣다가 세로쓰기로 넘어간다
TAIL_BREAK_PENALTY = 12.0
# 띄어쓰기 자리라도 붙여 두어야 읽히는 짝 — 한 글자 관형사·부사+다음 어절(두 분·이 거·안 되는), 숫자+단위
# (스무 살·3번), 이름+호칭(○○ 짱), 앞말+한 글자 의존명사(방금 거). 끊으면 한 번에 크기 5px만큼 깎는다
GLUED_BREAK_PENALTY = 20.0
# 뒷말에 붙어 읽히는 한 글자 부사 — 한국어판은 관형사 뒤에서는 끊지 않고, 이런 부사 뒤에서도 드물게 끊는다.
# 그 밖의 한 글자 어절은 여느 어절처럼 끊는다
_CLINGING_ADVERBS = ("안", "못", "잘", "다", "더", "또", "꼭", "좀", "참", "막", "딱", "쭉", "늘", "곧", "꽤", "확", "푹", "싹")
_UNITS = ("살", "번", "개", "명", "원", "권", "화", "회", "시", "분", "초", "년", "월", "일", "장", "층", "등", "배", "킬로", "미터")
_NUMBER_WORDS = ("다섯", "여섯", "일곱", "여덟", "아홉", "스무", "서른", "마흔", "예순", "일흔", "여든", "아흔")
_HONORIFICS = ("짱", "쨩", "씨", "군", "님", "선배", "양", "공")
# 앞말에 붙어 읽히는 한 글자 명사 — 의존명사와 때를 가리키는 말(하루 전·다음 날). '걸·건·게'는 '거'에
# 조사가 줄어 붙은 꼴이다 (거를·거는·것이)
_BOUND_NOUNS = ("거", "것", "걸", "건", "게", "수", "때", "적", "줄", "데", "뿐", "전", "후", "날", "뒤", "쯤", "째")
# 관형사처럼 뒷말을 꾸미는 한 글자 대명사 (내 꿈·제 일)
_POSSESSIVES = ("내", "제", "니")
_EDGE_MARKS = re.compile(r"^[^0-9A-Za-z가-힣]+|[^0-9A-Za-z가-힣]+$")
_KO_PARTICLE_TAIL = re.compile(r"(이|가|은|는|을|를|의|도|만|야|아|이야|한테|에게|랑|이랑|과|와)?[^0-9A-Za-z가-힣]*$")
_PHRASE_END = re.compile(r"[^0-9A-Za-z가-힣]$")
# 문장 부호로 끝난 어절 뒤에 같은 줄로 뒷말을 이으면 한 번에 크기 2px만큼 깎는다 — 한국어판은 이런
# 자리에서 대개 줄을 바꾼다
PHRASE_JOIN_PENALTY = 8.0
# 세 줄 이상인 덩어리에서 SHORT_RUN_CHARS 글자 이하 짧은 줄이 잇따르면 한 쌍마다 깎는다 — 부호 뒤에서 거푸 끊어 짧은 줄이
# 쌓이지 않게 (부호 뒤에 이어 붙이는 감점보다 조금 크게 잡아, 이어 붙인 한 줄이 짧은 두 줄을 이긴다)
SHORT_RUN_CHARS = 3
SHORT_RUN_PENALTY = 10.0
# 말풍선보다 한참 납작한 글자 덩어리를 깎는 기준 — 덩어리 세로/가로가 말풍선 세로/가로의 ASPECT_MIN배(정사각형까지)보다
# 낮으면 모자란 비율의 로그 × ASPECT_PENALTY. 한국어판도 세로로 긴 말풍선에 말풍선보다 납작한 덩어리를 흔히 쓴다 —
# 기준이 높으면 줄이 적은 배치가 들어가도 한 낱말씩 쌓은 기둥을 고른다.
# 0.5는 그런 한국어판 배치 대부분보다 납작한 덩어리만 깎는다
ASPECT_MIN = 0.5
ASPECT_CAP = 1.0
ASPECT_PENALTY = 15.0


def count_phrase_joins(wrapped_text):
    """문장 부호로 끝난 어절 뒤에서 줄을 바꾸지 않고 같은 줄에 뒷말을 이은 곳의 수."""
    return sum(1 for line in (wrapped_text or "").split("\n") for word in line.split()[:-1]
               if word[-1] in PHRASE_END_MARKS)


def _clings_forward(word):
    """한 글자 관형사·부사가 뒤 어절에 붙어 읽히는지('두 분'·'이 거'·'안 되는'). 쉼표·문장 부호로 끝나면('응,'·
    '어?') 거기서 말이 끊겨 아니다 — 그러지 않으면 부호 뒤에서 바꾼 줄이 붙일 짝 끊김으로 깎인다."""
    core = _EDGE_MARKS.sub("", word)
    return len(core) == 1 and not _PHRASE_END.search(word) and (_modifies_next(core) or core in _CLINGING_ADVERBS)


def _modifies_next(core):
    """뒷말을 꾸미는 한 글자 관형사·대명사 ('두 분'·'이 거'·'내 꿈') — 한국어판은 이 뒤에서 끊지 않는다."""
    return is_determiner(core) or core in _POSSESSIVES


def _keep_together(left, right):
    """공백으로 나뉜 두 어절을 한 줄에 두어야 하는지."""
    a, b = _EDGE_MARKS.sub("", left), _EDGE_MARKS.sub("", right)
    if not a or not b:
        return False
    if _clings_forward(left):
        return True
    if (a[-1].isdigit() or a in _NUMBER_WORDS) and b.startswith(_UNITS):
        return True
    core = _KO_PARTICLE_TAIL.sub("", b) or b
    if core in _HONORIFICS:
        return True
    return core in _BOUND_NOUNS


def count_glued_breaks(source_text, wrapped_text):
    """줄 경계 가운데 붙여 두어야 할 두 어절 사이에서 끊은 곳의 수."""
    boundaries = {}
    for paragraph_offset, paragraph in _paragraph_offsets(source_text or ""):
        words = paragraph.split(" ")
        cum = paragraph_offset
        for left, right in zip(words, words[1:]):
            cum += len(left)
            if left and right and _keep_together(left, right):
                boundaries[cum] = True
    count, cum = 0, 0
    lines = [line for line in (wrapped_text or "").split("\n") if line.strip()]
    for line in lines[:-1]:
        cum += len(line.replace(" ", ""))
        count += cum in boundaries
    return count


def count_single_word_breaks(source_text, wrapped_text):
    """줄 경계 가운데 한 글자 관형사 바로 뒤, 또는 한 글자 의존명사 바로 앞에서 끊은 곳의 수."""
    boundaries = set()
    for paragraph_offset, paragraph in _paragraph_offsets(source_text or ""):
        words = paragraph.split(" ")
        cum = paragraph_offset
        for left, right in zip(words, words[1:]):
            cum += len(left)
            right_core = _KO_PARTICLE_TAIL.sub("", _EDGE_MARKS.sub("", right)) or right
            if right and ((_clings_forward(left) and _modifies_next(_EDGE_MARKS.sub("", left)))
                          or right_core in _BOUND_NOUNS):
                boundaries.add(cum)  # 한 글자 관형사 뒤, 또는 의존명사 앞('먹은 / 거')
    count, cum = 0, 0
    for line in [line for line in (wrapped_text or "").split("\n") if line.strip()][:-1]:
        cum += len(line.replace(" ", ""))
        count += cum in boundaries
    return count


def count_determiner_breaks(source_text, wrapped_text):
    """줄 경계 가운데 한 글자 관형사와 꾸미는 말 사이, 또는 수와 단위 사이에서 끊은 곳의 수."""
    boundaries = set()
    for paragraph_offset, paragraph in _paragraph_offsets(source_text or ""):
        words = paragraph.split(" ")
        cum = paragraph_offset
        for left, right in zip(words, words[1:]):
            cum += len(left)
            head = _EDGE_MARKS.sub("", right)
            if not head or not ("가" <= head[0] <= "힣" or head[0].isdigit()):
                continue
            core = _EDGE_MARKS.sub("", left)
            if is_determiner(left) or ((core[-1:].isdigit() or core in _NUMBER_WORDS) and head.startswith(_UNITS)):
                boundaries.add(cum)
    count, cum = 0, 0
    for line in [line for line in (wrapped_text or "").split("\n") if line.strip()][:-1]:
        cum += len(line.replace(" ", ""))
        count += cum in boundaries
    return count


def _paragraph_offsets(text):
    offset = 0
    for paragraph in text.split("\n"):
        yield offset, paragraph
        offset += len(paragraph.replace(" ", ""))


def _whole_word_lines(source_text, lines):
    """원문의 어절을 통째로 담은 줄 번호 — 줄 앞뒤가 모두 원문 공백·끝과 맞는 줄."""
    boundaries = {0}
    n = 0
    for ch in source_text or "":
        if ch in (" ", "\n"):
            boundaries.add(n)
        else:
            n += 1
    boundaries.add(n)
    out, cum = set(), 0
    for idx, line in enumerate(lines):
        start, cum = cum, cum + len(line.replace(" ", ""))
        if start in boundaries and cum in boundaries:
            out.add(idx)
    return out


def _score_wrapped_candidate(req: FitRequest, wrapped_text, font_size, predicted_size=None):
    text_w, text_h = measure_text(wrapped_text, req.font_path, font_size, req.style)
    lines = [line for line in wrapped_text.split('\n') if line.strip()]
    if not lines:
        return float("-inf")

    target_width, target_height = req.target_width, req.target_height
    bubble_ratio = req.bubble_ratio
    line_widths = [measure_line(line, req.font_path, font_size, req.style) for line in lines]
    fit_overflow = fit_overflow_px(req, wrapped_text, font_size, text_w, text_h)
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

    # 세로 채움은 말풍선만 — 말풍선 밖 가로 글자는 줄을 늘려 원래 상자보다 높아지면 옆 칸과 겹친다
    if bubble_ratio <= 0.9:
        if req.in_bubble:
            score -= max(0.0, 0.72 - vertical_fill) * 16.0
        score -= max(0.0, 0.78 - horizontal_fill) * 10.0
        if len(lines) == 1 and visible_len(lines[0]) >= 12:
            score -= 10.0
    elif req.in_bubble:
        score -= max(0.0, 0.58 - vertical_fill) * 8.0

    # 말풍선보다 한참 납작한 글자 덩어리는 깎는다 (기준은 ASPECT_MIN — 한국어판 배치의 덩어리 비율로 정했다)
    if req.in_bubble and text_w > 0 and text_h > 0:
        wanted_aspect = min(ASPECT_MIN * bubble_ratio, ASPECT_CAP)
        if text_h / text_w < wanted_aspect:
            score -= math.log(wanted_aspect / (text_h / text_w)) * ASPECT_PENALTY

    if len(lines) > 1:
        edge_avg = (line_widths[0] + line_widths[-1]) / 2
        middle_max = max(line_widths[1:-1], default=max(line_widths))
        if bubble_ratio >= TALL_BUBBLE_RATIO:
            score -= max(0.0, edge_avg - middle_max) / max(target_width, 1) * 12.0
        else:
            score -= np.std(line_widths) / max(target_width, 1) * 4.0

    whole_word_lines = _whole_word_lines(req.text, lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if idx > 0 and stripped[0] in LINE_HEAD_FORBIDDEN:
            score -= FORBIDDEN_LINE_PENALTY
        if idx < len(lines) - 1 and stripped[-1] in LINE_TAIL_FORBIDDEN:
            score -= FORBIDDEN_LINE_PENALTY
        if idx in (0, len(lines) - 1) and visible_len(stripped) <= 2 and len(lines) >= 3:
            score -= 6.0
        elif 0 < idx < len(lines) - 1 and visible_len(stripped) <= 1 and idx in whole_word_lines:
            score -= MIDWORD_BREAK_PENALTY  # 가운데 한 글자 어절 줄('이')은 어절 중간 끊김과 같은 급

    if len(lines) >= 3:
        score -= SHORT_RUN_PENALTY * sum(1 for a, b in zip(lines, lines[1:])
                                         if visible_len(a) <= SHORT_RUN_CHARS and visible_len(b) <= SHORT_RUN_CHARS)

    # 2줄 레이아웃의 외톨이 끝줄(3+1 분할): 끝줄이 첫 줄의 절반에 한참
    # 못 미치면 균등 분할(2+2) 후보가 이기도록 패널티
    if len(lines) == 2:
        first_len, last_len = visible_len(lines[0]), visible_len(lines[1])
        if last_len <= 2 and last_len * 2 < first_len:
            score -= 6.0

    # 어절 중간 줄바꿈은 '마지막 수단': 한 번에 크기 10px(+40점)만큼 깎는다. 어절 경계만 쓰는 배치가
    # 원문 크기의 정책 하한(0.8배) 안에서 들어가면 크기가 몇 단계 작아지더라도 그쪽이 이기고, 어절 하나가
    # 한 줄에 안 들어갈 만큼 좁을 때만 끊는다 (작게 깎으면 크기 몇 px 이득에 져서 어절이 쪼개진다)
    if req.text:
        score -= count_midword_breaks(req.text, wrapped_text) * MIDWORD_BREAK_PENALTY
        score -= count_tail_breaks(req.text, wrapped_text) * TAIL_BREAK_PENALTY
        score -= count_glued_breaks(req.text, wrapped_text) * GLUED_BREAK_PENALTY
        # 한 글자 어절이 뒤 어절과 떨어진 줄바꿈('두 / 분은'·'이 / 사람')은 어절 중간 끊김과 같은 급으로 더 깎는다
        score -= count_single_word_breaks(req.text, wrapped_text) * (MIDWORD_BREAK_PENALTY - GLUED_BREAK_PENALTY)
        # 활용 꼬리 안(어미 앞)에서 끊은 줄바꿈은 한 번 더 깎는다: '수고/했어!'가
        # '수고했/어!'를 이기되, 꼬리를 지키느라 글씨가 2px 이상 작아지면 크기가 이긴다.
        score -= count_unnatural_breaks(req.text, wrapped_text) * 6.0
        if req.in_bubble:
            score -= count_phrase_joins(wrapped_text) * PHRASE_JOIN_PENALTY

    if req.char_ratio_target is not None:
        score -= _char_ratio_penalty(req, wrapped_text, font_size)

    return score


def _evaluate_fit_candidates(
    req: FitRequest,
    start_size,
    width_ratios,
    wrap_fns,
    minimum_size,
    preferred_minimum_size,
    predicted_size=None,
):
    best_fit_candidate = None
    best_relaxed_candidate = None

    for ratio_index, width_ratio in enumerate(width_ratios):
        candidate_width = max(1.0, req.target_width * width_ratio)
        for wrap_fn in wrap_fns:
            if ratio_index and wrap_fn.__name__ == "shape_wrap":
                continue  # 모양 따라 줄바꿈은 폭 비율과 상관없다
            wrapped_text, found_size, fits = _find_best_font_size(
                req,
                start_size,
                candidate_width,
                wrap_fn,
                min_size=minimum_size,
            )
            score = _score_wrapped_candidate(
                req,
                wrapped_text,
                found_size,
                predicted_size=predicted_size,
            )
            if found_size < preferred_minimum_size:
                score -= (preferred_minimum_size - found_size) * 12.0

            candidate = FitCandidate(score, wrapped_text, found_size, wrap_fn.__name__, width_ratio, fits)
            if fits:
                if best_fit_candidate is None or candidate.score > best_fit_candidate.score:
                    best_fit_candidate = candidate
            elif best_relaxed_candidate is None or candidate.score > best_relaxed_candidate.score:
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


# 말풍선에서 한 줄 평균 2자 이하로 5줄 이상 쪼개지면 크기를 한 단계(8%) 줄여 줄 수가 줄어드는 배치를 찾는다.
# 더 넓게 적용하면 세로로 긴 말풍선에서 글자가 가로로 퍼져 폭 끝에 붙고 위아래가 비어 원문 모양과 멀어졌다.
# 원문 크기의 정책 하한(0.8배) 아래로는 내려가지 않고, 어절 중간 끊김이 늘어나는 배치는 고르지 않는다
_NARROW_MAX_CHARS_PER_LINE = 2.0
_NARROW_MIN_LINES = 5
_NARROW_SIZE_STEPS = (0.92,)


def _chars_per_line(wrapped_text):
    lines = [line for line in (wrapped_text or "").split("\n") if line.strip()]
    return sum(visible_len(line) for line in lines) / max(len(lines), 1), len(lines)


def _fewer_lines_in_narrow_bubble(req: FitRequest, candidate: FitCandidate, floor_size) -> FitCandidate:
    per_line, line_count = _chars_per_line(candidate.wrapped_text)
    if line_count < _NARROW_MIN_LINES or per_line > _NARROW_MAX_CHARS_PER_LINE:
        return candidate
    midword = count_midword_breaks(req.text, candidate.wrapped_text)
    for step in _NARROW_SIZE_STEPS:
        size = int(round(candidate.font_size * step))
        if size < max(floor_size, config.MIN_FONT_SIZE) or size >= candidate.font_size:
            continue
        # 이 크기에서 들어가면서 줄이 줄어드는 배치 가운데 점수가 가장 높은 것 (점수 내림차순)
        for score, text in rank_wraps_for_size(req, size):
            if _chars_per_line(text)[1] >= line_count or count_midword_breaks(req.text, text) > midword:
                continue
            text_w, text_h = measure_text(text, req.font_path, size, req.style)
            if fit_overflow_px(req, text, size, text_w, text_h) <= 0.0:
                return FitCandidate(score, text, size, "narrow_fewer_lines", candidate.width_ratio, True)
    return candidate


# 어절 중간을 끊어야만 들어가는 크기라면, 원문 크기의 이 배까지 줄여서라도 어절 경계만 쓰는 배치를 찾는다.
# 0.8배로 올리면 어절이 쪼개지는 배치가 되살아난다 — 조금 작아도
# 어절이 온전한 쪽이 낫다
WHOLE_WORD_MIN_SIZE_RATIO = 0.65


def _whole_word_fallback(req: FitRequest, candidate: FitCandidate) -> FitCandidate:
    """어절 중간 끊김이 있는 배치 대신, 더 작은 크기에서 어절 경계만 쓰는 가장 큰 배치 (없으면 그대로)."""
    low = max(config.MIN_FONT_SIZE, math.ceil(req.initial_font_size * WHOLE_WORD_MIN_SIZE_RATIO))
    for size in range(int(candidate.font_size) - 1, low - 1, -1):
        for score, text in rank_wraps_for_size(req, size):
            if count_midword_breaks(req.text, text):
                continue
            text_w, text_h = measure_text(text, req.font_path, size, req.style)
            if fit_overflow_px(req, text, size, text_w, text_h) <= 0.0:
                return FitCandidate(score, text, size, "whole_word", candidate.width_ratio, True)
    return candidate


# 한 글자 관형사와 꾸미는 말('두 / 분은'·'한 / 명'), 수와 단위를 가르는 배치라면, 그 크기의 이 배까지 줄여서라도 가르지 않는
# 배치를 찾는다 — 크기를 먼저 고르고 줄바꿈을 나중에 고르면 한 단계 작은 크기의 깔끔한 배치를 놓친다
UNGLUED_MIN_SIZE_RATIO = 0.88


def _unglued_fallback(req: FitRequest, candidate: FitCandidate) -> FitCandidate:
    """관형사·단위를 가르는 배치 대신, 조금 작은 크기에서 가르지 않는 가장 큰 배치 (없으면 그대로)."""
    low = max(config.MIN_FONT_SIZE, math.ceil(candidate.font_size * UNGLUED_MIN_SIZE_RATIO))
    midword = count_midword_breaks(req.text, candidate.wrapped_text)
    for size in range(int(candidate.font_size) - 1, low - 1, -1):
        for score, text in rank_wraps_for_size(req, size):
            if count_determiner_breaks(req.text, text) or count_midword_breaks(req.text, text) > midword:
                continue
            text_w, text_h = measure_text(text, req.font_path, size, req.style)
            if fit_overflow_px(req, text, size, text_w, text_h) <= 0.0:
                return FitCandidate(score, text, size, "unglued", candidate.width_ratio, True)
    return candidate


# 가운데 줄에 한 글자만 남는 배치라면, 그 크기의 이 배까지 줄여서라도
# 그런 줄이 없는 배치를 찾는다 — 점수는 이런 줄을 어절 중간 끊김만큼 깎지만, 줄바꿈 방식마다 가장 크게 들어가는 크기만
# 견줘 한 단계 작은 크기의 깔끔한 배치를 놓친다. 끝줄 한 글자('봐.')는
# 한국어판도 쓰는 자리라 그대로 둔다
LONE_LINE_MIN_SIZE_RATIO = 0.88


def _lone_line_fallback(req: FitRequest, candidate: FitCandidate, floor_size) -> FitCandidate:
    """가운데 한 글자 줄이 있는 배치 대신, 조금 작은 크기에서 그런 줄이 없는 가장 큰 배치 (없으면 그대로)."""
    low = max(int(floor_size), config.MIN_FONT_SIZE, math.ceil(candidate.font_size * LONE_LINE_MIN_SIZE_RATIO))
    base = (count_midword_breaks(req.text, candidate.wrapped_text), count_glued_breaks(req.text, candidate.wrapped_text),
            count_determiner_breaks(req.text, candidate.wrapped_text))
    for size in range(int(candidate.font_size), low - 1, -1):
        for score, text in rank_wraps_for_size(req, size):
            if count_lone_middle_lines(text):
                continue
            counts = (count_midword_breaks(req.text, text), count_glued_breaks(req.text, text),
                      count_determiner_breaks(req.text, text))
            if any(new > old for new, old in zip(counts, base)):
                continue
            text_w, text_h = measure_text(text, req.font_path, size, req.style)
            if fit_overflow_px(req, text, size, text_w, text_h) <= 0.0:
                return FitCandidate(score, text, size, "no_lone_line", candidate.width_ratio, True)
    return candidate


def find_best_fit_font(req: FitRequest):
    max_allowed_size = _get_model_font_upper_bound(req.initial_font_size)
    start_size = min(_get_font_search_start(req.initial_font_size, req.target_height), max_allowed_size)
    preferred_minimum_size = max(
        config.MIN_FONT_SIZE,
        min(config.MAX_FONT_SIZE, math.ceil(req.initial_font_size * config.MODEL_FONT_SIZE_FLOOR_RATIO)),
    )
    width_ratios = _width_ratios_for(req.bubble_ratio)
    wrap_fns = _wrap_candidates(req)

    best_candidate, relaxed_candidate = _evaluate_fit_candidates(
        req,
        start_size,
        width_ratios,
        wrap_fns,
        preferred_minimum_size,
        preferred_minimum_size,
        predicted_size=req.initial_font_size,
    )

    if best_candidate is None:
        best_candidate, fallback_candidate = _evaluate_fit_candidates(
            req,
            start_size,
            width_ratios,
            wrap_fns,
            config.MIN_FONT_SIZE,
            preferred_minimum_size,
            predicted_size=req.initial_font_size,
        )
        if best_candidate is None:
            best_candidate = fallback_candidate or relaxed_candidate

    if best_candidate is None:
        return req.text, max(config.MIN_FONT_SIZE, min(req.initial_font_size, config.MAX_FONT_SIZE))

    if best_candidate.fits and req.text and count_midword_breaks(req.text, best_candidate.wrapped_text):
        best_candidate = _whole_word_fallback(req, best_candidate)
    if best_candidate.fits and req.text and count_determiner_breaks(req.text, best_candidate.wrapped_text):
        best_candidate = _unglued_fallback(req, best_candidate)
    if req.in_bubble and best_candidate.fits:
        best_candidate = _fewer_lines_in_narrow_bubble(req, best_candidate, preferred_minimum_size)
    if req.in_bubble and best_candidate.fits and count_lone_middle_lines(best_candidate.wrapped_text):
        best_candidate = _lone_line_fallback(req, best_candidate, preferred_minimum_size)

    logger.debug(
        f"[font-fit] text='{req.text[:20]}...' target=({req.target_width:.0f}x{req.target_height:.0f}) "
        f"initial={start_size} -> best={best_candidate.font_size}, wrap={best_candidate.wrap_name}, "
        f"width_ratio={best_candidate.width_ratio:.2f}, score={best_candidate.score:.2f}, "
        f"fits={best_candidate.fits}, floor={preferred_minimum_size}, cap={max_allowed_size}"
    )
    return best_candidate.wrapped_text, best_candidate.font_size


def find_best_fit_font_vertical(req: FitRequest, max_columns=None):
    """Returns (text, font_size, fits).

    세로쓰기는 항상 단일 단(1-column)이 기본이다. fits=False면 어떤 크기로도
    한 단에 들어가지 않았다는 뜻 (말풍선은 이 신호로 가로쓰기로 되돌린다).
    """
    max_columns = 1 if max_columns is None else max(1, int(max_columns))

    max_allowed_size = _get_model_font_upper_bound(req.initial_font_size)
    min_allowed_size = max(
        config.MIN_FONT_SIZE,
        min(max_allowed_size, math.ceil(req.initial_font_size * config.MODEL_FONT_SIZE_FLOOR_RATIO)),
    )

    best_score = None
    best_size = None

    # 가로 피팅과 동일: 시작점 = 원문 측정 크기 (가독성 하한 보장, 상한 캡)
    vertical_start_size = max(
        config.MIN_FONT_SIZE,
        min(max_allowed_size, max(int(round(req.initial_font_size)), config.MIN_READABLE_TEXT_SIZE + 2)),
    )

    for current_size in range(vertical_start_size, config.MIN_FONT_SIZE - 1, -1):
        text_w, text_h, n_columns = measure_vertical_block(
            req.text, req.font_path, current_size, req.style,
            max_column_height=req.target_height,
            max_columns=max_columns,
        )
        if n_columns == 0:
            continue
        if text_h > req.target_height or text_w > req.target_width:
            continue

        # 가로 피팅과 동일하게 원문 추정 크기에서 보상 클램프 (전역 인플레 방지)
        score = min(current_size, max(1, int(round(req.initial_font_size)))) * 4.0
        fill_ratio = (text_w * text_h) / max(req.target_width * req.target_height, 1.0)
        if fill_ratio < config.FONT_AREA_FILL_RATIO:
            score -= (config.FONT_AREA_FILL_RATIO - fill_ratio) * 18.0
        if current_size < min_allowed_size:
            score -= (min_allowed_size - current_size) * 12.0
        # 같은 값이면 단 수가 적은 쪽(읽기 쉬운 쪽)을 선호
        score -= (n_columns - 1) * 2.0

        if req.char_ratio_target is not None:
            score -= _char_ratio_penalty(req, req.text, current_size)

        if best_score is None or score > best_score:
            best_score, best_size = score, current_size

    if best_size is not None:
        return req.text, best_size, True

    return req.text, max(config.MIN_FONT_SIZE, min(req.initial_font_size, max_allowed_size)), False


# ── 검수/통일 단계가 쓰는 고정 크기 보조 함수 ──────────────────────────────

def policy_floor_size(initial_font_size) -> int:
    """정책상 글자 크기 하한: 원문 크기 x MODEL_FONT_SIZE_FLOOR_RATIO.

    식자 검수(typeset_qa)의 크기 축소와 페이지 통일(page_consistency)은 이 값
    아래로는 내려가지 않는다 — 경고를 없애려고 정책보다 작게 쓰지 않는다.
    """
    floor_ratio = float(config.MODEL_FONT_SIZE_FLOOR_RATIO)
    floor_size = math.ceil(max(1.0, float(initial_font_size)) * floor_ratio)
    return max(config.MIN_FONT_SIZE, min(config.MAX_FONT_SIZE, floor_size))


def rank_wraps_for_size(req: FitRequest, font_size):
    """고정 글자 크기에서 가능한 줄바꿈 결과를 점수 내림차순 (점수, 텍스트)로 돌려준다.

    크기 탐색은 하지 않고, 기존 줄바꿈 전략 x 폭 비율 조합만 돈다 (중복 제거).
    원문의 글자·어절 경계를 보존하지 않는 결과(공백 삭제 등)는 버린다.
    동점은 텍스트 사전순으로 고정해 결정적이다.
    """
    fixed_size = max(1, int(font_size))
    scored = {}
    wrap_fns = _wrap_candidates(req)
    for ratio_index, width_ratio in enumerate(_width_ratios_for(req.bubble_ratio)):
        candidate_width = max(1.0, req.target_width * width_ratio)
        for wrap_fn in wrap_fns:
            if ratio_index and wrap_fn.__name__ == "shape_wrap":
                continue  # 모양 따라 줄바꿈은 폭 비율과 상관없다
            wrapped = wrap_fn(req.text, req.font_path, fixed_size, req.style, candidate_width)
            if not wrapped.strip() or not is_layout_equivalent(req.text, wrapped):
                continue
            score = _score_wrapped_candidate(req, wrapped, fixed_size, predicted_size=req.initial_font_size)
            if wrapped not in scored or score > scored[wrapped]:
                scored[wrapped] = score
    return sorted(((score, text) for text, score in scored.items()), key=lambda item: (-item[0], item[1]))


def select_wrap_for_size(req: FitRequest, font_size) -> str:
    """고정 글자 크기에서 점수가 가장 높은 줄바꿈 텍스트 하나를 고른다."""
    ranked = rank_wraps_for_size(req, font_size)
    return ranked[0][1] if ranked else req.text


def score_wrap_for_size(req: FitRequest, wrapped_text: str, font_size) -> float:
    """주어진 줄바꿈 텍스트를 고정 크기에서 피팅 스코어러로 채점한다 (후보 비교용)."""
    return _score_wrapped_candidate(req, wrapped_text, max(1, int(font_size)), predicted_size=req.initial_font_size)
