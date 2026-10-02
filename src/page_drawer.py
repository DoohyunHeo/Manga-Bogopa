"""페이지 식자 오케스트레이션.

순서: 말풍선 계획 -> 페이지 대사 크기 통일(page_consistency) -> 말풍선 검수·보정
(typeset_qa) -> 합성 -> 프리텍스트 계획(그려진 말풍선 글자를 본다) -> 검수·보정 -> 합성.
draw_text_on_image는 이미지 배열만 돌려주고, 진단은
render_page_text가 돌려주는 PageTypesetReport와 로그로 노출한다.
"""
import logging
import math
import re
from dataclasses import replace
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src import config, page_consistency, typeset_qa
from src.data_models import Attachment, PageData
from src.text_fitting import BubbleRoom, policy_floor_size
from src.text_layout import (
    FreeformGeometry,
    TextRenderPlan,
    bubble_vertical_rescue,
    horizontal_wrap_alternatives,
    orphan_free_wrap_alternatives,
    place_freeform_plan,
    plan_bubble_text,
    plan_freeform_text,
    apply_text_look,
    resolve_bubble_style,
    resolve_freeform_geometry,
    resolve_freeform_style,
    rewrap_horizontal_at_size,
    short_word_forced_break,
    short_word_split,
    vertical_candidate,
    horizontal_candidate,
)
from src.line_detector import attachment_continues
from src.reading_order import find_panels, panel_of
from src.text_filters import dialogue_size
from src.text_renderer import measure_text, measure_vertical_text, replace_unsupported_chars
from src.text_fitting import count_glued_breaks, count_determiner_breaks
from src.text_wrapping import count_lone_middle_lines, count_midword_breaks, worsens_line_breaks

logger = logging.getLogger(__name__)


# 설정 '외침은 기울여 쓰기'의 기울기 — 한국어판이 외침을 기울이는 각도(7~16도)의 가운데 10도
_EXCLAMATION_SLANT = math.tan(math.radians(10))


def page_dialogue_px(page_data):
    """쪽 대사 중앙 크기 — 말풍선이 적은 쪽은 책 전체 대사 중앙(글씨체 판정 때 붙임)으로 (말풍선이 하나뿐인 쪽은
    크기 맞추기·키우기의 기준이 없다)."""
    return dialogue_size(page_data.speech_bubbles, getattr(page_data, "book_dialogue_px", None))


def _get_alignment_for_bubble(attachment, text_box, bubble_box, page_gray=None):
    """Position text inside a speech bubble.

    앵커는 원문 텍스트 박스가 아니라 **말풍선 중심** 기준이다. 일본어 세로
    텍스트는 말풍선 한쪽에 치우쳐 있는 경우가 많아, 그 박스 중심을 따라가면
    한국어 가로 식자가 위/옆으로 떠 보인다.
    """
    text_x1, _, text_x2, _ = text_box
    bubble_x1, bubble_y1, bubble_x2, bubble_y2 = bubble_box
    center_y = (bubble_y1 + bubble_y2) // 2
    # 컷 테두리에 잘린 말풍선은 원문이 보이는 부분 가운데에 있어도 글자를 벽 쪽에 붙인다 — 한국어판이 그렇게
    # 쓴다. 붙은 쪽 줄이 컷 테두리가 아니면(말풍선 위아래로 이어지지도 컷
    # 모서리로 꺾이지도 않는 번쩍 말풍선의 뾰족한 테두리, 양옆이 곧은 네모·팔각 상자) 가운데로
    if page_gray is not None and not attachment_continues(page_gray, bubble_box, attachment,
                                                          config.BUBBLE_ATTACHMENT_EDGE_RATIO,
                                                          config.BUBBLE_ATTACHMENT_MIN_LENGTH_RATIO):
        attachment = Attachment.NONE

    if attachment == Attachment.LEFT:
        align, anchor = 'left', 'lm'
        center_x = max(text_x1 - config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x1 + config.BUBBLE_EDGE_SAFE_MARGIN)
    elif attachment == Attachment.RIGHT:
        align, anchor = 'right', 'rm'
        center_x = min(text_x2 + config.ATTACHED_BUBBLE_TEXT_MARGIN, bubble_x2 - config.BUBBLE_EDGE_SAFE_MARGIN)
    else:
        align, anchor = 'center', 'mm'
        center_x = (bubble_x1 + bubble_x2) // 2
    return align, anchor, center_x, center_y


@dataclass
class _TextJob:
    """요소 하나의 계획과, 검수·보정이 대안을 만들 때 필요한 문맥."""
    kind: str                       # "bubble" | "freeform"
    index: int
    element: object
    font_path: str
    style: object
    plan: TextRenderPlan
    target_width: float
    target_height: float
    attached: bool = False
    safe_rect: Optional[Tuple[int, int, int, int]] = None
    target_rect: Optional[Tuple[int, int, int, int]] = None
    geometry: Optional[FreeformGeometry] = None   # 프리텍스트만
    occupied_rects: Tuple[tuple, ...] = ()        # 프리텍스트 배치 시 피할 사각형들
    img_size: Tuple[int, int] = (0, 0)
    rendered: Optional[typeset_qa.RenderedText] = None  # 계획을 이미 렌더한 캐시 (plan과 일치할 때만 사용)
    emphasis: bool = False  # 큰 말풍선에 맞춰 키운 외침 — 쪽 안 크기 맞추기에서 뺀다
    interior: Optional[tuple] = field(default=None, compare=False)  # 말풍선만: 안쪽 흰 영역의 테두리까지 거리 (_bubble_interior)
    bubble_box: Optional[Tuple[float, float, float, float]] = None
    ref_size: float = 0.0    # 키우기 상한의 기준 원문 크기 (_reference_size) — 키운 사본이 덮어쓰지 않는다
    entry_size: float = 0.0  # 식자에 들어온 크기 (키우기 전) — 쪽 안 크기 맞추기·줄이기 하한은 이 값을 본다


_HANGUL = re.compile(r"[\uac00-\ud7a3\u3131-\u318e]")
# 늘임표는 하나만 — 한국어판은 늘인 소리를 '네—.'처럼 늘임표 하나로 쓴다. 둘씩 겹쳐 쓰면('네——')
# 좁은 말풍선에서 글자가 작아진다
_DASH_RUN = re.compile("[—―─]{2,}")


def _korean_long_marks(text):
    """한국어 번역문에 남은 일본어 장음 'ー'(ｰ)와 늘임표로 쓴 자모 'ㅡ'를 '—'로 — 한글 사이에서는 어색하다.
    겹쳐 쓴 늘임표('——', '―――')는 하나로 줄인다.

    자모 'ㅡ'는 글자를 이루지 못하고 홀로 쓰인 것이라 늘임표('했다ㅡ!!')로 본다.
    """
    if not text or not _HANGUL.search(text):
        return text
    if any(mark in text for mark in "ーｰㅡ"):
        text = text.replace("ー", "—").replace("ｰ", "—").replace("ㅡ", "—")
    return _DASH_RUN.sub("—", text)


def _prepare_element(element, font_path):
    """그릴 수 없는 특수문자를 대체한 '작업용 복사본'을 만든다.

    page_data에 저장된 번역 원문은 바꾸지 않는다 (체크포인트·재실행 시 원본 유지).
    """
    prepared = replace_unsupported_chars(_korean_long_marks(element.translated_text), font_path)
    if prepared == element.translated_text:
        return element
    return replace(element, translated_text=prepared)


def _rewrap_allowed(element) -> bool:
    """줄바꿈 대안을 만들어도 되는 텍스트인지.

    기존 줄바꿈 전략은 연속 공백/탭을 한 칸으로 정리하므로, 원문 번역에 의도된
    연속 공백·탭이 있으면 보존을 증명할 수 없어 줄바꿈 대안을 만들지 않는다.
    """
    text = element.translated_text or ""
    return "\t" not in text and "  " not in text


def _size_steps(current_size, floor_size):
    """현재 크기 아래로 정책 하한까지 최대 MAX_SIZE_STEPS단계."""
    current = int(current_size)
    low = max(int(floor_size), config.MIN_FONT_SIZE)
    return [size for size in range(current - 1, current - 1 - typeset_qa.MAX_SIZE_STEPS, -1) if size >= low]


def _outward_rect(x1, y1, x2, y2):
    return (int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2)))


def _bubble_safe_rect(bubble_box, target_width, target_height):
    """말풍선 사각 안전 영역 — 기존 배치 규칙이 허용하는 가장 바깥 경계.

    피팅 여백(BUBBLE_PADDING_RATIO)과 붙은 쪽 배치가 쓰는 BUBBLE_EDGE_SAFE_MARGIN
    중 더 관대한 쪽으로 안쪽 여백을 잡는다. 사각형 탐지 박스 기준이므로 타원형
    말풍선의 모서리 안쪽은 보장하지 못한다 (문서 참고).
    """
    x1, y1, x2, y2 = (float(v) for v in bubble_box[:4])
    pad_x = max(0.0, ((x2 - x1) - target_width) / 2)
    pad_y = max(0.0, ((y2 - y1) - target_height) / 2)
    inset_x = min(float(config.BUBBLE_EDGE_SAFE_MARGIN), pad_x)
    inset_y = min(float(config.BUBBLE_EDGE_SAFE_MARGIN), pad_y)
    rect = _outward_rect(x1 + inset_x, y1 + inset_y, x2 - inset_x, y2 - inset_y)
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return _outward_rect(x1, y1, x2, y2)
    return rect


def _bubble_target_rect(bubble_box, target_width, target_height):
    cx = (bubble_box[0] + bubble_box[2]) / 2
    cy = (bubble_box[1] + bubble_box[3]) / 2
    return _outward_rect(cx - target_width / 2, cy - target_height / 2, cx + target_width / 2, cy + target_height / 2)


# 말풍선 가운데 띠(높이의 가운데 이 비율)에서 가운데를 지나는 흰 구간 폭의 하위 20%를 안쪽 폭으로 본다.
# 물결·뾰족 테두리는 탐지 상자 안쪽으로 들어와 있어 상자 기준 폭으로는 글자가 테두리에 닿는다
_INNER_BAND = 0.6
_INNER_WHITE = 200


def _inner_width(gray, bubble_box):
    """말풍선 안쪽 흰 영역의 폭(px). 잴 수 없으면 None."""
    if gray is None:
        return None
    x1, y1, x2, y2 = (int(round(v)) for v in bubble_box[:4])
    h, w = gray.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    cx = (x1 + x2) // 2
    band = int((y2 - y1) * (1 - _INNER_BAND) / 2)
    widths = []
    for y in range(y1 + band, y2 - band, 2):
        row = gray[y, x1:x2] >= _INNER_WHITE
        c = cx - x1
        if not row[c]:
            continue
        left = c - int(np.argmin(row[c::-1])) if not row[:c + 1].all() else 0
        right = c + int(np.argmin(row[c:])) if not row[c:].all() else len(row)
        widths.append(right - left)
    return float(np.percentile(widths, 20)) if len(widths) >= 5 else None


# 말풍선 안쪽 흰 영역 — 지운 쪽에서 글자 자리를 품은 흰 덩어리의 구멍(지우고 남은 얼룩)을 메운 모양. 식자 검수는 글자
# 잉크를 이 모양의 테두리에서 글자 크기의 _OUTLINE_MARGIN_RATIO배 안쪽에 둔다 — 사각 안전 영역만 볼 때는 가시 말풍선의
# 가시, 둥근 말풍선 아래 곡선, 말풍선 안으로 들어온 그림에 글자가 닿는다. 흰 덩어리가 말풍선 상자 넓이의
# _INTERIOR_MIN_SHARE보다 작으면(검은·톤 말풍선) 쓰지 않는다
_OUTLINE_MARGIN_RATIO = 0.2
_INTERIOR_MIN_SHARE = 0.3
_INTERIOR_PAD = 4


def _bubble_interior(gray, bubble_box, text_box):
    """(안쪽 흰 영역에서 테두리까지 거리 지도, 지도의 쪽 좌표 x, y) 또는 None."""
    if gray is None:
        return None
    h, w = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in bubble_box[:4])
    x1, y1, x2, y2 = max(0, x1 - _INTERIOR_PAD), max(0, y1 - _INTERIOR_PAD), min(w, x2 + _INTERIOR_PAD), min(h, y2 + _INTERIOR_PAD)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    white = (gray[y1:y2, x1:x2] >= _INNER_WHITE).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    tx1, ty1 = max(0, int(text_box[0]) - x1), max(0, int(text_box[1]) - y1)
    tx2, ty2 = min(x2 - x1, int(text_box[2]) - x1), min(y2 - y1, int(text_box[3]) - y1)
    inside = labels[ty1:ty2, tx1:tx2]
    inside = inside[inside > 0]
    if inside.size == 0:
        return None
    region = labels == int(np.bincount(inside).argmax())  # 글자 자리에 가장 많이 걸친 흰 덩어리
    _, rest = cv2.connectedComponents((~region).astype(np.uint8), connectivity=8)
    edge = set(np.unique(np.concatenate([rest[0], rest[-1], rest[:, 0], rest[:, -1]]))) - {0}
    region |= (rest > 0) & ~np.isin(rest, list(edge))  # 가장자리에 닿지 않는 조각은 구멍 — 메운다
    if region.sum() < _INTERIOR_MIN_SHARE * (bubble_box[2] - bubble_box[0]) * (bubble_box[3] - bubble_box[1]):
        return None
    return cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3), x1, y1


# 말풍선 모양 따라 쓰기는 매끈한 말풍선만 — 흰 모양 넓이 ÷ 그 모양을 감싼 볼록한 테두리 넓이가 _SMOOTH_MIN_SOLIDITY 이상.
# 둥근·팔각 말풍선은 0.98~1.0, 가시·폭발 말풍선은 0.86~0.90쯤이다. 가시 끝을
# 따라 쓰면 외침이 작아진다 — 식자가도 가시 끝은 두고 가운데 몸통에 크게 쓴다
_SMOOTH_MIN_SOLIDITY = 0.95


def _shape_solidity(interior):
    """안쪽 흰 모양 넓이 ÷ 그 모양을 감싼 볼록한 테두리 넓이 — 잴 수 없으면 None."""
    region = (interior[0] > 0).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(contour))
    return cv2.contourArea(contour) / hull if hull > 0 else None


def _is_smooth_shape(interior):
    """안쪽 흰 모양이 매끈한지 (가시·물결 테두리가 아닌지)."""
    solidity = _shape_solidity(interior)
    return solidity is not None and solidity >= _SMOOTH_MIN_SOLIDITY


# 네모 해설 상자 — 안쪽 흰 영역이 닫혀 있고(말풍선 상자 밖으로 이어지지 않음), 자기 테두리 사각형을 _CAPTION_FILL 이상 채우며,
# 모서리 _CAPTION_CORNERS곳 이상이 각지고(모서리 칸의 _CAPTION_CORNER 이상이 흰 영역), 네 변 바깥 3px 안에 어두운 선이 변
# 길이의 _CAPTION_SIDE 이상 이어지고, 글자 자리 밖 안쪽의 _CAPTION_CLEAN 이상이 흰 바탕(톤 없음)인 말풍선
_CAPTION_FILL = 0.95
_CAPTION_CORNER = 0.8
_CAPTION_CORNERS = 3
_CAPTION_SIDE = 0.8
_CAPTION_CLEAN = 0.95
_CAPTION_LINE = 160    # 테두리 선으로 볼 밝기 (미만)
_CAPTION_PAPER = 230   # 흰 바탕으로 볼 밝기 (이상)


def is_caption_box(gray, bubble_box, text_box):
    """말풍선이 꼬리 없는 네모 해설 상자인가 (page_gray: 원본 흑백 쪽)."""
    interior = _bubble_interior(gray, bubble_box, text_box)
    if interior is None:
        return False
    region = interior[0] > 0
    if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
        return False
    ox, oy = interior[1], interior[2]
    crop = gray[oy:oy + region.shape[0], ox:ox + region.shape[1]]
    ys, xs = np.nonzero(region)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    box = region[y1:y2, x1:x2]
    if box.mean() < _CAPTION_FILL:
        return False
    k = max(3, int(round(0.06 * min(box.shape))))
    corners = (box[:k, :k].mean(), box[:k, -k:].mean(), box[-k:, :k].mean(), box[-k:, -k:].mean())
    if sum(c >= _CAPTION_CORNER for c in corners) < _CAPTION_CORNERS:
        return False
    h, w = crop.shape
    bands = ((crop[max(0, y1 - 3):y1, x1:x2], 0), (crop[y2:min(h, y2 + 3), x1:x2], 0),
             (crop[y1:y2, max(0, x1 - 3):x1], 1), (crop[y1:y2, x2:min(w, x2 + 3)], 1))
    if any(band.size == 0 or (band.min(axis=axis) < _CAPTION_LINE).mean() < _CAPTION_SIDE for band, axis in bands):
        return False
    paper = region.copy()
    paper[max(0, int(text_box[1]) - oy - 3):max(0, int(text_box[3]) - oy + 3),
          max(0, int(text_box[0]) - ox - 3):max(0, int(text_box[2]) - ox + 3)] = False
    return bool(paper.any()) and float((crop[paper] >= _CAPTION_PAPER).mean()) >= _CAPTION_CLEAN


# 큰 말풍선의 짧은 외침 — 글자 덩어리가 말풍선 목표 영역 넓이의 20%에 못 미치면, 원문 크기의 1.27배
# (원문 글자 폭의 약 1.4배)와 쪽 대사 중앙 크기의 1.6배 가운데 작은 값까지 키워 다시 맞춘다 (1단 세로 후보 포함).
# 원문 크기가 쪽 대사 중앙 이상이거나 굵기·모양으로 외침인 대사만 — 보통 대사는 쪽 안 크기 맞추기가 맡는다
# (원문 크기를 작게 잰 외침도 굵기로 알아본다)
_SHOUT_FILL_AREA = 0.2  # 글자 덩어리가 목표 영역 넓이의 이 비율에 못 미칠 때 (가로·세로 절반씩이면 0.25)
_DIALOGUE_FILL_AREA = 0.25
_DECORATED_FILL_AREA = 0.35
_DIALOGUE_GROW_SOURCE = 1.23
_DIALOGUE_GROW_PAGE = 1.4
_SHOUT_GROW_SOURCE = 1.27
_SHOUT_GROW_PAGE = 1.6
# 키우기 상한 — 키우기 단계(짧은 외침·보통 대사 12%·강조·쪽 안 크기 맞추기)는 원문 기준 크기의 이 배를 넘지 않는다.
# 기준 = 식자에 들어온 크기와, 원문 글자 폭으로 잰 보이는 크기(font_relative._match_visible_size가 붙이는 visible_size)
# 가운데 큰 것 — 원문을 작게 잰 줄도 글자 폭만큼은 키울 수 있다. 상한이 없으면 강조 키우기가
# 원문의 두 배 가까이 키운다
_GROW_REF_CAP = 1.27


def _reference_size(element):
    """키우기 상한의 기준 원문 크기 — 식자에 들어온 크기와 원문 글자 폭 기준 보이는 크기 중 큰 것."""
    return max(float(element.font_size or 0), float(getattr(element, "visible_size", 0) or 0))


def _ref_cap(ref_size):
    return int(_GROW_REF_CAP * float(ref_size)) if ref_size else None


def _text_extent(plan):
    if plan.vertical:
        return measure_vertical_text(plan.text, plan.font_path, int(plan.font_size), plan.style,
                                     max_column_height=plan.vertical_column_height,
                                     max_columns=plan.vertical_columns)
    return measure_text(plan.text, plan.font_path, int(plan.font_size), plan.style)


def _grow_short_shout(element, plan, alignment, target_width, target_height, font_path, style, bubble_box, dialogue_px,
                      ref_size=None):
    """(키운 요소, 키운 계획) 또는 None.

    외침은 말풍선 넓이의 20% 미만일 때 (원문 크기 1.27배·쪽 중앙 1.6배까지), 보통 대사도 25% 미만이면 원문 글자 상자에
    묶인 크기를 풀고 말풍선 안쪽 영역 기준으로 다시 맞춘다 (원문 크기 1.23배 — 원문 글자 폭의 약 1.35배 — 와 쪽 중앙
    1.4배까지). 원본은 말풍선을 채우는 세로 글씨인데 한국어가 작은 가로 몇 줄로 남아 말풍선이 비었다. 어절 끊김이
    늘면 쓰지 않는다. 원문 기준 크기(ref_size, 없으면 식자에 들어온 크기)의 _GROW_REF_CAP배는 넘지 않는다
    """
    ref = float(ref_size or element.font_size or 0)
    if not dialogue_px or not element.font_size or not ref:
        return None
    shout = element.font_style_reason == "bold_weight" or element.font_style in ("angry", "pop")
    shout = shout or ref >= dialogue_px
    fill, grow_source, grow_page = ((_SHOUT_FILL_AREA, _SHOUT_GROW_SOURCE, _SHOUT_GROW_PAGE) if shout
                                    else (_DIALOGUE_FILL_AREA, _DIALOGUE_GROW_SOURCE, _DIALOGUE_GROW_PAGE))
    if element.font_style in ("pop", "angry"):
        fill = _DECORATED_FILL_AREA  # 장식 글꼴은 획이 속이 비거나 거칠어 같은 넓이라도 작아 보인다
    width, height = _text_extent(plan)
    if width * height >= fill * target_width * target_height:
        return None
    # 원문 크기를 작게 잰 대사는 원문 기준 배수로는 거의 못 키워 — 쪽 대사 중앙까지는 허용하되,
    # 그것도 원문 기준 크기의 _GROW_REF_CAP배 안에서만
    grown = int(min(grow_page * dialogue_px, max(grow_source * ref, dialogue_px), _GROW_REF_CAP * ref))
    if grown <= element.font_size:
        return None
    ratio = grown / float(element.font_size)
    bigger = replace(element, font_size=grown,
                     font_char_ratio=element.font_char_ratio * ratio if element.font_char_ratio else None)
    grown_plan = plan_bubble_text(bigger, alignment, target_width, target_height, font_path, style,
                                  bubble_box=bubble_box, vertical=plan.vertical)
    if grown_plan.vertical != plan.vertical:
        return None  # 키우기는 가로·세로를 바꾸지 않는다 (키운 크기로 한 단 세로가 안 들어가면 안 키운다)
    source = element.translated_text or ""
    if count_midword_breaks(source, grown_plan.text) > count_midword_breaks(source, plan.text):
        return None
    return (bigger, grown_plan) if grown_plan.font_size > plan.font_size else None


def _final_bubble_plan(element, alignment, target_width, target_height, font_path, style, bubble_box, dialogue_px,
                       vertical, ref_size=None):
    """(요소, 계획, 키웠나) — 한 방향으로 맞추고 짧은 대사는 같은 방향으로 키운, 실제로 그릴 말풍선 계획.
    vertical인데 한 단 세로가 안 들어가면 None."""
    plan = plan_bubble_text(element, alignment, target_width, target_height, font_path, style, bubble_box=bubble_box,
                            vertical=vertical)
    if plan.vertical != vertical:
        return None
    grown = _grow_short_shout(element, plan, alignment, target_width, target_height, font_path, style, bubble_box,
                              dialogue_px, ref_size=ref_size)
    return (grown[0], grown[1], True) if grown is not None else (element, plan, False)


# 이어진 말풍선(탐지가 한 말풍선을 맞닿은 두 상자로 나눈 것)은 한 사람의 말이다 — 굵기는 한 조각이라도 굵으면 모두
# 굵게(보통·굵은 대사끼리), 가로·세로는 글자 상자가 가장 큰 조각을 따른다
_JOINED_GAP_PX = 4
_JOINED_MAX_X_OVERLAP = 0.2


# 맞닿은 두 상자가 탐지가 한 말풍선을 둘로 나눈 것인지 — 지운 그림에서 두 글자 자리가 같은 흰 바탕 덩어리에 있으면 한
# 말풍선, 사이에 테두리가 있어 흰 바탕이 끊기면 서로 다른 말풍선이다 (옆 사람 말풍선에 살짝 닿은 말풍선을 묶지 않게).
# 그림이 없거나 글자 자리가 흰 바탕이 아니면(검은 말풍선) 넓이가 _JOINED_MAX_AREA_RATIO배 안으로 비슷할 때만 묶는다
_JOINED_LIGHT = 200
_JOINED_MAX_AREA_RATIO = 3.0


def _one_interior(page_gray, a, b):
    """두 말풍선 상자를 합친 영역에서 두 글자 자리가 같은 흰 바탕 덩어리(4-연결)인가. 판단 못 하면 None."""
    h, w = page_gray.shape[:2]
    boxes = (a.bubble_box, b.bubble_box)
    x1, y1 = max(0, int(min(bx[0] for bx in boxes))), max(0, int(min(bx[1] for bx in boxes)))
    x2 = min(w, int(math.ceil(max(bx[2] for bx in boxes))))
    y2 = min(h, int(math.ceil(max(bx[3] for bx in boxes))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    light = (np.asarray(page_gray)[y1:y2, x1:x2] >= _JOINED_LIGHT).astype(np.uint8)
    _, labels = cv2.connectedComponents(light, connectivity=4)

    def seed(bubble):
        tx1, ty1, tx2, ty2 = bubble.text_element.text_box[:4]
        sx1, sy1 = max(0, int(tx1) - x1), max(0, int(ty1) - y1)
        sx2, sy2 = min(x2 - x1, int(math.ceil(tx2)) - x1), min(y2 - y1, int(math.ceil(ty2)) - y1)
        if sx2 <= sx1 or sy2 <= sy1:
            return 0
        found = labels[sy1:sy2, sx1:sx2]
        found = found[found > 0]
        return int(np.bincount(found).argmax()) if found.size else 0

    la, lb = seed(a), seed(b)
    if not la or not lb:
        return None
    return la == lb


def _same_bubble(a, b, page_gray):
    if page_gray is not None:
        one = _one_interior(page_gray, a, b)
        if one is not None:
            return one
    area_a, area_b = _box_area(a.bubble_box), _box_area(b.bubble_box)
    return max(area_a, area_b) <= _JOINED_MAX_AREA_RATIO * max(1.0, min(area_a, area_b))


def _joined_bubble_groups(bubbles, page_gray=None):
    """맞닿거나 겹치는 말풍선 상자끼리 묶은 번호 목록들 (둘 이상인 무리만)."""
    parent = list(range(len(bubbles)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(bubbles):
        for j in range(i + 1, len(bubbles)):
            b = bubbles[j].bubble_box
            ab = a.bubble_box
            # 옆으로 맞붙은 두 상자만 — 가로로 거의 겹치지 않고 맞닿으며(틈 4px 이하, 겹침 폭 20% 이하), 세로로는
            # 낮은 쪽 높이의 절반 넘게 겹친다. 그냥 가까이 있는 다른 사람 말풍선까지 묶지 않게
            x_overlap = min(ab[2], b[2]) - max(ab[0], b[0])
            y_overlap = min(ab[3], b[3]) - max(ab[1], b[1])
            narrow = min(ab[2] - ab[0], b[2] - b[0])
            low = min(ab[3] - ab[1], b[3] - b[1])
            if (-_JOINED_GAP_PX <= x_overlap <= _JOINED_MAX_X_OVERLAP * narrow and y_overlap >= 0.5 * low
                    and _same_bubble(a, bubbles[j], page_gray)):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(bubbles)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def _box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _unify_joined_styles(page_data, page_gray=None):
    for group in _joined_bubble_groups(page_data.speech_bubbles, page_gray):
        # 굵기(보통↔외침)만 맞춘다 — 글꼴 모양이 다른 조각(명조·손글씨 등)은 그대로
        members = [i for i in group if page_data.speech_bubbles[i].text_element.font_style in ("standard", "shouting")]
        if len(members) < 2:
            continue
        # 굵음과 보통이 섞이면 굵음으로 — 굵기를 되돌려 가늘게 만들지 않는다
        if any(page_data.speech_bubbles[i].text_element.font_style == "shouting" for i in members):
            for i in members:
                page_data.speech_bubbles[i].text_element.font_style = "shouting"
    _unify_attached_weights(page_data)


# 원문 획 굵기로 굵은 대사가 된 말풍선에 맞닿은 보통 대사는, 원문 획 굵기와 글자 크기가 둘 다 이 배 안이면 굵은 대사로 맞춘다
# (원문에 느낌표가 없어 보통 굵기로 내린 쪽). 획 굵기가 더 다르면 원문도 보통 대사 옆에 굵은 외침을 둔 것이라 그대로 둔다
_ATTACHED_STROKE_RATIO = 1.2
_MEASURED_BOLD = ("bold_weight", "shape_threshold_bold")


def _unify_attached_weights(page_data):
    bubbles = page_data.speech_bubbles
    bold = [b for b in bubbles if b.text_element.font_style == "shouting"
            and b.text_element.font_style_reason in _MEASURED_BOLD and getattr(b.text_element, "stroke_weight", None)]
    for plain in bubbles:
        element = plain.text_element
        weight = getattr(element, "stroke_weight", None)
        if element.font_style != "standard" or not weight:
            continue
        for other in bold:
            source = other.text_element
            if (max(weight, source.stroke_weight) <= _ATTACHED_STROKE_RATIO * min(weight, source.stroke_weight)
                    and _attached_boxes(other.bubble_box, plain.bubble_box) and _same_size_px(source, element)):
                element.font_style = "shouting"
                break


# 방향을 맞추느라 지금 크기의 이 배보다 작아지면 제 방향 그대로 둔다 (가로→세로·세로→가로 모두)
_JOINED_SWITCH_MIN = 0.9


def _unify_joined_directions(page_data, jobs, page_gray=None):
    by_index = {job.index: job for job in jobs}
    for group in _joined_bubble_groups(page_data.speech_bubbles, page_gray):
        members = [by_index[i] for i in group if i in by_index]
        if len(members) < 2 or len({job.plan.vertical for job in members}) < 2:
            continue
        leader = max(members, key=lambda job: _box_area(job.element.text_box))
        if leader.plan.vertical:
            continue  # 가로 조각을 세로 조각 따라 세로로 바꾸지 않는다 — 조각마다 혼자서 구제 시험을 치렀다
        for job in members:
            if not job.plan.vertical:
                continue
            bubble_box = page_data.speech_bubbles[job.index].bubble_box
            text, size = horizontal_candidate(job.element, job.target_width, job.target_height,
                                              job.font_path, job.style)
            if size < _JOINED_SWITCH_MIN * job.plan.font_size:
                continue
            job.plan = replace(job.plan, text=text, font_size=size, vertical=False, vertical_column_height=None,
                               align="center", anchor="mm", center_x=(bubble_box[0] + bubble_box[2]) / 2,
                               center_y=(bubble_box[1] + bubble_box[3]) / 2)


def _plan_speech_bubbles(page_data, page_gray=None) -> List[_TextJob]:
    jobs = []
    dialogue_px = page_dialogue_px(page_data)
    _unify_joined_styles(page_data, page_gray)
    for index, bubble in enumerate(page_data.speech_bubbles):
        element = bubble.text_element
        if not element.translated_text:
            continue

        font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
        ref_size, entry_size = _reference_size(element), float(element.font_size or 0)  # 사본을 만들기 전에 (visible_size)
        element = _prepare_element(element, font_path)

        bubble_width = bubble.bubble_box[2] - bubble.bubble_box[0]
        bubble_height = bubble.bubble_box[3] - bubble.bubble_box[1]
        text_w = element.text_box[2] - element.text_box[0]
        text_h = element.text_box[3] - element.text_box[1]
        target_width = bubble_width * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
        target_height = bubble_height * (1.0 - (config.BUBBLE_PADDING_RATIO * 2))
        wide_width = target_width
        inner = _inner_width(page_gray, bubble.bubble_box)
        if inner is not None:
            # 안쪽 흰 폭에서 양쪽 여백(글자 크기의 _TARGET_MARGIN_RATIO배)을 뺀 만큼까지만 — 타원 말풍선은 거의 그대로
            margin = 2 * _TARGET_MARGIN_RATIO * float(element.font_size or 0)
            box_width = target_width
            target_width = max(bubble_width * 0.4, min(target_width, inner - margin))
            # 다시 맞춰 볼 넓은 줄 폭 — 여백을 테두리 검수 여백(_OUTLINE_MARGIN_RATIO)까지만 뺀다 (_WIDE_RETRY_RATIO)
            wide_width = max(target_width, min(box_width, inner - 2 * _OUTLINE_MARGIN_RATIO * float(element.font_size or 0)))
        logger.debug(
            f"[bubble] '{element.translated_text[:15]}...' bubble=({bubble_width}x{bubble_height}) "
            f"text_box=({text_w:.0f}x{text_h:.0f}) target=({target_width:.0f}x{target_height:.0f}) "
            f"pred_font={element.font_size}"
        )

        style = resolve_bubble_style(element, bubble.bubble_box, target_width, target_height)
        style = apply_text_look(style, element.look, freeform=False, font_size=element.font_size) or style
        # 설정 '외침은 기울여 쓰기' — 느낌표가 붙은 대사의 가로 계획만 기울인다 (세로 후보는 똑바로)
        across = style
        if config.SLANT_EXCLAMATIONS and re.search("[!！]", element.translated_text or ""):
            across = replace(style, slant=_EXCLAMATION_SLANT)
        alignment = _get_alignment_for_bubble(bubble.attachment, element.text_box, bubble.bubble_box, page_gray)
        interior = _bubble_interior(page_gray, bubble.bubble_box, element.text_box)
        if interior is not None and alignment[0] == "center" and _is_smooth_shape(interior):
            # 가운데 정렬의 매끈한 말풍선은 줄마다 그 높이의 말풍선 안쪽 폭으로 맞춘다 (벽에 붙인 글자·가시 말풍선은 사각
            # 목표 영역 그대로 — 테두리는 식자 검수가 지킨다)
            element = replace(element, room=BubbleRoom(*interior, cx=alignment[2], cy=alignment[3],
                                                       margin_ratio=_OUTLINE_MARGIN_RATIO))
        # 가로와 한 단 세로를 각각 실제로 그릴 계획(맞춤·키우기까지)으로 만들어 구제 시험으로 고른다 — 가로가 기본
        original = element
        element, plan, grown = _final_bubble_plan(original, alignment, target_width, target_height, font_path, across,
                                                  bubble.bubble_box, dialogue_px, vertical=False, ref_size=ref_size)
        upright = (_final_bubble_plan(original, alignment, target_width, target_height, font_path, style,
                                      bubble.bubble_box, dialogue_px, vertical=True, ref_size=ref_size)
                   if config.ENABLE_VERTICAL_TEXT else None)
        forced = False
        word = short_word_split(original, plan.text)
        if word is not None:
            # 한 줄이 들어가는지는 그린 말풍선 안쪽 폭으로 본다 — 탐지 상자가 말풍선 일부만 잡아 목표 폭이 좁아도 검수가 안쪽
            # 흰 영역에 한 줄로 다시 맞춘다
            probe = original
            if probe.room is None and interior is not None:
                probe = replace(original, room=BubbleRoom(*interior, cx=alignment[2], cy=alignment[3],
                                                          margin_ratio=_OUTLINE_MARGIN_RATIO))
            forced = short_word_forced_break(probe, plan.text, target_width, target_height, font_path, across,
                                             plan.font_size)
            if not forced:
                # 그린 말풍선에는 한 줄로 들어간다 — 뾰족한 말풍선은 좁은 네모 목표 영역으로 맞추느라 짧은 말도 끊고, 끊긴 배치는
                # 테두리를 넘지 않아 검수도 고치지 않는다
                plan = replace(plan, text=word)
        if upright is not None and bubble_vertical_rescue(original, plan.font_size, upright[1].font_size, forced):
            element, plan, grown = upright
        elif original.room is None and wide_width > target_width + 1 and plan.font_size < _WIDE_RETRY_RATIO * entry_size:
            wide = _final_bubble_plan(original, alignment, wide_width, target_height, font_path, across,
                                      bubble.bubble_box, dialogue_px, vertical=False, ref_size=ref_size)
            source = original.translated_text or ""
            if (wide[1].font_size > plan.font_size and short_word_split(original, wide[1].text) is None
                    and count_midword_breaks(source, wide[1].text) <= count_midword_breaks(source, plan.text)):
                element, plan, grown = wide
                target_width = wide_width
        jobs.append(_TextJob(
            kind="bubble", index=index, element=element, font_path=font_path, style=style if plan.vertical else across,
            plan=plan,
            target_width=target_width, target_height=target_height, emphasis=grown,
            interior=interior,
            bubble_box=bubble.bubble_box, ref_size=ref_size, entry_size=entry_size,
            attached=bubble.attachment != Attachment.NONE,
            safe_rect=_bubble_safe_rect(bubble.bubble_box, target_width, target_height),
            target_rect=_bubble_target_rect(bubble.bubble_box, target_width, target_height),
        ))
    return jobs


# ── 검수·보정이 쓰는 대안 생성기 ─────────────────────────────────────────────

def _bubble_wrap_variants(job: _TextJob):
    def build(plan):
        if plan.vertical or not _rewrap_allowed(job.element):
            return []
        texts = horizontal_wrap_alternatives(
            job.element, job.target_width, job.target_height, job.font_path, job.style,
            plan.font_size, typeset_qa.MAX_WRAP_ALTERNATIVES + 1,
        )
        return [replace(plan, text=text) for text in texts if text != plan.text][:typeset_qa.MAX_WRAP_ALTERNATIVES]
    return build


def _bubble_size_variants(job: _TextJob):
    def build(plan):
        out = []
        # 줄이기 하한은 식자에 들어온 크기로 — 키운 사본의 크기로 재면 키운 줄은 원래 크기까지도 못 줄인다
        for size in _size_steps(plan.font_size, policy_floor_size(job.entry_size or job.element.font_size)):
            if plan.vertical or not _rewrap_allowed(job.element):
                out.append(replace(plan, font_size=size))  # 줄바꿈은 그대로, 크기만
            else:
                wrapped, _ = rewrap_horizontal_at_size(
                    job.element, job.target_width, job.target_height, job.font_path, job.style, size
                )
                out.append(replace(plan, text=wrapped, font_size=size))
        return out
    return build


def _bubble_soft_variants(job: _TextJob):
    def build(plan):
        if plan.vertical or not _rewrap_allowed(job.element):
            return []
        texts = orphan_free_wrap_alternatives(
            job.element, job.target_width, job.target_height, job.font_path, job.style,
            plan.font_size, plan.text, typeset_qa.MAX_SOFT_ALTERNATIVES,
        )
        return [replace(plan, text=text) for text in texts]
    return build


def _place_freeform_variant(job: _TextJob, text, size, vertical, column_height, columns=1):
    return _follow_rim(place_freeform_plan(
        job.element, job.geometry, text, size, vertical, column_height,
        job.font_path, job.style, list(job.occupied_rects), job.img_size, columns=columns,
    ), job.element)


def _freeform_wrap_variants(job: _TextJob):
    def build(plan):
        if plan.vertical or not _rewrap_allowed(job.element):
            return []
        texts = horizontal_wrap_alternatives(
            job.element, job.geometry.target_width, job.geometry.target_height, job.font_path, job.style,
            plan.font_size, typeset_qa.MAX_WRAP_ALTERNATIVES + 1, in_bubble=False,
        )
        return [
            _place_freeform_variant(job, text, plan.font_size, False, None)
            for text in texts if text != plan.text
        ][:typeset_qa.MAX_WRAP_ALTERNATIVES]
    return build


def _freeform_size_variants(job: _TextJob):
    def build(plan):
        out = []
        for size in _size_steps(plan.font_size, policy_floor_size(job.element.font_size)):
            if plan.vertical:
                out.append(_place_freeform_variant(job, plan.text, size, True, plan.vertical_column_height,
                                                   plan.vertical_columns))
            elif not _rewrap_allowed(job.element):
                out.append(_place_freeform_variant(job, plan.text, size, False, None))
            else:
                wrapped, _ = rewrap_horizontal_at_size(
                    job.element, job.geometry.target_width, job.geometry.target_height,
                    job.font_path, job.style, size, in_bubble=False,
                )
                out.append(_place_freeform_variant(job, wrapped, size, False, None))
        return out
    return build


def _freeform_soft_variants(job: _TextJob):
    def build(plan):
        if plan.vertical or not _rewrap_allowed(job.element):
            return []
        texts = orphan_free_wrap_alternatives(
            job.element, job.geometry.target_width, job.geometry.target_height, job.font_path, job.style,
            plan.font_size, plan.text, typeset_qa.MAX_SOFT_ALTERNATIVES, in_bubble=False,
        )
        return [_place_freeform_variant(job, text, plan.font_size, False, None) for text in texts]
    return build


# ── 페이지 통일 ─────────────────────────────────────────────────────────────

# 말풍선 보통 대사는 지금 크기보다 최대 12% 큰 크기를 먼저 해 본다 — 안쪽 영역(여백 포함)에 들어가고 어절 끊김·붙일 짝
# 끊김이 늘지 않을 때만 (보통 대사는 말풍선 여백에 비해 작게 들어가기 쉽다). 쪽 안 크기 맞추기는 그 뒤 크기로 한다
_DIALOGUE_GROW = 1.12
_GROW_STYLES = ("standard", "narration", "handwriting", "scared")
# 말풍선 글자가 쪽 대사 중앙 크기(1단계 측정)의 이 배보다 작으면 들어가는 데까지 키운다 — 세로·손글씨 대사는 쪽 안 크기
# 맞추기 대상이 아니라 작게 남는다
_MIN_SIZE_OF_PAGE = 0.7


def _fits_plan(job, plan):
    if plan.vertical:
        width, height = measure_vertical_text(plan.text, plan.font_path, int(plan.font_size), plan.style,
                                              max_column_height=plan.vertical_column_height,
                                              max_columns=plan.vertical_columns)
    else:
        width, height = measure_text(plan.text, plan.font_path, int(plan.font_size), plan.style)
    return width <= job.target_width and height <= job.target_height


def _resized(job, size):
    """그 크기로 다시 줄바꿈한 계획 (들어가지 않거나 끊김·가운데 한 글자 줄이 늘면 None)."""
    plan = job.plan
    if plan.vertical:
        candidate = replace(plan, font_size=size)
        return candidate if _fits_plan(job, candidate) else None
    if not _rewrap_allowed(job.element):
        return None
    wrapped, fits = rewrap_horizontal_at_size(job.element, job.target_width, job.target_height,
                                              job.font_path, job.style, size)
    if not fits:
        return None
    source = job.element.translated_text
    if (count_midword_breaks(source, wrapped) > count_midword_breaks(source, plan.text)
            or count_glued_breaks(source, wrapped) > count_glued_breaks(source, plan.text)
            or count_determiner_breaks(source, wrapped) > count_determiner_breaks(source, plan.text)
            or count_lone_middle_lines(wrapped) > count_lone_middle_lines(plan.text)):
        return None
    return replace(plan, text=wrapped, font_size=size)


def _grow_dialogue(jobs, dialogue_px):
    for job in jobs:
        if job.emphasis:
            continue
        current = int(job.plan.font_size)
        small = dialogue_px and current < _MIN_SIZE_OF_PAGE * dialogue_px
        if job.element.font_style not in _GROW_STYLES and not small:
            continue
        top = int(current * _DIALOGUE_GROW)
        if small:
            top = max(top, int(round(_MIN_SIZE_OF_PAGE * dialogue_px)))
        cap = _ref_cap(job.ref_size)
        if cap:
            top = min(top, cap)
        for size in range(top, current, -1):
            candidate = _resized(job, size)
            if candidate is not None:
                job.plan = candidate
                break


# 강조 대사 — 원문이 쪽 대사 중앙 크기의 1.25배 이상이거나 굵은 외침·장식 글씨로 판정된 말풍선은 한국어도 쪽 대사 중앙의
# 1.25~1.5배를 목표로 키운다(들어가는 데까지). 강조가 보통 대사와 같은 크기로 들어가 원문의 크기 차이가 사라졌다
_EMPHASIS_SOURCE_RATIO = 1.25
_EMPHASIS_MIN = 1.25
_EMPHASIS_MAX = 1.5
_EMPHASIS_MAX_FILL = 0.6  # 글자 덩어리가 목표 영역의 이 비율을 넘게 키우지 않는다 — 말풍선이 답답해졌다


def _is_emphasis(element, dialogue_px, ref=None):
    """원문 기준 크기(ref, 없으면 _reference_size)가 쪽 대사 중앙의 1.25배 이상이거나, 굵기·모양으로 강조이면서 기준 크기가
    쪽 대사 이상인 줄. 굵기·모양만으로 강조인 작은 줄(기준 크기 < 쪽 대사)은 쪽 대사 1.25~1.5배 목표를 쓰지 않는다."""
    if not dialogue_px:
        return False
    size = float(ref) if ref else _reference_size(element)
    if size >= _EMPHASIS_SOURCE_RATIO * dialogue_px:
        return True
    styled = (element.font_style_reason in ("bold_weight", "shape_threshold_bold")
              or element.font_style in ("angry", "pop"))
    return styled and size >= dialogue_px


def _grow_emphasis(jobs, dialogue_px, skip=()):
    for job in jobs:
        if job.index in skip or not _is_emphasis(job.element, dialogue_px, job.ref_size):
            continue
        current = int(job.plan.font_size)
        top = int(_EMPHASIS_MAX * dialogue_px)
        cap = _ref_cap(job.ref_size)
        if cap:
            top = min(top, cap)
        if current < _EMPHASIS_MIN * dialogue_px:
            for size in range(top, current, -1):
                candidate = _resized(job, size)
                if candidate is None:
                    continue
                width, height = _text_extent(candidate)
                if width * height <= _EMPHASIS_MAX_FILL * job.target_width * job.target_height:
                    job.plan = candidate
                    break
        job.emphasis = True  # 쪽 안 크기 맞추기가 보통 대사 크기로 되돌리지 않게


# 화면 글줄 묶음 — 채팅 목록·LINE 창처럼 가로 글자 상자들이 세로로 가지런히 쌓였거나 한 줄에 나란하면 한 화면으로 보고
# 글자 크기(가장 작은 것)·글씨체(가장 많은 것)를 맞춘다 — 줄마다 따로 맞추면 크기와 굵기가 들쭉날쭉하다
_SCREEN_MIN_ASPECT = 1.8    # 가로 ÷ 세로 이 값 이상인 상자만 글줄로 본다
_SCREEN_SIZE_RATIO = 1.5    # 묶음 안 원문 글자 크기가 이 배 안으로 고르다
_SCREEN_MIN_LINES = 3
_SCREEN_GAP = 1.2           # 위아래 틈이 상자 높이의 이 배 이하
_SCREEN_ALIGN = 1.0         # 왼쪽·가운데·오른쪽 끝 가운데 하나가 상자 높이의 이 배 안으로 맞아야 쌓인 것
_SCREEN_ROW_GAP = 1.0       # 한 줄에 나란한 두 상자의 옆 틈 (상자 높이 배)


def _screen_line_groups(page_data):
    """[(("bubble"|"freeform", 번호), ...), ...] — 둘 이상 묶인 화면 글줄 무리."""
    items = [(("bubble", i), b.text_element) for i, b in enumerate(page_data.speech_bubbles)]
    items += [(("freeform", i), t) for i, t in enumerate(page_data.freeform_texts)]
    lines = []
    for key, element in items:
        x1, y1, x2, y2 = element.text_box[:4]
        if element.translated_text and element.font_size and (x2 - x1) >= _SCREEN_MIN_ASPECT * (y2 - y1):
            lines.append((key, element))
    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (_, a) in enumerate(lines):
        for j in range(i + 1, len(lines)):
            b = lines[j][1]
            if max(a.font_size, b.font_size) > _SCREEN_SIZE_RATIO * min(a.font_size, b.font_size):
                continue
            ax1, ay1, ax2, ay2 = a.text_box[:4]
            bx1, by1, bx2, by2 = b.text_box[:4]
            h = max(ay2 - ay1, by2 - by1)
            v_gap = max(ay1, by1) - min(ay2, by2)
            h_gap = max(ax1, bx1) - min(ax2, bx2)
            stacked = (0 <= v_gap <= _SCREEN_GAP * h and h_gap < 0
                       and min(abs(ax1 - bx1), abs(ax2 - bx2), abs((ax1 + ax2 - bx1 - bx2) / 2)) <= _SCREEN_ALIGN * h)
            row = -v_gap >= 0.5 * min(ay2 - ay1, by2 - by1) and 0 <= h_gap <= _SCREEN_ROW_GAP * h
            if stacked or row:
                parent[find(i)] = find(j)
    groups = {}
    for i, (key, _) in enumerate(lines):
        groups.setdefault(find(i), []).append(key)
    sizes = {key: element.font_size for key, element in lines}
    # 셋 이상 쌓이고 크기가 고른 것만 — 표지·예고의 제목·부제·안내(크기 1.5배 넘게 차이)를 묶어 줄이지 않게
    light = {key for key, element in lines if (getattr(element, "look", None) or {}).get("polarity") == "light"
             or getattr(element, "style_inverted", False)}
    # 둘뿐인 묶음은 둘 다 어두운 바탕의 흰 글자(채팅 화면 줄)일 때만
    return [tuple(g) for g in groups.values()
            if (len(g) >= _SCREEN_MIN_LINES or (len(g) == 2 and all(k in light for k in g)))
            and max(sizes[k] for k in g) <= _SCREEN_SIZE_RATIO * min(sizes[k] for k in g)]


def _unify_screen_styles(page_data, groups):
    """묶음마다 가장 많은 글씨체로 맞춘다 (같으면 보통체)."""
    for group in groups:
        elements = [page_data.speech_bubbles[i].text_element if kind == "bubble" else page_data.freeform_texts[i]
                    for kind, i in group]
        counts = {}
        for element in elements:
            counts[element.font_style] = counts.get(element.font_style, 0) + 1
        top = max(counts.values())
        style = "standard" if counts.get("standard") == top else max(counts, key=counts.get)
        for element in elements:
            if element.font_style != style:
                element.font_style, element.font_style_reason = style, "screen_group"


def _screen_group_sizes(img_pil, page_data, groups, bubble_jobs):
    """묶음마다 가장 작은 식자 크기로 맞춘다 — 말풍선 계획은 바로 줄이고, 말풍선 밖 글자는 {번호: 크기}로 돌려준다."""
    by_index = {job.index: job for job in bubble_jobs}
    page_gray = np.asarray(img_pil.convert("L"))
    dialogue_px = page_dialogue_px(page_data)
    img_size = (img_pil.width, img_pil.height)
    sizes = {}
    for group in groups:
        planned = {}
        for kind, i in group:
            if kind == "bubble":
                if i in by_index and not by_index[i].plan.vertical:
                    planned[(kind, i)] = int(by_index[i].plan.font_size)
            elif page_data.freeform_texts[i].translated_text:
                element, font_path, style = _freeform_setup(img_pil, page_data.freeform_texts[i], dialogue_px)
                plan = plan_freeform_text(element, [], img_size, font_path, style, page_gray=page_gray)
                if not plan.vertical:
                    planned[(kind, i)] = int(plan.font_size)
        if len(planned) < 2:
            continue
        size = min(planned.values())
        for (kind, i), current in planned.items():
            if current <= size:
                continue
            if kind == "freeform":
                sizes[i] = size
                continue
            job = by_index[i]
            job.plan = _resized(job, size) or replace(job.plan, font_size=size)
            job.emphasis = True
    return sizes


# 맞닿은 말풍선끼리, 같은 칸의 말풍선 밖 평문끼리는 같은 크기로 쓴다. 글씨체(보통·해설·굵은 대사)가 다르거나 원문 글자 크기가
# _SAME_SIZE_RATIO배 넘게 다른 글은 묶지 않는다
_ATTACHED_GAP_PX = 6     # 말풍선 상자끼리 이 틈 안으로 맞닿거나 겹치고
_ATTACHED_FACE = 0.3     # 마주 보는 길이가 작은 쪽 높이(옆으로 붙음)나 폭(위아래로 붙음)의 이 배 이상 — 모서리만 닿은 것은 빼고
_SAME_SIZE_RATIO = 1.2   # 한 무리 안 원문 글자 크기의 최대 ÷ 최소
_ATTACHED_SHRINK_FLOOR = 0.75  # 맞닿은 무리를 맞추려고 어느 말풍선도 지금 크기의 이 배 밑으로 줄이지 않는다 — 그래야 들어가는 작은 상자가 끼면 맞추지 않는다
_SAME_SIZE_STYLES = ("standard", "narration", "shouting")


def _source_px(element):
    """원문 글자 크기 — 원문 글자 폭으로 잰 보이는 크기(font_relative._match_visible_size), 못 쟀으면 1단계 크기."""
    return float(getattr(element, "visible_size", 0) or element.font_size or 0)


def _same_size_px(a, b):
    """두 원문 글자 크기가 _SAME_SIZE_RATIO배 안인가."""
    sa, sb = _source_px(a), _source_px(b)
    return sa > 0 and sb > 0 and max(sa, sb) <= _SAME_SIZE_RATIO * min(sa, sb)


def _same_size_source(a, b):
    """두 원문이 같은 글씨체의 대사·평문이고 글자 크기가 _SAME_SIZE_RATIO배 안인가."""
    return a.font_style in _SAME_SIZE_STYLES and a.font_style == b.font_style and _same_size_px(a, b)


def _attached_boxes(a, b):
    """두 말풍선 상자가 옆이나 위아래로 맞닿거나 겹치는가."""
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    if x_overlap < -_ATTACHED_GAP_PX or y_overlap < -_ATTACHED_GAP_PX:
        return False
    return (y_overlap >= _ATTACHED_FACE * min(a[3] - a[1], b[3] - b[1])
            or x_overlap >= _ATTACHED_FACE * min(a[2] - a[0], b[2] - b[0]))


def _linked_groups(keys, linked):
    """linked(a, b)로 이어지는 키 무리들 (둘 이상인 것만)."""
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if linked(keys[i], keys[j]):
                parent[find(i)] = find(j)
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(find(i), []).append(key)
    return [g for g in groups.values() if len(g) > 1]


def _size_clusters(groups, source):
    """무리마다 원문 크기(source) 순으로 놓고 가장 작은 글의 _SAME_SIZE_RATIO배 안까지만 한 무리로 나눈다 — 이웃끼리만 비슷해도
    이어져 작은 글과 큰 글이 한 무리가 되지 않게 (둘 이상인 것만)."""
    clusters = []
    for group in groups:
        run = []
        for key in sorted(group, key=source):
            if run and source(key) > _SAME_SIZE_RATIO * source(run[0]):
                clusters.append(run)
                run = []
            run.append(key)
        clusters.append(run)
    return [c for c in clusters if len(c) > 1]


def _plan_at(job, size):
    """그 크기의 계획 — 줄이기는 늘 되고(다시 줄바꿈이 끊김을 늘리면 줄바꿈 그대로), 키우기는 들어가고 키우기 상한 안일 때만."""
    current = int(job.plan.font_size)
    if size == current:
        return job.plan
    if size > current:
        cap = _ref_cap(job.ref_size)
        return None if cap and size > cap else _resized(job, size)
    return _resized(job, size) or replace(job.plan, font_size=size)


def _unify_attached_sizes(page_data, jobs, img_size):
    """맞닿은 말풍선 무리마다 크기 하나로 — 쪽 보통 대사 크기 쪽으로, 모두 들어가는 가장 큰 크기. 무리(말풍선 번호들)를 돌려준다."""
    by_index = {job.index: job for job in jobs}

    def linked(i, j):
        a, b = by_index[i], by_index[j]
        return (a.emphasis == b.emphasis and _attached_boxes(a.bubble_box, b.bubble_box)
                and _same_size_source(page_data.speech_bubbles[i].text_element, page_data.speech_bubbles[j].text_element))

    groups = _size_clusters(_linked_groups(sorted(by_index), linked),
                            lambda i: _source_px(page_data.speech_bubbles[i].text_element))
    if not groups:
        return groups
    usual = [int(job.plan.font_size) for job in jobs
             if not job.emphasis and not job.plan.vertical and job.element.font_style in ("standard", "narration")]
    renders = {job.index: job.rendered if job.rendered is not None and job.rendered.plan == job.plan
               else typeset_qa.render_plan(job.plan) for job in jobs}
    for group in groups:
        members = [by_index[i] for i in group]
        sizes = [int(job.plan.font_size) for job in members]
        low, high = min(sizes), max(sizes)
        if low == high:
            continue
        anchor = float(np.median(usual)) if len(usual) >= 3 else float(np.median(sizes))
        floor = max(low, int(math.ceil(_ATTACHED_SHRINK_FLOOR * high)))
        for size in range(int(round(min(high, max(floor, anchor)))), floor - 1, -1):
            plans = [_plan_at(job, size) for job in members]
            if any(plan is None for plan in plans):
                continue
            fresh = {}
            for job, plan in zip(members, plans):
                if plan.font_size <= job.plan.font_size:
                    continue  # 줄이면 잘림·벗어남·겹침이 늘지 않는다
                ctx = typeset_qa.QAContext(img_size=img_size, safe_rect=job.safe_rect,
                                           occupied=tuple(r for i, r in renders.items() if i != job.index),
                                           interior=job.interior, interior_margin_ratio=_OUTLINE_MARGIN_RATIO)
                before, candidate = typeset_qa.evaluate(renders[job.index], ctx), typeset_qa.render_plan(plan)
                after = typeset_qa.evaluate(candidate, ctx)
                if (after.clipped_px > before.clipped_px or after.outside_px > before.outside_px
                        or after.collision_px > before.collision_px):
                    break
                fresh[job.index] = candidate
            else:
                for job, plan in zip(members, plans):
                    if plan is not job.plan:
                        job.plan = plan
                        renders[job.index] = job.rendered = fresh.get(job.index) or typeset_qa.render_plan(plan)
                logger.debug("[attached] 말풍선 %s 크기 %s -> %d", group, sizes, size)
                break
    return groups


# 검수가 무리 가운데 하나만 더 줄였으면(테두리·겹침을 고치느라) 나머지도 그 크기로 다시 검수한다 (최대 _MATCH_ROUNDS번)
_MATCH_ROUNDS = 2


def _match_attached_finals(jobs, finalized, groups, img_size, report):
    position = {job.index: k for k, job in enumerate(jobs)}
    for group in groups:
        for _ in range(_MATCH_ROUNDS):
            sizes = {i: int(finalized[position[i]].plan.font_size) for i in group}
            size = min(sizes.values())
            if max(sizes.values()) == size or size < _ATTACHED_SHRINK_FLOOR * max(sizes.values()):
                break
            for i, current in sizes.items():
                if current <= size:
                    continue
                k = position[i]
                job = jobs[k]
                job.plan, job.rendered = _plan_at(job, size), None
                report.elements = [e for e in report.elements if not (e.kind == "bubble" and e.index == i)]
                finalized[k] = _finalize_job(job, img_size, [r for n, r in enumerate(finalized) if n != k], report)


def _panel_narration_sizes(img_pil, page_data, dialogue_px):
    """같은 칸 말풍선 밖 평문 무리마다 가장 작은 식자 크기로 — {번호: 크기}. 가로 글은 가로 글끼리, 세로 글은 세로 글끼리."""
    texts = page_data.freeform_texts
    keys = [i for i, text in enumerate(texts) if text.translated_text and text.font_style in ("standard", "narration")]
    if len(keys) < 2:
        return {}
    erase = [bubble.bubble_box for bubble in page_data.speech_bubbles] + [text.text_box for text in texts]
    panels = find_panels(np.asarray(img_pil.convert("RGB")), erase)
    panel = {i: panel_of(texts[i].text_box, panels) if panels else 0 for i in keys}
    groups = _size_clusters(_linked_groups(keys, lambda i, j: panel[i] == panel[j] and _same_size_source(texts[i], texts[j])),
                            lambda i: _source_px(texts[i]))
    page_gray = np.asarray(img_pil.convert("L"))
    img_size = (img_pil.width, img_pil.height)
    sizes = {}
    for group in groups:
        planned = {False: {}, True: {}}
        for i in group:
            element, font_path, style = _freeform_setup(img_pil, texts[i], dialogue_px)
            plan = plan_freeform_text(element, [], img_size, font_path, style, page_gray=page_gray)
            planned[bool(plan.vertical)][i] = int(plan.font_size)
        for same_direction in planned.values():
            if len(same_direction) < 2:
                continue
            size = min(same_direction.values())
            sizes.update({i: size for i, current in same_direction.items() if current > size})
    return sizes


def _apply_dialogue_consistency(jobs: List[_TextJob], img_size, report):
    """비교 가능한 대사 말풍선 크기를 글자 몸통 기준으로 맞춘다 (제한된 폭, 알파 재검증 포함).

    제안된 크기는 실제 렌더 알파로 다시 검사한다: 페이지 밖 잘림·사각 안전 영역
    이탈·다른 말풍선 글자와의 겹침 중 하나라도 조정 전보다 늘면 그 제안은 버린다.
    렌더 결과는 job.rendered에 캐시해 검수 단계가 재사용한다.
    """
    if len(jobs) < page_consistency.MIN_SAMPLES:
        return
    by_index = {job.index: job for job in jobs}  # 빈 말풍선이 걸러져도 원래 말풍선 번호로 보고
    members = [
        page_consistency.ConsistencyMember(
            key=job.index, plan=job.plan,
            style_name="emphasis" if job.emphasis else job.element.font_style,
            predicted_size=float(job.entry_size or job.element.font_size),
        )
        for job in jobs
    ]

    def rewrap(member, size):
        job = by_index[member.key]
        if _rewrap_allowed(job.element):
            wrapped, fits = rewrap_horizontal_at_size(
                job.element, job.target_width, job.target_height, job.font_path, job.style, size
            )
        else:
            wrapped, fits = job.plan.text, size <= job.plan.font_size
        source = job.element.translated_text
        if fits and (worsens_line_breaks(source, job.plan.text, wrapped)
                     or count_determiner_breaks(source, wrapped) > count_determiner_breaks(source, job.plan.text)):
            return None  # 크기를 맞추려고 단어를 쪼개거나 한 글자 줄을 만들지 않는다
        return replace(job.plan, text=wrapped, font_size=size) if fits else None

    proposals = page_consistency.plan_dialogue_consistency(members, rewrap)
    if not proposals:
        return
    renders = {job.index: typeset_qa.render_plan(job.plan) for job in jobs}
    for key, (plan, adjustment) in proposals.items():
        job = by_index[key]
        cap = _ref_cap(job.ref_size)
        if cap and plan.font_size > max(cap, job.plan.font_size):
            continue  # 크기 맞추기도 원문 기준 크기의 _GROW_REF_CAP배를 넘겨 키우지 않는다
        others = tuple(r for i, r in renders.items() if i != key)
        ctx = typeset_qa.QAContext(img_size=img_size, safe_rect=job.safe_rect, occupied=others,
                                   interior=job.interior, interior_margin_ratio=_OUTLINE_MARGIN_RATIO)
        before = typeset_qa.evaluate(renders[key], ctx)
        candidate = typeset_qa.render_plan(plan)
        after = typeset_qa.evaluate(candidate, ctx)
        if (after.clipped_px > before.clipped_px or after.outside_px > before.outside_px
                or after.collision_px > before.collision_px):
            logger.debug("[page-consistency] bubble#%d 크기 %d->%d 는 잘림/경계/겹침이 늘어 포기",
                         key, adjustment.size_before, adjustment.size_after)
            continue
        job.plan = plan
        renders[key] = candidate
        report.consistency.append(adjustment)
    for job in jobs:
        job.rendered = renders[job.index]


# ── 검수·보정·합성 ───────────────────────────────────────────────────────────

def _finalize_job(job: _TextJob, img_size, finalized, report) -> typeset_qa.RenderedText:
    # 말풍선 모양 따라 맞춘 글자는 안쪽 흰 모양 검사(테두리에서 글자 크기 _OUTLINE_MARGIN_RATIO배)가 여백을 맡는다 — 사각
    # 영역을 글자 크기의 _TEXT_MARGIN_RATIO배 더 좁히면 모양 따라 넓게 쓴 가운데 줄을 다시 막는다
    shaped = job.kind == "bubble" and getattr(job.element, "room", None) is not None
    ctx = typeset_qa.QAContext(
        img_size=img_size,
        safe_rect=(_with_text_margin(job.safe_rect, job.plan.font_size) if job.kind == "bubble" and not shaped
                   else job.safe_rect),
        optical_rect=job.target_rect if job.kind == "bubble" else None,
        occupied=tuple(finalized),
        interior=job.interior if job.kind == "bubble" else None,
        interior_margin_ratio=_OUTLINE_MARGIN_RATIO,
    )
    policy = typeset_qa.ShiftPolicy.for_alignment(job.plan.align, job.attached)
    if job.kind == "bubble":
        factories = (_bubble_wrap_variants(job), _bubble_size_variants(job), _bubble_soft_variants(job))
    else:
        factories = (_freeform_wrap_variants(job), _freeform_size_variants(job), _freeform_soft_variants(job))
    outcome = typeset_qa.repair(
        job.plan, ctx, policy=policy,
        wrap_alternatives=factories[0], size_alternatives=factories[1], soft_wrap_alternatives=factories[2],
        base_rendered=job.rendered, source_text=job.element.translated_text,
    )
    report.elements.append(typeset_qa.ElementReport(
        kind=job.kind, index=job.index,
        text_preview=(job.plan.text or "").replace("\n", " ")[:12],
        strategy=outcome.strategy, renders=outcome.renders,
        baseline=outcome.baseline, final=outcome.final,
        plan_before=job.plan, plan_after=outcome.rendered.plan,
    ))
    return outcome.rendered


# 말풍선 글자 잉크는 말풍선 경계에서 글자 크기의 이 배 이상 떨어져야 한다 — 큰 글자가 테두리에 붙어 보였다.
# 어기면 식자 검수가 위치 조정 → 줄바꿈 → 축소 순으로 고친다
_TEXT_MARGIN_RATIO = 0.5
_TARGET_MARGIN_RATIO = 0.35  # 처음 맞출 때 안쪽 폭에서 빼는 여백 (0.5로 잡으면 좁은 말풍선이 한두 글자 폭이 돼 어절이 쪼개졌다)
# 말풍선 모양대로 맞추지 않는(네모 목표 영역) 말풍선의 가로 계획이 원문 크기의 이 배 아래로 들어가면, 안쪽 폭에서 여백을
# 테두리 검수 여백(_OUTLINE_MARGIN_RATIO)만큼만 뺀 줄 폭으로 한 번 더 맞춰 더 크면 그 계획과 줄 폭을 쓴다 — 처음 여백은
# 글자 크기에 비례해 큰 글자일수록 말풍선을 좁게 보고, 어절을 지키느라 원문보다 한참 작게 넣는다. 세로 구제는 처음 줄 폭의
# 계획으로 먼저 판단하고, 짧은 말을 끊거나 어절 중간 끊김이 늘어나는 넓은 계획은 쓰지 않는다
_WIDE_RETRY_RATIO = 0.9


def _with_text_margin(safe_rect, font_size):
    """안전 영역을 (이미 둔 BUBBLE_EDGE_SAFE_MARGIN 너머로) 글자 크기의 _TEXT_MARGIN_RATIO배까지 더 좁힌다."""
    if safe_rect is None:
        return None
    extra = max(0, int(round(_TEXT_MARGIN_RATIO * float(font_size))) - int(config.BUBBLE_EDGE_SAFE_MARGIN))
    x1, y1, x2, y2 = safe_rect
    if extra <= 0 or x2 - x1 <= 4 * extra or y2 - y1 <= 4 * extra:
        return safe_rect
    return (x1 + extra, y1 + extra, x2 - extra, y2 - extra)


# 말풍선 밖 제목·큰 해설 글씨(쪽 대사 중앙 크기의 이 배 이상)는 흰 테두리를 글자 크기에 맞춰 두껍게 — 고정 2px은
# 큰 제목에서 집중선·그림에 묻힌다. 흰 테두리를 그리는 경우에만 두께를 바꾼다
_TITLE_SIZE_RATIO = 1.6
_TITLE_RIM_RATIO = 0.06
_TITLE_RIM_PX = (2.0, 8.0)


def _title_rim(style, font_size, dialogue_px):
    if not dialogue_px or not font_size or font_size < _TITLE_SIZE_RATIO * dialogue_px:
        return style
    if style.stroke_width <= 0 or min(style.stroke_color[:3]) < 200:
        return style
    width = min(_TITLE_RIM_PX[1], max(_TITLE_RIM_PX[0], _TITLE_RIM_RATIO * float(font_size)))
    return replace(style, stroke_width=max(style.stroke_width, width))


_DARK_BACKGROUND_LUMA = 100  # 이보다 어두운 픽셀을 먹칠로 본다
# 지운 자리에서 먹칠 픽셀이 이 비율 이상일 때만 흰 글자로 바꾼다 — 평균 밝기(100 미만)로 가르면 어두운 톤·그림 위의 흰 외곽선
# 검은 글자가 흰 글자로 뒤집힌다. 0.9(거의 다 어두울 때)로 좁히면 먹칠 그라데이션·어두운 톤 위
# 흰 글자(먹칠 0.65~0.9)가 검은 글자로 바뀐다
_DARK_BACKGROUND_SHARE = 0.6


def _invert_style_for_dark_background(img_pil, text_box, style):
    """어두운 컷(먹칠 배경) 위 프리텍스트는 글자색·외곽선색을 맞바꿔 대비를 확보."""
    x1, y1, x2, y2 = (max(0, int(v)) for v in text_box[:4])
    x2 = min(img_pil.width, x2)
    y2 = min(img_pil.height, y2)
    if x2 <= x1 or y2 <= y1:
        return style
    region = np.asarray(img_pil.crop((x1, y1, x2, y2)).convert("L"))
    if region.size == 0 or float((region < _DARK_BACKGROUND_LUMA).mean()) < _DARK_BACKGROUND_SHARE:
        return style
    return replace(style, color=style.stroke_color, stroke_color=style.color)


# 말풍선 밖 글자 테두리는 글자 밖으로 보이는 폭이 글자 크기의 0.07배(1.5~3px)는 되게 한다 — 설정 두께 2는 skia 전체
# 두께라 밖으로 1px만 보이고, 3배로 그렸다 줄이면 회색 번짐이 돼 톤·그림 위 흰 외곽선 검은 글자와 검은 외곽선 흰 글자가
# 그림과 갈리지 않는다. 흰·검은 테두리
# 모두 — 흰 종이·먹칠 위에서는 바탕과 같은 색이라 보이지 않는다. 바쁜 바탕·제목 테두리는 이보다 두꺼울 수 있다.
# 두께 0(테두리 없음)은 그대로 둔다
_FREEFORM_RIM_RATIO = 0.07
_FREEFORM_RIM_PX = (1.5, 3.0)


def _rim_floor(font_size):
    """말풍선 밖 글자 테두리가 글자 밖으로 보여야 하는 최소 폭 (원문 글자 크기 기준)."""
    return min(_FREEFORM_RIM_PX[1], max(_FREEFORM_RIM_PX[0], _FREEFORM_RIM_RATIO * float(font_size or 0)))


def _visible_rim(style, font_size):
    if style.stroke_width <= 0:
        return style
    return replace(style, stroke_width=max(float(style.stroke_width), 2.0 * _rim_floor(font_size)))  # skia 테두리는 윤곽선 양쪽


# 원문 테두리 두께 따라 하기 — 지운 그림을 바탕 삼아 잰 원문 테두리 폭(look["rim_px"], font_attributes.measure_rim)을
# 글자 크기처럼 한국어 크기 비율로 옮긴다. 자동으로 잰 폭은 실제보다 1px 남짓 두껍게 나와 1px 빼고,
# 스티커처럼 보이지 않게 한국어 글자 크기의 0.12배·8px로 막는다(apply_text_look의 외곽선 상한과 같다). 기본 테두리(_rim_floor)보다
# 얇게는 그리지 않고, 잰 테두리 색이 그릴 글자의 테두리 색과 같을 때만 따른다. 배치 크기가 정해지는 곳마다 다시 계산한다
_RIM_FOLLOW_BIAS_PX = 1.0
_RIM_FOLLOW_MAX_RATIO = 0.12
_RIM_FOLLOW_MAX_PX = 8.0


def _follow_rim(plan, element):
    look = element.look or {}
    rim_px, polarity = look.get("rim_px"), look.get("rim_polarity")
    if not rim_px or not polarity or not element.font_size or plan.style.stroke_width <= 0:
        return plan
    if (min(plan.style.stroke_color[:3]) >= 128) != (polarity == "dark"):  # 검은 글자엔 흰 테두리, 흰 글자엔 검은 테두리
        return plan
    size = float(plan.font_size)
    scaled = (float(rim_px) - _RIM_FOLLOW_BIAS_PX) * size / float(element.font_size)
    visible = max(_rim_floor(element.font_size), min(_RIM_FOLLOW_MAX_PX, _RIM_FOLLOW_MAX_RATIO * size, scaled))
    return replace(plan, style=replace(plan.style, stroke_width=2.0 * visible))


def _freeform_setup(img_pil, element, dialogue_px):
    """(준비된 요소, 글꼴 경로, 글자 모양) — 말풍선 밖 글자. 화면 글줄 크기 맞추기와 식자가 모두 이 모양을 쓴다."""
    font_path = config.FONT_MAP.get(element.font_style, config.DEFAULT_FONT_PATH)
    element = _prepare_element(element, font_path)
    box_width = element.text_box[2] - element.text_box[0]
    box_height = element.text_box[3] - element.text_box[1]
    style = resolve_freeform_style(element, box_width, box_height)
    # 원문 글자색·외곽선을 잴 수 있었으면 그대로 따르고, 모르면 검은 글자+흰 테두리 — 지운 자리가 먹칠이면 흰 글자
    style = (apply_text_look(style, element.look, freeform=True, font_size=element.font_size)
             or _invert_style_for_dark_background(img_pil, element.text_box, style))
    style = _busy_background_rim(img_pil, element, style)
    style = _title_rim(style, element.font_size, dialogue_px)
    return element, font_path, _visible_rim(style, element.font_size)


# 지운 자리 뒤에 칸 테두리·그림 선이 지나가면(어두운 픽셀이 조금 있지만 먹칠 바탕은 아님) 검은 글자에 흰 테두리를 두른다 —
# 테두리 없이 얹으면 글자가 칸 선과 엉켜 보인다 (원문도 대개 흰 테두리를 두른다)
_BUSY_DARK = (0.03, 0.5)   # 글자 상자 안 어두운 픽셀 비율이 이 사이면 선이 지나는 바탕
_BUSY_RIM_RATIO = 0.12
_BUSY_RIM_MIN = 2.0


def _busy_background_rim(img_pil, element, style):
    # 큰 글씨는 제목 테두리 상한까지만 — 아주 큰 외침에 두꺼운 테두리가 둘러져 스티커처럼 보이지 않게
    width = min(_TITLE_RIM_PX[1], max(_BUSY_RIM_MIN, _BUSY_RIM_RATIO * float(element.font_size or 0)))
    if style.stroke_width >= width or sum(style.color[:3]) > 3 * 128:
        return style
    x1, y1, x2, y2 = (max(0, int(v)) for v in element.text_box[:4])
    x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)
    if x2 <= x1 or y2 <= y1:
        return style
    region = np.asarray(img_pil.crop((x1, y1, x2, y2)).convert("L"))
    dark = float((region < _DARK_BACKGROUND_LUMA).mean())
    if not (_BUSY_DARK[0] <= dark < _BUSY_DARK[1]):
        return style
    return replace(style, stroke_width=width, stroke_color=(255, 255, 255))


# 말풍선 밖 긴 세로 캡션 — 지금도 세로로 그리는 길쭉한 말밖 상자(세로 ÷ 가로 _CAPTION_MIN_ASPECT 이상: 강제 세로 6배와,
# 가로가 줄어들어 세로로 바꾸는 text_layout._should_switch_to_vertical의 4배)에서 한 단 크기가 쪽 대사의
# _CAPTION_RESCUE_RATIO배에 못 미치고 여러 단이 한 단의 _CAPTION_MIN_GAIN배 이상 커질 때만, 원문처럼 오른쪽에서 왼쪽으로
# 여러 단을 원래 상자 폭 안에 쓴다. 크기는 식자에 들어온 크기(1단계 크기를 원문 글자 폭 기준으로 고친 값)를 넘지 않는다.
# 말풍선과 가로 글은 바꾸지 않는다. _CAPTION_MIN_GAIN을 1.2배보다 낮추면 끝 몇 글자만 다음 단으로 떨어지는
# 배치까지 고른다
_CAPTION_RESCUE_RATIO = 0.6
_CAPTION_MIN_ASPECT = 4.0
_CAPTION_MIN_GAIN = 1.2
_CAPTION_MAX_COLUMNS = 8


def _multi_column_caption(element, plan, font_path, style, dialogue_px):
    """(단 수, 크기, 글) 또는 None — 긴 세로 캡션을 여러 단으로 살릴 때만."""
    x1, y1, x2, y2 = element.text_box[:4]
    box_w, box_h = x2 - x1, y2 - y1
    if (not dialogue_px or not plan.vertical or plan.vertical_columns != 1 or box_w <= 0
            or box_h < _CAPTION_MIN_ASPECT * box_w or plan.font_size >= _CAPTION_RESCUE_RATIO * dialogue_px):
        return None
    column_height = plan.vertical_column_height or box_h
    limit = int(element.font_size or 0)
    best = None
    for columns in range(2, _CAPTION_MAX_COLUMNS + 1):
        text, size, fits = vertical_candidate(element, box_w, column_height, font_path, style, max_columns=columns)
        size = min(int(size), limit)
        if fits and (best is None or size > best[1]):
            best = (columns, size, text)
    if best is None or best[1] < _CAPTION_MIN_GAIN * plan.font_size:
        return None
    return best


def _draw_freeform_texts(img_pil, page_data, bubble_text_rects, finalized=None, report=None, screen_sizes=None):
    """Draw translated freeform text.

    겹침 회피 대상에 '이미 그린 프리텍스트'도 누적한다 — 효과음이 군집한
    컷(효과음 연타 등)에서 프리텍스트끼리 포개지는 것 방지.
    """
    finalized = [] if finalized is None else finalized
    report = report if report is not None else typeset_qa.PageTypesetReport()
    img_size = (img_pil.width, img_pil.height)
    occupied_rects = list(bubble_text_rects)
    # 컷 경계 스캔용 그레이스케일 (가로 확장이 옆 칸/이미지 끝을 넘지 않게)
    page_gray = np.asarray(img_pil.convert("L"))
    dialogue_px = page_dialogue_px(page_data)
    for index, element in enumerate(page_data.freeform_texts):
        if not element.translated_text:
            continue

        element, font_path, style = _freeform_setup(img_pil, element, dialogue_px)
        plan = _follow_rim(plan_freeform_text(element, occupied_rects, img_size, font_path, style, page_gray=page_gray),
                           element)
        geometry = resolve_freeform_geometry(element, page_gray=page_gray)
        rescued = _multi_column_caption(element, plan, font_path, style, dialogue_px)
        if rescued is not None:
            columns, size, text = rescued
            plan = _follow_rim(place_freeform_plan(element, geometry, text, size, True, plan.vertical_column_height,
                                                   font_path, style, occupied_rects, img_size, columns=columns), element)
        forced = (screen_sizes or {}).get(index)
        if forced is not None and forced < plan.font_size and not plan.vertical:
            # 한 줄이던 글줄은 작아져도 한 줄 그대로 둔다
            wrapped = plan.text if "\n" not in plan.text else rewrap_horizontal_at_size(
                element, geometry.target_width, geometry.target_height, font_path, style, forced, in_bubble=False)[0]
            plan = _follow_rim(place_freeform_plan(element, geometry, wrapped, forced, False, None, font_path, style,
                                                   occupied_rects, img_size), element)
        elif forced is not None and forced < plan.font_size:
            plan = _follow_rim(place_freeform_plan(element, geometry, plan.text, forced, True, plan.vertical_column_height,
                                                   font_path, style, occupied_rects, img_size,
                                                   columns=plan.vertical_columns), element)
        job = _TextJob(
            kind="freeform", index=index, element=element, font_path=font_path, style=style, plan=plan,
            target_width=geometry.target_width, target_height=geometry.target_height,
            attached=plan.align != "center", geometry=geometry,
            occupied_rects=tuple(occupied_rects), img_size=img_size,
        )
        rendered = _finalize_job(job, img_size, finalized, report)
        typeset_qa.composite(img_pil, rendered)
        finalized.append(rendered)
        occupied_rects.append(rendered.bbox)


def render_page_text(inpainted_image, page_data: PageData):
    """번역 글자를 그리고 (이미지 배열, PageTypesetReport)를 돌려준다."""
    img_pil = Image.fromarray(inpainted_image)
    img_size = (img_pil.width, img_pil.height)
    report = typeset_qa.PageTypesetReport()

    screen_groups = _screen_line_groups(page_data)
    _unify_screen_styles(page_data, screen_groups)
    screen_bubbles = {i for group in screen_groups for kind, i in group if kind == "bubble"}
    dialogue_px = page_dialogue_px(page_data)
    page_gray = np.asarray(img_pil.convert("L"))
    bubble_jobs = _plan_speech_bubbles(page_data, page_gray)
    _unify_joined_directions(page_data, bubble_jobs, page_gray)
    _grow_dialogue(bubble_jobs, dialogue_px)
    _grow_emphasis(bubble_jobs, dialogue_px, skip=screen_bubbles)
    _apply_dialogue_consistency(bubble_jobs, img_size, report)
    attached_groups = _unify_attached_sizes(page_data, bubble_jobs, img_size)
    screen_sizes = _screen_group_sizes(img_pil, page_data, screen_groups, bubble_jobs)
    for index, size in _panel_narration_sizes(img_pil, page_data, dialogue_px).items():
        screen_sizes[index] = min(size, screen_sizes.get(index, size))

    finalized: List[typeset_qa.RenderedText] = []
    for job in bubble_jobs:
        finalized.append(_finalize_job(job, img_size, finalized, report))
    _match_attached_finals(bubble_jobs, finalized, attached_groups, img_size, report)
    bubble_text_rects = []
    for rendered in finalized:
        typeset_qa.composite(img_pil, rendered)
        bubble_text_rects.append(rendered.bbox)

    _draw_freeform_texts(img_pil, page_data, bubble_text_rects, finalized=finalized, report=report,
                         screen_sizes=screen_sizes)

    report.log(page_data.source_page)
    return np.array(img_pil), report


def draw_text_on_image(inpainted_image, page_data: PageData):
    """Draw translated text on the inpainted image."""
    image, _report = render_page_text(inpainted_image, page_data)
    return image
