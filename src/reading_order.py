"""대사의 읽는 순서 추정 — 컷을 먼저 나누고, 컷 안에서는 줄 단위로 정렬한다.

일본 만화는 오른쪽 위 컷부터 읽는다. 흰 홈통(컷 사이 여백)이 영역을 가로지르면 위아래로,
세로지르면 좌우로 나누기를 되풀이해 컷 순서를 정한다. 말풍선과 글자는 홈통에 걸쳐 그려지는
일이 많아 나누기 전에 지운다. 나눈 칸은 페이지를 빈틈없이 덮고, 대사는 중심이 든 칸의 컷에
속한다. 두 쪽 펼침면은 오른쪽 면부터 따로 나눈다. 그림이 없거나 홈통을 못 찾으면 페이지 전체를
한 컷으로 보고 줄 단위로만 정렬한다. 컷 바깥 여백과 가장자리 광고 기둥의 글자(공지·감수 표기·
쪽 번호 등)는 맨 뒤로 보낸다.
"""
from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.data_models import PageData, TextElement

Box = Tuple[int, int, int, int]
Masks = Tuple[np.ndarray, np.ndarray]

# 종이보다 이 비율 이하로 어두우면 먹으로 본다
_INK_DARKNESS = 0.8
# 종이 밝기가 이보다 어두우면(검은 바탕 페이지 등) 흰 홈통이 없다고 보고 나누지 않는다
_MIN_PAPER_LEVEL = 140
# 한 줄의 먹 픽셀이 줄 길이의 이 비율 이하면 빈 줄 — 스캔 잡티는 넘기고 컷 테두리는 막는다
_BLANK_INK_RATIO = 0.0025
# 홈통의 모든 줄에서 지운 말풍선·글자가 이 비율보다 많으면 또렷한 테두리가 있을 때만 홈통으로 본다
_MAX_ERASED_RATIO = 0.5
# 홈통 바로 옆 띠에서 (지운 자리를 뺀) 위치의 이 비율 이상에 먹이 닿으면 컷 테두리로 본다
_MIN_BORDER_RATIO = 0.5
_STRONG_BORDER_RATIO = 0.8
# 띠의 이 비율 이상이 지운 자리면 테두리 여부를 판단하지 않는다
_MIN_KNOWN_RATIO = 0.2
# 테두리를 찾아볼 띠의 폭 (홈통 바로 옆, 페이지 크기 대비)
_BORDER_SEARCH_RATIO = 0.003
# 홈통으로 인정할 최소 두께 (페이지 크기 대비)
_MIN_GUTTER_RATIO = 0.002
# 이보다 얇은 조각은 컷으로 보지 않고 옆 칸에 넘긴다 (페이지 크기 대비) — 쪽 번호 줄·그림 부스러기
_MIN_PIECE_RATIO = 0.03
# 말풍선·글자를 지울 때 테두리까지 덮도록 넓히는 폭 (페이지 폭 대비)
_ERASE_PAD_RATIO = 0.004
# 폭이 페이지의 이 비율보다 좁고 칸이 페이지 옆 가장자리에 붙은 컷은 잡지 광고·안내 기둥으로 보고
# 그 글자를 맨 뒤로 보낸다
_MARGIN_COLUMN_RATIO = 0.1
# 가로가 세로의 이 배수보다 길면 두 쪽 펼침면으로 본다
_SPREAD_ASPECT = 1.1
# 세로 홈통이 가로 홈통보다 이 배수 이상 넓을 때만 좌우를 먼저 나눈다 (4컷 만화의 두 줄 배치)
_COLUMN_FIRST_RATIO = 1.5
_MAX_DEPTH = 8


class Panel(NamedTuple):
    cell: Box  # 대사를 배정하는 칸 — 칸들은 페이지를 빈틈없이 나눈다
    box: Box  # 칸 안 컷의 먹 범위


def _masks(image_rgb: np.ndarray, erase_boxes: Sequence[Sequence[float]]) -> Optional[Masks]:
    """먹 마스크와 지운 자리 마스크 (종이가 너무 어두우면 None)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    paper = float(np.percentile(gray[::4, ::4], 95))
    if paper < _MIN_PAPER_LEVEL:
        return None
    ink = gray < paper * _INK_DARKNESS
    erased = np.zeros_like(ink)
    h, w = ink.shape
    pad = max(2, int(round(w * _ERASE_PAD_RATIO)))
    for box in erase_boxes:
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        erased[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = True
    ink &= ~erased
    return ink, erased


def _band(mask: np.ndarray, box: Box, rows: bool, start: int, end: int) -> np.ndarray:
    """box 안 start..end번째 가로줄(rows) 또는 세로줄 띠."""
    x1, y1, x2, y2 = box
    return mask[y1 + start:y1 + end, x1:x2] if rows else mask[y1:y2, x1 + start:x1 + end].T


def _line_sums(mask: np.ndarray, box: Box, rows: bool) -> Tuple[np.ndarray, int]:
    """box 안의 가로줄(rows) 또는 세로줄마다 mask 픽셀 수, 줄 길이."""
    band = _band(mask, box, rows, 0, (box[3] - box[1]) if rows else (box[2] - box[0]))
    return band.sum(axis=1), band.shape[1]


def _blank_lines(masks: Masks, box: Box, rows: bool) -> np.ndarray:
    """box 안의 가로줄(rows) 또는 세로줄마다 먹이 없는지."""
    counts, length = _line_sums(masks[0], box, rows)
    return counts <= max(1, int(length * _BLANK_INK_RATIO))


def _border(masks: Masks, box: Box, rows: bool, start: int, end: int) -> Optional[float]:
    """홈통 옆 띠(start..end번째 줄)가 컷 테두리처럼 먹으로 덮인 비율 — 지운 자리가 대부분이면 None.

    기울어진 테두리도 잡도록 띠의 어느 줄에든 먹이 닿은 위치를 (지운 자리를 빼고) 센다.
    """
    known = ~_band(masks[1], box, rows, start, end).any(axis=0)
    if known.sum() < _MIN_KNOWN_RATIO * known.size:
        return None
    touched = _band(masks[0], box, rows, start, end).any(axis=0) & known
    return float(touched.sum() / known.sum())


def _is_border(cover: Optional[float], ratio: float = _MIN_BORDER_RATIO) -> bool:
    return cover is not None and cover >= ratio


def _is_open(cover: Optional[float]) -> bool:
    """테두리 없이 열린 쪽 (판단할 수 있고 테두리가 아님)."""
    return cover is not None and cover < _MIN_BORDER_RATIO


def _trim(masks: Masks, box: Box) -> Optional[Box]:
    """box를 먹이 있는 범위로 좁힌다 (여백 제거)."""
    ys = np.flatnonzero(~_blank_lines(masks, box, rows=True))
    xs = np.flatnonzero(~_blank_lines(masks, box, rows=False))
    if ys.size == 0 or xs.size == 0:
        return None
    x1, y1 = box[0], box[1]
    return (x1 + int(xs[0]), y1 + int(ys[0]), x1 + int(xs[-1]) + 1, y1 + int(ys[-1]) + 1)


def _split(masks: Masks, cell: Box, box: Box, rows: bool,
           limits: Tuple[int, int, int]) -> Tuple[List[Tuple[Box, Box]], int]:
    """box를 홈통으로 나눠 (칸, 조각) 쌍들(위→아래 또는 왼쪽→오른쪽)과 조각 사이 가장 넓은 틈.

    limits = (최소 홈통 두께, 최소 조각 두께, 테두리를 찾아볼 폭). 칸은 cell을 빈틈없이 나누고
    조각은 칸 안에서 다시 나눠 볼 먹 범위다. 얇은 조각은 버리고 그 자리는 옆 칸에 넘긴다.
    """
    min_width, min_piece, reach = limits
    blank = _blank_lines(masks, box, rows)
    n = len(blank)
    edges = np.diff(np.concatenate(([0], blank.astype(np.int8), [0])))
    erased, length = _line_sums(masks[1], box, rows)
    gutters = {}  # 시작 → (끝, 앞쪽 테두리, 뒤쪽 테두리)
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        s, e = int(s), int(e)
        if e - s < min_width or s == 0 or e == n:
            continue
        before = _border(masks, box, rows, max(0, s - reach), s)
        after = _border(masks, box, rows, e, min(n, e + reach))
        if erased[s:e].min() > length * _MAX_ERASED_RATIO:
            # 쌓인 말풍선을 지운 자리가 만든 가짜일 수 있다 — 한쪽 테두리가 또렷해야 한다
            real = _is_border(before, _STRONG_BORDER_RATIO) or _is_border(after, _STRONG_BORDER_RATIO)
        else:
            # 컷 테두리 없이 흰 바탕에 그린 그림 사이의 틈이 아니어야 한다
            real = _is_border(before) or _is_border(after)
        if real:
            gutters[s] = (e, before, after)
    starts = [0] + [end for end, _, _ in gutters.values()]
    ends = list(gutters) + [n]
    pieces = [(a, b) for a, b in zip(starts, ends) if b - a >= min_piece]
    if not pieces:
        return [], 0
    ends_at = {end: after for end, _, after in gutters.values()}
    cuts = []
    for (_, a_end), (b_start, _) in zip(pieces, pieces[1:]):
        a_border, b_border = gutters[a_end][1], ends_at[b_start]
        if _is_border(a_border) and _is_open(b_border):
            cuts.append(a_end)  # 뒤 컷이 테두리 없이 열려 있으면 빈 곳은 뒤 컷 몫
        elif _is_border(b_border) and _is_open(a_border):
            cuts.append(b_start)  # 앞 컷이 열려 있으면 빈 곳은 앞 컷 몫
        else:
            cuts.append((a_end + b_start) // 2)
    offset = box[1] if rows else box[0]
    marks = [cell[1] if rows else cell[0]] + [offset + c for c in cuts] + [cell[3] if rows else cell[2]]
    parts = []
    for (a, b), lo, hi in zip(pieces, marks, marks[1:]):
        if rows:
            parts.append(((cell[0], lo, cell[2], hi), (box[0], offset + a, box[2], offset + b)))
        else:
            parts.append(((lo, cell[1], hi, cell[3]), (offset + a, box[1], offset + b, box[3])))
    widest = max((b[0] - a[1] for a, b in zip(pieces, pieces[1:])), default=0)
    return parts, widest


def _panels(masks: Masks, cell: Box, region: Box, limits: dict, depth: int = 0) -> List[Panel]:
    box = _trim(masks, region)
    if box is None:
        return []
    if depth >= _MAX_DEPTH:
        return [Panel(cell, box)]
    tiers, tier_gap = _split(masks, cell, box, True, limits["y"])
    columns, column_gap = _split(masks, cell, box, False, limits["x"])
    if len(tiers) > 1 and not (len(columns) > 1 and column_gap >= tier_gap * _COLUMN_FIRST_RATIO):
        parts = tiers  # 위 단부터
    elif len(columns) > 1:
        parts = columns[::-1]  # 오른쪽 컷부터
    elif tiers and tiers[0][1] != box:
        parts = tiers  # 얇은 조각만 떼어 냈다
    elif columns and columns[0][1] != box:
        parts = columns
    else:
        return [Panel(cell, box)]
    return [panel for part_cell, part in parts for panel in _panels(masks, part_cell, part, limits, depth + 1)]


def find_panels(image_rgb: np.ndarray, erase_boxes: Sequence[Sequence[float]] = ()) -> List[Panel]:
    """페이지를 컷들로 나눠 읽는 순서대로 돌려준다 (나눌 수 없으면 빈 목록)."""
    masks = _masks(image_rgb, erase_boxes)
    if masks is None:
        return []
    h, w = masks[0].shape

    def axis_limits(size: int) -> Tuple[int, int, int]:
        return (max(2, round(size * _MIN_GUTTER_RATIO)), round(size * _MIN_PIECE_RATIO),
                max(2, round(size * _BORDER_SEARCH_RATIO)))

    limits = {"x": axis_limits(w), "y": axis_limits(h)}
    if w > h * _SPREAD_ASPECT:
        # 두 쪽을 한 장에 담은 펼침면 — 가운데를 그림이 가로질러도 오른쪽 면부터 읽는다
        right, left = (w // 2, 0, w, h), (0, 0, w // 2, h)
        return _panels(masks, right, right, limits) + _panels(masks, left, left, limits)
    page = (0, 0, w, h)
    return _panels(masks, page, page, limits)


def _inside_ratio(box: Sequence[float], panel: Sequence[float]) -> float:
    """box 넓이 중 panel 안에 든 비율."""
    w = min(box[2], panel[2]) - max(box[0], panel[0])
    h = min(box[3], panel[3]) - max(box[1], panel[1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return max(0.0, w) * max(0.0, h) / area if area > 0 else 0.0


def panel_of(text_box: Sequence[float], panels: List[Panel]) -> int:
    """대사 중심이 든 칸, 칸 밖이면(페이지 밖 좌표) 중심에서 가장 가까운 컷."""
    cx, cy = (text_box[0] + text_box[2]) / 2, (text_box[1] + text_box[3]) / 2

    def distance(box: Box) -> float:
        dx = max(box[0] - cx, 0, cx - box[2])
        dy = max(box[1] - cy, 0, cy - box[3])
        return dx * dx + dy * dy

    for i, panel in enumerate(panels):
        x1, y1, x2, y2 = panel.cell
        if x1 <= cx < x2 and y1 <= cy < y2:
            return i
    return min(range(len(panels)), key=lambda i: distance(panels[i].box))


def _row_order(elements: List[TextElement]) -> List[TextElement]:
    """위에서부터 줄을 짓고 같은 줄은 오른쪽부터.

    줄 안의 어느 대사와 시작 높이 차이가 둘 중 짧은 쪽 높이의 절반(적어도 작은 쪽 글자 1.5개)보다
    작으면 같은 줄로 본다 — 나란한 대사는 오른쪽부터 읽고, 세로로 긴 대사 옆에서 한참 아래에
    시작하는 대사는 뒤로 미룬다. 짝마다 작은 쪽을 기준으로 삼아 큰 효과음 하나가 줄을 넓히지 않게 한다.
    """
    def close(a: TextElement, b: TextElement) -> bool:
        tolerance = max(0.5 * min(a.text_box[3] - a.text_box[1], b.text_box[3] - b.text_box[1]),
                        1.5 * min(a.font_size or 0, b.font_size or 0))
        return abs(a.text_box[1] - b.text_box[1]) < tolerance

    rows: List[List[TextElement]] = []
    for element in sorted(elements, key=lambda e: e.text_box[1]):
        if rows and any(close(member, element) for member in rows[-1]):
            rows[-1].append(element)
        else:
            rows.append([element])
    return [element for row in rows for element in sorted(row, key=lambda e: -e.text_box[2])]


def reading_order(page_data: PageData) -> List[TextElement]:
    """페이지 대사를 읽는 순서로 돌려준다 — 그림이 있으면 컷부터 나눈다."""
    elements = page_data.text_elements()
    if len(elements) < 2:
        return elements
    panels = []
    if page_data.image_rgb is not None:
        erase_boxes = [bubble.bubble_box for bubble in page_data.speech_bubbles]
        erase_boxes += [element.text_box for element in elements]
        panels = find_panels(page_data.image_rgb, erase_boxes)
    if len(panels) < 2:
        return _row_order(elements)
    width = page_data.image_rgb.shape[1]
    edge_columns = {
        i for i, p in enumerate(panels)
        if p.box[2] - p.box[0] < width * _MARGIN_COLUMN_RATIO and (p.cell[0] == 0 or p.cell[2] == width)
    }
    content = _union([p.box for p in panels])
    core = _union([p.box for i, p in enumerate(panels) if i not in edge_columns]) or content
    freeform = {id(element) for element in page_data.freeform_texts}
    by_panel = [[] for _ in panels]
    margin = []
    for element in elements:
        index = panel_of(element.text_box, panels)
        x1, y1, x2, y2 = element.text_box[:4]
        outside_core = not (core[0] <= (x1 + x2) / 2 <= core[2] and core[1] <= (y1 + y2) / 2 <= core[3])
        # 모든 컷 바깥 글자, 가장자리 광고·안내 기둥의 글자, 본문 컷 범위 밖에 중심이 있는
        # 말풍선 밖 글자(공지·발매일 안내·쪽 번호 등)는 맨 뒤로
        if (_inside_ratio(element.text_box, content) == 0 or index in edge_columns
                or (id(element) in freeform and outside_core)):
            margin.append(element)
        else:
            by_panel[index].append(element)
    return [element for members in by_panel for element in _row_order(members)] + _row_order(margin)


def _union(boxes: List[Box]) -> Optional[Box]:
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
