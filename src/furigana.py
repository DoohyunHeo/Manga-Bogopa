"""대사 조각에서 후리가나(읽기용 작은 글자)를 찾아 '본문(읽기)' 쌍으로 읽는다.

글자 읽기 모델은 후리가나를 빼고 본문만 읽는다. 그런데 인명 한자는 읽는 법이 여럿이라
번역기가 흔한 읽기로 짐작하면 이름이 틀린다. 그래서 후리가나를 따로 읽어
번역 요청에 '읽기'로 함께 싣는다.

- 세로쓰기: 본문 열 오른쪽에 붙은 좁은 열 / 가로쓰기: 본문 줄 위에 붙은 좁은 줄
- 후리가나 덩어리마다 옆 본문 글자 칸을 골라 둘 다 manga-ocr로 읽는다 (조각 여러 개를 한꺼번에)
- 읽기가 가나만이고 본문이 한자(또는 로마자 약어 — 약어 위 가타카나는 뜻풀이 말장난이 많다)일 때만 남긴다
  — 느낌표·말줄임표·테두리 선 같은 좁은 조각은 여기서 걸러진다
"""
import logging
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.glyph_metrics import _band_score, _ink_mask, _refined_bands

logger = logging.getLogger(__name__)

_KANA_READING = re.compile(r"^[ぁ-ゖァ-ヺー]{1,12}$")
_BASE_TEXT = re.compile(r"^[㐀-䶿一-鿿豈-﫿々〆ヶ]{1,8}$")
_BASE_KANA_TAIL = re.compile(r"(?<=[㐀-䶿一-鿿豈-﫿々〆ヶ])[ぁ-ゖ]+$")
# 로마자 약어·숫자에 붙은 읽기 (약어 위 가타카나 뜻풀이) — 글자 수보다 읽기가 훨씬 길다
_LATIN_BASE = re.compile(r"^[A-Za-zＡ-Ｚａ-ｚ0-9０-９]{1,6}$")

_MAIN_BAND_RATIO = 0.7     # 본문 열(줄): 가장 굵은 열의 이 비율 이상
_RUBY_MAX_RATIO = 0.62     # 후리가나 열 굵기 상한 (본문 글자 크기 대비)
_RUBY_MIN_RATIO = 0.2
_RUBY_MAX_GAP = 0.45       # 본문 열과 후리가나 열 사이 틈 상한 (본문 글자 크기 대비)
_RUBY_SPLIT_GAP = 0.6      # 후리가나 글자 크기의 이만큼 비면 다른 낱말의 읽기
_RUBY_JOIN_GAP = 1.2       # 이보다 가까이 이어진 작은 조각은 앞 읽기의 나머지 (획이 가늘고 띄엄띄엄한 가나)
_BASE_REACH = 0.3          # 후리가나 덩어리 양 끝에서 본문 글자 칸을 고를 때 더 보는 폭 (본문 글자 크기 대비)
# 정사각형에 가까운 조각의 방향 — 굵은 띠 사이 틈(띠 굵기 대비)이 한 축은 줄 사이만큼(이상), 다른 축은 글자 사이만큼(미만)
_LINE_GAP_MIN = 0.25
_CHAR_GAP_MAX = 0.2
_MIN_CROP = 24


Box = Tuple[int, int, int, int]


def _runs(profile: np.ndarray, gap: float) -> List[Tuple[int, int]]:
    """잉크가 있는 구간들 — gap보다 짧은 빈칸은 이어 붙인다."""
    runs, start, last = [], None, None
    for i, value in enumerate(profile):
        if value <= 0:
            continue
        if start is None:
            start = i
        elif i - last - 1 >= gap:
            runs.append((start, last + 1))
            start = i
        last = i
    if start is not None:
        runs.append((start, last + 1))
    return runs


def _cells(profile: np.ndarray, em: float) -> List[Tuple[int, int]]:
    """본문 열의 글자 칸 — 빈칸으로 나뉜 조각을 한 글자 크기(em)까지 묶는다 (三처럼 획 사이가 빈 글자)."""
    cells = []
    for start, end in _runs(profile, 1):
        if cells and end - cells[-1][0] <= 1.15 * em:
            cells[-1] = (cells[-1][0], end)
        else:
            cells.append((start, end))
    return cells


def _pairs_along(ink: np.ndarray, ruby_after: bool) -> List[Tuple[Box, Box]]:
    """열(세로쓰기 기준) 방향으로 후리가나 덩어리와 옆 본문 글자 칸 쌍을 찾는다. 상자는 (x1, y1, x2, y2).

    ruby_after: 후리가나가 본문보다 뒤(오른쪽) 좌표에 있는지 — 세로쓰기 True, 가로쓰기를 뒤집어 넣으면 False.
    """
    h, _ = ink.shape
    profile = ink.sum(axis=0) / 255.0
    bands = _refined_bands(profile, h)
    if len(bands) < 2:
        return []
    widest = max(end - start for start, end in bands)
    mains = [(s, e) for s, e in bands if e - s >= widest * _MAIN_BAND_RATIO]
    em = float(np.median([e - s for s, e in mains]))
    pairs = []
    for start, end in bands:
        width = end - start
        if not (_RUBY_MIN_RATIO * em <= width <= _RUBY_MAX_RATIO * em):
            continue
        if ruby_after:
            neighbours = [(s, e) for s, e in mains if e <= start and start - e <= _RUBY_MAX_GAP * em]
            base = max(neighbours, key=lambda band: band[1]) if neighbours else None
        else:
            neighbours = [(s, e) for s, e in mains if s >= end and s - end <= _RUBY_MAX_GAP * em]
            base = min(neighbours, key=lambda band: band[0]) if neighbours else None
        if base is None:
            continue
        cells = _cells(ink[:, base[0]:base[1]].sum(axis=1), em)
        groups = []  # [후리가나 위, 아래, 첫 칸, 끝 칸]
        for top, bottom in _runs(ink[:, start:end].sum(axis=1), _RUBY_SPLIT_GAP * width):
            reach = _BASE_REACH * em
            chosen = [i for i, (a, b) in enumerate(cells) if top - reach <= (a + b) / 2 <= bottom + reach]
            if groups and top - groups[-1][1] <= _RUBY_JOIN_GAP * width and (
                    not chosen or bottom - top < 0.5 * width):
                # 앞 읽기에 바짝 붙은 작은 조각·본문 칸이 없는 조각은 앞 읽기의 나머지 글자다
                groups[-1][1] = bottom
                if chosen:
                    groups[-1][3] = max(groups[-1][3], chosen[-1])
                continue
            if bottom - top < 0.5 * width:
                continue  # 점 하나 크기 — 후리가나가 아니다
            if not chosen:
                continue
            if groups and chosen[0] <= groups[-1][3] + 1:
                # 이어진 한자 낱말에 글자마다 따로 붙은 읽기는 한 낱말로 묶는다 — 가나 한두 자만 읽으면 틀리기 쉽다
                groups[-1] = [groups[-1][0], bottom, groups[-1][2], max(groups[-1][3], chosen[-1])]
            else:
                groups.append([top, bottom, chosen[0], chosen[-1]])
        for top, bottom, first, last in groups:
            pairs.append(((start, top, end, bottom), (base[0], cells[first][0], base[1], cells[last][1])))
    return pairs


def find_ruby(crop: Image.Image) -> List[Tuple[Box, Box, bool]]:
    """조각 안의 (후리가나 상자, 본문 상자, 세로쓰기인지) 목록 — 조각 좌표."""
    gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    if h < _MIN_CROP or w < _MIN_CROP:
        return []
    ink = _ink_mask(gray)
    if h > w * 1.6:
        vertical = True
    elif w > h * 1.6:
        vertical = False
    else:
        columns, rows = ink.sum(axis=0) / 255.0, ink.sum(axis=1) / 255.0
        column_gap, row_gap = _line_gap(columns, h), _line_gap(rows, w)
        # 한 축의 띠 사이가 줄 사이만큼 비고 다른 축은 글자끼리 거의 붙어 있으면 앞의 축이 단(줄) 사이다 — 방향 점수는
        # 칸이 가지런한 세로 단의 글자 칸(고른 가로 띠)을 가로줄로 보아 세로 대사의 후리가나를 통째로 놓칠 수 있다.
        # 글자가 빽빽해 띠가 뭉친 축은 틈이 들쭉날쭉 커서 줄 사이로 믿지 않고 방향 점수에 맡긴다
        if column_gap >= _LINE_GAP_MIN and row_gap < _CHAR_GAP_MAX:
            vertical = True
        elif row_gap >= _LINE_GAP_MIN and column_gap < _CHAR_GAP_MAX:
            vertical = False
        else:
            vertical = _band_score(columns, h) >= _band_score(rows, w)
    if vertical:
        return [(ruby, base, True) for ruby, base in _pairs_along(ink, ruby_after=True)]
    # 가로쓰기: 뒤집어 같은 방법으로 찾고 상자 좌표를 되돌린다 (후리가나는 본문 줄 위 = 작은 좌표)
    swap = lambda box: (box[1], box[0], box[3], box[2])  # noqa: E731
    return [(swap(ruby), swap(base), False) for ruby, base in _pairs_along(ink.T, ruby_after=False)]


def _line_gap(profile: np.ndarray, cross_len: int) -> float:
    """굵은 띠(본문 단·줄) 사이 틈의 중앙 ÷ 띠 굵기의 중앙 — 띠가 하나뿐이면 0."""
    bands = _refined_bands(profile, cross_len)
    if not bands:
        return 0.0
    widest = max(end - start for start, end in bands)
    mains = [(start, end) for start, end in bands if end - start >= widest * _MAIN_BAND_RATIO]
    if len(mains) < 2:
        return 0.0
    gaps = [b[0] - a[1] for a, b in zip(mains, mains[1:])]
    return float(np.median(gaps)) / float(np.median([end - start for start, end in mains]))


def _crop(image: Image.Image, box: Box, pad: int) -> Image.Image:
    x1, y1, x2, y2 = box
    return image.crop((max(0, x1 - pad), max(0, y1 - pad), min(image.width, x2 + pad), min(image.height, y2 + pad)))


def _closest_window(main: str, base: str) -> Optional[str]:
    """본문 판독(main)에서 base와 같은 길이에 한 글자만 다른 곳 — 딱 한 곳일 때만 (없거나 여럿이면 None)."""
    size = len(base)
    found = {main[i:i + size] for i in range(len(main) - size + 1)
             if sum(a != b for a, b in zip(main[i:i + size], base)) <= 1}
    return found.pop() if len(found) == 1 else None


def read_furigana(ocr, crops: Sequence[Image.Image],
                  main_texts: Optional[Sequence[str]] = None) -> List[Optional[List[str]]]:
    """조각마다 '본문(읽기)' 목록 (없으면 None). ocr: manga-ocr처럼 PIL 목록을 받아 판독 목록을 돌려주는 모델.

    main_texts는 조각마다 본문 전체의 판독 — 주면 본문 칸만 따로 잘라 읽다 틀린 한자(二十歳 → 三十歳)를 본문 판독에서 같은
    길이에 한 글자만 다른 곳으로 맞추고, 그런 곳이 없으면 그 쌍을 버린다 — 틀린 한자가 읽기와 함께 번역 요청에 실리면
    번역도 틀린다.
    """
    jobs = []  # (조각 번호, 후리가나 그림, 본문 그림)
    for index, crop in enumerate(crops):
        try:
            found = find_ruby(crop)
        except Exception as exc:  # noqa: BLE001 — 후리가나는 곁들이는 정보, 못 찾으면 없는 셈 친다
            logger.debug(f"후리가나 찾기 실패: {exc}")
            continue
        for ruby, base, _ in found:
            pad = max(2, (ruby[2] - ruby[0]) // 4)
            jobs.append((index, _crop(crop, ruby, pad), _crop(crop, base, pad)))
    results: List[Optional[List[str]]] = [None] * len(crops)
    if not jobs:
        return results
    texts = list(ocr([image for _, ruby, base in jobs for image in (ruby, base)]))
    for n, (index, _, _) in enumerate(jobs):
        reading, base = texts[2 * n].strip(), texts[2 * n + 1].strip()
        # 본문은 한자(또는 로마자 약어)만, 읽기는 가나 두 자 이상에 본문 글자 수와 맞는 길이일 때만 — 가나 본문 열을
        # 후리가나로 잘못 본 경우와 가나 한 자만 읽힌(대개 틀린) 경우를 뺀다
        if not _KANA_READING.match(reading) or len(reading) < 2:
            continue
        # 한자 뒤에 딸려 온 히라가나(送り仮名·조사)는 뗀다 — 읽기가 단 끝까지 이어지면 조사까지 본문으로 잘려 한자만이라는
        # 검사에 걸린다
        base = _BASE_KANA_TAIL.sub("", base)
        main = (main_texts[index] if main_texts is not None and index < len(main_texts) else None) or ""
        if main and base and base not in main:
            base = _closest_window(main, base) or ""
        if _BASE_TEXT.match(base):
            if not len(base) <= len(reading) <= 4 * len(base) + 1:
                continue
        elif not (_LATIN_BASE.match(base) and len(reading) >= 2 * len(base)):
            continue
        pair = f"{base}({reading})"
        if results[index] is None:
            results[index] = []
        if pair not in results[index]:
            results[index].append(pair)
    return results
