"""잉크 기하 측정으로 원문 글자의 크기와 기울기를 추정합니다.

원문 글자 크기·기울기는 이 측정으로만 낸다 (모델은 글씨체 분류에만 쓴다).

크기 추정 구성:
- OTSU 이진화 + 극성 자동 반전 (흰 글자/검은 배경 대응)
- 축 선택: 종횡비가 애매하면 양축 밴드 구조 점수로 결정
- 투영 밴드: 거친 밴드를 피크의 40%로 재절단해 후리가나·병합 줄 분리
  (나눔선이 글자를 가로지르면 글자 안 획 틈이라 나누지 않는다)
- 연결 성분 높이 p70 (성분 3개 이상일 때만 신뢰)
- 글자 1~3개짜리 크롭은 가까운 조각을 한 글자로 묶어 가장 큰 글자의 긴 변
  (글자 인식이 1~3자로 읽었으면 바탕 그림 부스러기는 세지 않는다)
- 말풍선 대사는 글자 인식이 읽은 막대 부호(장음·줄표·물결·느낌표·세로줄) 수만큼 가장 길쭉한 조각을
  크기 근거에서 빼고, 세로가 가로보다 긴 크롭은 세로쓰기로 본다
- 줄/컬럼 피치와 전각 정사각 성질을 이용한 교차 검증 (상향 전용)
- 흰 바탕 위 테두리 두른 흰 글자(袋文字)는 테두리 안 흰 몸통으로 잰다 (붙은 테두리 덩어리는 2~3배로 나온다)

주의: 자기상관(주기성) 기반 피치 추정은 글자 주기가 아닌 획 간격을
잡는 경향이 있어 사용하지 않는다.
"""
import math
import unicodedata

import cv2
import numpy as np


def count_glyphs(text):
    """글자 인식 결과에서 문장 부호·기호를 뺀 글자 수 (estimate_glyph_height의 glyphs)."""
    return sum(1 for ch in text or "" if unicodedata.category(ch)[0] in "LN")


def _ink_mask(gray):
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if ink.mean() > 140:  # 영역 대부분이 잉크 판정 → 흰 글자/어두운 배경 → 극성 반전
        ink = 255 - ink
    return ink


def _coarse_bands(profile, threshold, merge_gap=2, min_size=3):
    flags = profile > threshold
    bands, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            bands.append([start, i])
            start = None
    if start is not None:
        bands.append([start, len(flags)])

    merged = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = band[1]
        else:
            merged.append(band)
    return [(a, b) for a, b in merged if b - a >= min_size]


# 재절단 나눔선을 가로지르는 글자 조각의 잉크가 이 비율 이상이면 글자 안 획 틈으로 보고 나누지 않는다
_SPLIT_CROSSING_SHARE = 0.35


def _split_cuts_through_glyphs(ink, axis, start, end, subs):
    """밴드 [start, end)를 subs로 나누는 선이 글자를 가로지르는지.

    후리가나 단·붙은 두 줄은 서로 다른 글자 조각이라 나눔선을 넘지 않지만, 'お'의 몸통과 점·'学校'의 위아래
    획 틈에서 나누면 같은 글자 조각이 선을 넘는다. 읽는 방향으로 밴드 폭의 2.5배 넘게 이어진 덩어리(바탕·선)는
    글자가 아니라 세지 않고, 그런 덩어리가 잉크 대부분이면 판단하지 않는다(나눈다).
    axis는 투영이 따라가는 축 — 'x'면 세로 단(열 투영), 'y'면 가로 줄(행 투영).
    """
    region = np.ascontiguousarray(ink[:, start:end] if axis == "x" else ink[start:end, :])
    num, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    cuts = [(subs[k][1] + subs[k + 1][0]) / 2 for k in range(len(subs) - 1)]
    counted = crossing = total = 0
    for i in range(1, num):
        x, y, comp_w, comp_h, area = (int(v) for v in stats[i])
        total += area
        if (comp_h if axis == "x" else comp_w) > 2.5 * (end - start):
            continue
        lo, hi = (x, x + comp_w) if axis == "x" else (y, y + comp_h)
        counted += area
        if any(lo < cut - 1 and hi > cut + 1 for cut in cuts):
            crossing += area
    if counted <= 0 or counted < 0.3 * total:
        return False
    return crossing >= _SPLIT_CROSSING_SHARE * counted


def _refined_bands(profile, cross_len, ink=None, axis=None):
    """거친 밴드 내부를 자기 피크 40%로 재절단 (후리가나·병합 줄 분리).

    글자 내부 획 틈으로 인한 과분할(최대 하위밴드 < 원래의 42%)이면 원래 밴드 유지.
    ink·axis가 오면 나눔선이 글자를 가로지르는 재절단도 하지 않는다 (_split_cuts_through_glyphs).
    """
    out = []
    for a, b in _coarse_bands(profile, max(2.0, 0.03 * cross_len)):
        segment = profile[a:b]
        sub = _coarse_bands(segment, max(2.0, 0.40 * segment.max()), merge_gap=1)
        if len(sub) >= 2 and max(e - s for s, e in sub) >= 0.42 * (b - a) and not (
                ink is not None and _split_cuts_through_glyphs(ink, axis, a, b, sub)):
            out.extend((a + s, a + e) for s, e in sub)
        else:
            out.append((a, b))
    return out


def _band_estimate(profile, cross_len, ink=None, axis=None):
    sizes = [e - s for s, e in _refined_bands(profile, cross_len, ink, axis)]
    if not sizes:
        return 0.0, 0
    main = max(sizes)
    majors = [s for s in sizes if s >= main * 0.55]  # 후리가나 줄/컬럼 제외
    return float(np.median(majors)), len(majors)


def _band_score(profile, cross_len):
    """이 축이 '텍스트 줄/컬럼' 축일 그럴듯함: 밴드 수 × 두께 균일성."""
    sizes = [e - s for s, e in _refined_bands(profile, cross_len)]
    if not sizes:
        return -1.0
    main = max(sizes)
    majors = [s for s in sizes if s >= main * 0.55]
    if not majors:
        return -1.0
    uniformity = 1.0 - min(1.0, float(np.std(majors)) / max(np.mean(majors), 1e-6))
    return len(majors) * (0.5 + 0.5 * uniformity)


_RUBY_PIECE_RATIO = 1.6  # 주요 밴드 안 조각 높이 중앙이 밖의 이 배 이상이면 밖은 후리가나

# 막대 부호는 한 방향으로 길쭉한 조각이라 글자 조각 높이로 세면 글자 크기를 틀리게 잰다 — 긴 변이 짧은 변의 이 배
# 이상인 조각만 막대 부호로 본다
_BAR_ASPECT = 3.0


def _bar_pieces(sizes, bars):
    """sizes((폭, 높이) 목록)에서 막대 부호로 보고 뺄 번호 — 가장 길쭉한 것부터 bars개 가운데 _BAR_ASPECT배 이상인 것."""
    if bars <= 0:
        return set()
    aspect = [max(s) / max(1, min(s)) for s in sizes]
    ranked = sorted(range(len(sizes)), key=lambda i: -aspect[i])
    return {i for i in ranked[:bars] if aspect[i] >= _BAR_ASPECT}


def _component_heights(ink, h, w, bands=None, vertical=False, inside=True, bars=0):
    """글자 조각 높이 목록. bands(선택 축의 (시작, 끝) 목록)를 주면 가운데가 그 안에 있는(inside=False면 밖에 있는) 조각만.
    bars개까지 막대 부호 조각은 뺀다 (_bar_pieces)."""
    num, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    pieces = []
    # 최소 높이: 작은 크롭에선 상대(10%) 노이즈 컷, 큰 크롭(다줄 박스)에선
    # 12px로 캡 (큰 박스에서 실제 글자까지 걸러지는 것 방지).
    min_h = max(5.0, min(h * 0.10, 12.0))
    for i in range(1, num):
        x, y, comp_w, comp_h, area = stats[i]
        if area < 16 or comp_h < min_h or comp_h > h * 0.95:
            continue
        if comp_w > w * 0.95 or comp_h > comp_w * 8:
            continue
        if bands is not None:
            center = x + comp_w / 2 if vertical else y + comp_h / 2
            if any(a <= center < b for a, b in bands) != inside:
                continue
        pieces.append((comp_w, comp_h))
    dropped = _bar_pieces(pieces, bars)
    return [comp_h for i, (_, comp_h) in enumerate(pieces) if i not in dropped]


def _major_bands(profile, cross_len):
    """재절단 밴드 중 주요(최대의 55% 이상) 밴드 목록·두께 중앙값·중심 피치."""
    bands = _refined_bands(profile, cross_len)
    if not bands:
        return [], 0.0, 0.0
    mx = max(e - s for s, e in bands)
    majors = [(s, e) for s, e in bands if (e - s) >= mx * 0.55]
    med = float(np.median([e - s for s, e in majors]))
    centers = [(s + e) / 2 for s, e in majors]
    pitch = float(np.median(np.diff(centers))) if len(centers) >= 2 else 0.0
    return majors, med, pitch


def _line_pitch_estimate(ink, major_bands, vertical):
    """줄/컬럼별 글자 셀 피치(≈전각 em)를 재고, 줄 간 일치할 때만 신뢰합니다.

    한자와 가나는 잉크 크기가 달라도 차지하는 전각 칸은 같으므로, 읽기 방향의
    글자 중심 간격이 잉크 높이보다 본질적인 크기 신호다 (가나 위주 문장에서
    잉크 기반 추정이 과소평가하는 것을 보정).

    Returns (pitch, strong): strong=True는 2개 이상 줄에서 교차 일치한 경우.
    """
    line_ems = []
    for a, b in major_bands:
        strip = ink[:, a:b] if vertical else ink[a:b, :]
        # 읽기축 투영: 세로쓰기 컬럼이면 행 방향, 가로쓰기 줄이면 열 방향
        profile = (strip.sum(axis=1) if vertical else strip.sum(axis=0)) / 255.0
        thickness = max(1, b - a)
        cells = _coarse_bands(profile, max(1.0, 0.10 * thickness), merge_gap=1, min_size=3)
        if len(cells) < 3:
            continue
        centers = np.array([(s + e) / 2 for s, e in cells])
        advances = np.diff(centers)
        median_adv = float(np.median(advances))
        if median_adv <= 4:
            continue
        # 균일성: 간격 60% 이상이 중앙값 ±25% 안에 있어야 (부분 병합/분할 배제)
        uniform = float(np.mean(np.abs(advances - median_adv) <= median_adv * 0.25))
        if uniform < 0.6:
            continue
        # 간격 지배 게이트: 셀 사이 빈 간격이 피치의 45%를 넘으면 글자 간격이 아니라
        # 줄 간격(다단 격자에서 축이 섞인 경우)이나 어절 띄어쓰기를 잡은 것이다.
        gaps = [cells[i + 1][0] - cells[i][1] for i in range(len(cells) - 1)]
        if gaps and float(np.median(gaps)) > median_adv * 0.45:
            continue
        line_ems.append(median_adv)

    if len(line_ems) >= 2:
        # 줄 간 교차 일치: 최대/최소가 1.3배 이내일 때만 신뢰 (서로 다른 두 값이
        # std/mean 검사를 우연히 통과하는 것 방지)
        if max(line_ems) / max(min(line_ems), 1e-6) <= 1.3:
            return float(np.median(line_ems)), True
        return 0.0, False
    if len(line_ems) == 1:
        return line_ems[0], False
    return 0.0, False


def _flat_glyph_estimate(ink, h, w):
    """납작 글자(ヘ·ー·一·つ 등) 지배 크롭의 글자 크기를 폭·피치로 추정합니다.

    납작 글자는 잉크 높이가 em의 ~30%뿐이라 밴드/성분 높이가 크기를 심하게
    과소평가한다. CJK는 정사각 em이므로 이때는 ① 납작 성분의 '폭'과
    ② 글자들이 쌓인 '피치(중심 간격)'가 올바른 크기 신호다.
    납작 성분이 다수가 아니면 0을 반환해 기존 경로를 따른다.
    """
    num, _, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps = []
    for i in range(1, num):
        _, _, comp_w, comp_h, area = stats[i]
        if area < 16 or comp_h < 2 or comp_w < 4:
            continue
        if comp_w > w * 0.97 and comp_h > h * 0.97:
            continue
        comps.append((comp_w, comp_h, float(cents[i][0]), float(cents[i][1])))
    if not comps or len(comps) > 8:
        return 0.0

    flats = [c for c in comps if c[0] >= c[1] * 1.8]
    if len(flats) * 2 < len(comps):
        return 0.0

    width_estimate = float(np.median([c[0] for c in flats]))
    estimate = width_estimate

    # 세로 스택 피치: 성분들이 한 컬럼에 쌓여 있으면 중심 y 간격 ≈ em + 행간
    if len(comps) >= 2:
        comps_sorted = sorted(comps, key=lambda c: c[3])
        xs = [c[2] for c in comps_sorted]
        if float(np.std(xs)) < width_estimate * 0.35:
            dys = np.diff([c[3] for c in comps_sorted])
            if len(dys) and (dys > 2).all():
                pitch = float(np.median(dys))
                estimate = max(estimate, pitch * 0.88)

    return float(min(estimate, max(h, w)))


def _deskew(ink):
    """±14° 탐색으로 행 투영 분산(줄 선명도)을 최대화하는 각도로 회전."""
    h, w = ink.shape
    small = cv2.resize(ink, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA)
    best_angle, best_var = 0, -1.0
    for angle in range(-14, 15, 2):
        matrix = cv2.getRotationMatrix2D((small.shape[1] / 2, small.shape[0] / 2), angle, 1.0)
        rotated = cv2.warpAffine(small, matrix, (small.shape[1], small.shape[0]))
        var = float(np.var(rotated.sum(axis=1)))
        if var > best_var:
            best_var, best_angle = var, angle
    if best_angle == 0:
        return ink
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
    return cv2.warpAffine(ink, matrix, (w, h))


# 기울기 측정 문턱
_TILT_SEARCH_LIMIT = 30        # 찾는 범위 ±도 (2도 간격으로 찾고 0.5도로 다듬는다)
_TILT_MAX = 28.0               # 이보다 크면 글이 아니라 그림·효과음의 선을 잡은 것으로 본다
_TILT_MIN_GAIN = 1.08          # 0도 대비 투영 선명도 이득
_TILT_GLYPH_MIN_GAIN = 1.05    # 글자 조각만으로 잴 때의 이득 (획 방향도 같은 쪽이어야 한다)
_TILT_GLYPH_MIN_PIECES = 6
_STROKE_AGREE = (0.25, 2.5)    # 획 기울기 / 줄 기울기 — 획 방향은 굽은 획·사선 획 때문에 실제보다 작게 나온다
_STROKE_MIN_CONCENTRATION = 0.10


def _glyph_ink(gray):
    """바탕·검은 띠·칸 테두리를 뺀 글자 잉크(0/255)와 글자 조각 수. 글자가 없으면 (None, 0).

    크롭 테두리를 차지한 쪽을 바탕으로 보고, 반대쪽에서 글자가 안 나오면(흰 여백 안 검은 띠 위의 흰 글자 등)
    반대 극성으로 다시 본다.
    """
    h, w = gray.shape
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    border = np.concatenate([dark[0], dark[-1], dark[:, 0], dark[:, -1]])
    first = dark if border.mean() < 127 else 255 - dark
    min_area = max(12.0, 0.0004 * h * w)
    for mask in (first, 255 - first):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = []
        for i in range(1, num):
            x, y, comp_w, comp_h, area = (int(v) for v in stats[i])
            if area < min_area:
                continue
            touches = int(x <= 0) + int(y <= 0) + int(x + comp_w >= w) + int(y + comp_h >= h)
            if touches >= 2 and (comp_w > 0.5 * w or comp_h > 0.5 * h):
                continue  # 크롭 테두리에 걸친 큰 덩어리 = 바탕·칸 테두리
            if comp_w >= 0.92 * w and comp_h >= 0.92 * h:
                continue
            if area > 0.1 * h * w and area > 0.8 * comp_w * comp_h:
                continue  # 꽉 찬 큰 덩어리 = 검은 띠·상자 (글자는 이만큼 꽉 차지 않는다)
            keep.append(i)
        if keep:
            glyphs = np.isin(labels, keep)
            if glyphs.sum() >= max(40, 0.01 * h * w):
                return (glyphs * 255).astype(np.uint8), len(keep)
    return None, 0


def _projection_tilt(ink):
    """후보 각도로 돌렸을 때 행·열 투영이 가장 선명해지는(줄·단이 가로·세로로 선) 각도의 부호 반전.

    (기울기(렌더러 규약), 0도 대비 선명도 이득).
    """
    h, w = ink.shape
    pad = int(0.3 * max(h, w))
    padded = cv2.copyMakeBorder(ink, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    if max(padded.shape) > 320:
        scale = 320 / max(padded.shape)
        padded = cv2.resize(
            padded, (int(padded.shape[1] * scale), int(padded.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ph, pw = padded.shape

    def _sharpness(angle):
        matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        rotated = cv2.warpAffine(padded, matrix, (pw, ph))
        return float(np.var(rotated.sum(axis=1)) + np.var(rotated.sum(axis=0)))

    base = _sharpness(0.0)
    best_angle, best_value = 0.0, base
    for angle in range(-_TILT_SEARCH_LIMIT, _TILT_SEARCH_LIMIT + 1, 2):
        if angle == 0:
            continue
        value = _sharpness(float(angle))
        if value > best_value:
            best_angle, best_value = float(angle), value
    for angle in np.arange(best_angle - 2, best_angle + 2.01, 0.5):
        value = _sharpness(float(angle))
        if value > best_value:
            best_angle, best_value = float(angle), value
    return -best_angle, best_value / max(base, 1e-6)


def _stroke_tilt(gray, glyphs):
    """글자 조각마다 획 방향(가로·세로 획을 90도로 접은 평균)을 재 조각 크기로 가중 평균한다.

    (기울기(렌더러 규약), 집중도 0~1). 글자 자체가 돌아갔는지 보는 신호라 값보다 줄 기울기와 같은 쪽인지를 본다.
    칸 테두리·집중선처럼 긴 선은 뺀다.
    """
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(glyphs, connectivity=8)
    if num <= 1:
        return 0.0, 0.0
    typical = float(np.median([max(stats[i][2], stats[i][3]) for i in range(1, num)]))
    kernel = np.ones((3, 3), np.uint8)
    total, weight = 0j, 0.0
    for i in range(1, num):
        x, y, comp_w, comp_h, area = (int(v) for v in stats[i])
        if max(comp_w, comp_h) > 3 * typical and area < 0.25 * comp_w * comp_h:
            continue
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(glyphs.shape[1], x + comp_w + 2), min(glyphs.shape[0], y + comp_h + 2)
        edge = cv2.dilate((labels[y0:y1, x0:x1] == i).astype(np.uint8), kernel) > 0
        mag = magnitude[y0:y1, x0:x1][edge]
        if mag.size < 8:
            continue
        strong = mag > np.percentile(mag, 50)
        theta = np.arctan2(gy[y0:y1, x0:x1][edge][strong], gx[y0:y1, x0:x1][edge][strong])
        mag = mag[strong]
        total += math.sqrt(area) * np.sum(mag * np.exp(4j * theta)) / max(float(np.sum(mag)), 1e-6)
        weight += math.sqrt(area)
    if weight <= 0:
        return 0.0, 0.0
    mean = total / weight
    # 영상 좌표(y 아래)의 각도 → 렌더러 규약(+ = 반시계)
    return float(-np.degrees(np.angle(mean) / 4.0)), float(abs(mean))


def _is_column(glyphs, tilt):
    """기울기만큼 되돌린 글자 덩어리가 세로로 길면 세로쓰기."""
    ys, xs = np.nonzero(glyphs)
    a = math.radians(tilt)
    turned_x = xs * math.cos(a) - ys * math.sin(a)
    turned_y = xs * math.sin(a) + ys * math.cos(a)
    return (turned_y.max() - turned_y.min() + 1) > (turned_x.max() - turned_x.min() + 1) * 1.15


def estimate_text_angle(crop_pil) -> float:
    """원문 글의 기울기(°, 렌더러 규약: 양수 = 반시계)를 잰다. 기울지 않았거나 잴 수 없으면 0.

    - 줄 기울기: 행·열 투영이 가장 선명해지는 각도. 0도보다 8% 넘게 선명해져야 믿는다. 잉크 조각이
      6개 미만이면(흰 테두리로 글자가 이어진 글 등) 바탕을 뺀 글자 조각으로 다시 재고, 획 방향도 같아야 믿는다.
    - 세로쓰기인데 글자 획은 똑바르고 줄만 비스듬하면(손글씨) 0 — 한국어는 가로로 쓰므로 그 비스듬함을
      따라 기울이면 원문에 없던 기울기가 된다. 가로쓰기는 줄 기울기를 따른다 (비스듬한 간판·올라가는 대사).
    - 28도를 넘으면 글이 아니라 그림·효과음의 선을 잡은 것으로 보고 0.
    글자 1~2개짜리 크롭은 기준선이 없어 대개 0이 된다.
    """
    gray = np.array(crop_pil.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    glyphs, pieces = _glyph_ink(gray)
    if glyphs is None:
        return 0.0
    ink = _ink_mask(gray)
    num, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # 글자 1~2개(블롭 몇 개)는 어느 각도로 돌려도 '정렬'될 수 있어 선명도 문턱이 무력하다
    enough_ink = h >= 24 and w >= 24 and sum(1 for i in range(1, num) if stats[i][4] >= 16) >= 6
    tilt, gain = _projection_tilt(ink if enough_ink else glyphs)
    stroke, concentration = _stroke_tilt(gray, glyphs)
    ratio = stroke / tilt if abs(tilt) > 1e-6 else 0.0
    strokes_agree = (_STROKE_AGREE[0] <= ratio <= _STROKE_AGREE[1]
                     and concentration >= _STROKE_MIN_CONCENTRATION)
    if enough_ink:
        if gain < _TILT_MIN_GAIN:
            return 0.0
    elif pieces < _TILT_GLYPH_MIN_PIECES or gain < _TILT_GLYPH_MIN_GAIN or not strokes_agree:
        return 0.0
    if abs(tilt) > _TILT_MAX:
        return 0.0
    if abs(tilt) > 2 and not strokes_agree and _is_column(glyphs, tilt):
        return 0.0
    return tilt


# 글자 1~3개짜리 크롭 — 조각 수·글자 수 상한, 조각을 묶는 거리(가장 큰 조각의 비율), 여러 조각 덩어리의 가로세로 상한
_FEW_MAX_PIECES = 5
_FEW_MAX_GLYPHS = 3
_FEW_MERGE_RATIO = 0.08
_FEW_MAX_ASPECT = 1.8
# 글자 인식이 1~3자로 읽은 크롭은 조각이 많아도 가장 큰 조각 넓이의 이 비율 이상만 글자 조각으로 센다 — 바탕 그림
# 부스러기가 많으면 큰 글자를 부스러기 크기로 잴 수 있다. 글자 수를 모르면 쓰지 않는다: 테두리로 이어진 여러 글자·바탕
# 덩어리를 글자 하나로 봐 몇 배로 잴 수 있다
_FEW_CLUTTER_AREA = 0.1


def _few_glyph_size(ink, h, w, glyphs=None, bars=0):
    """글자 1~3개짜리 크롭의 글자 크기: 가까운 조각(탁점·점·'は'의 두 획)을 한 글자로 묶어 가장 큰 글자의 긴 변.

    이런 크롭은 줄·단이 없어 투영 밴드가 획 조각('お'의 몸통과 점)을 줄로 잰다. 조각이 더 많거나(글자 인식이
    1~3자로 읽었으면 바탕 부스러기는 빼고 센다), 여러 조각이
    길게 묶이거나(글자 여럿이 붙음), 길쭉한 크롭을 거의 다 채우는 덩어리(테두리로 이어진 여러 글자)거나,
    잉크 대부분이 크롭을 가로지르는 덩어리(바탕)면 None — 원래 경로를 따른다. bars개까지 막대 부호 조각은 지우고 잰다.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if bars > 0:
        solid = [i for i in range(1, num) if stats[i][4] >= 16]
        dropped = [solid[j] for j in _bar_pieces([(stats[i][2], stats[i][3]) for i in solid], bars)]
        if dropped:
            ink = ink.copy()
            ink[np.isin(labels, dropped)] = 0
            num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    solid = [i for i in range(1, num) if stats[i][4] >= 16]
    # 크롭 한 변을 거의 다 가로지르는 조각은 칸 테두리·말풍선 선이지 글자가 아니다
    pieces = [i for i in solid if stats[i][2] < 0.95 * w and stats[i][3] < 0.95 * h]
    if len(pieces) > _FEW_MAX_PIECES and glyphs is not None and 0 < glyphs <= _FEW_MAX_GLYPHS:
        biggest_area = max(stats[i][4] for i in pieces)
        pieces = [i for i in pieces if stats[i][4] >= _FEW_CLUTTER_AREA * biggest_area]
    if not pieces or len(pieces) > _FEW_MAX_PIECES:
        return None
    if sum(stats[i][4] for i in pieces) < 0.5 * sum(stats[i][4] for i in solid):
        return None
    largest = max(max(stats[i][2], stats[i][3]) for i in pieces)
    k = max(2, int(round(_FEW_MERGE_RATIO * largest)))
    kept = np.isin(labels, pieces).astype(np.uint8) * 255
    merged = cv2.dilate(kept, np.ones((k, k), np.uint8))
    num_m, labels_m, stats_m, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    glyphs = []
    for i in range(1, num_m):
        x, y, bw, bh, _ = (int(v) for v in stats_m[i])
        inside = (kept[y:y + bh, x:x + bw] > 0) & (labels_m[y:y + bh, x:x + bw] == i)
        if inside.sum() < 16:
            continue
        ys, xs = np.nonzero(inside)
        ids = np.unique(labels[y:y + bh, x:x + bw][inside])
        gw, gh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        biggest = max(ids, key=lambda j: max(stats[j][2], stats[j][3]))
        if max(gw, gh) > 1.25 * max(stats[biggest][2], stats[biggest][3]):
            # 묶으니 가장 큰 조각보다 한참 커졌다 = 따로인 글자·문장부호('あ。')까지 붙었다 → 가장 큰 조각만
            gw, gh = int(stats[biggest][2]), int(stats[biggest][3])
        glyphs.append((gw, gh, len(ids)))
    if not glyphs:
        return None
    top = max(max(gw, gh) for gw, gh, _ in glyphs)
    main = [g for g in glyphs if max(g[0], g[1]) >= 0.5 * top]
    if len(main) > _FEW_MAX_GLYPHS:
        return None
    if any(n >= 2 and max(gw, gh) > _FEW_MAX_ASPECT * min(gw, gh) for gw, gh, n in main):
        return None
    if top > 0.8 * max(h, w) and max(h, w) > 1.3 * min(h, w):
        return None
    if len(main) == 1:
        gw, gh, _ = main[0]
        if max(gw, gh) > 2.2 * min(gw, gh) and max(gw, gh) > 0.7 * max(h, w):
            return None
    return float(top)


# 흰 몸통에 검은 테두리를 두른 글(袋文字): 테두리끼리 붙은 잉크 덩어리 하나가 몸통 구멍을 _BODY_HOLES_PER_BLOB개 이상
# 품고, 잉크의 _BODY_HUGGING_MIN 이상이 몸통 가까이 있으며(테두리가 몸통을 감쌈), 몸통 굵기가 테두리의 _BODY_THICK_MAX배
# 이하일 때(속 빈 도안 로고는 몸통이 뚱뚱하다) 몸통으로 잰다. 잉크로 잰 값이 몸통으로 잰 값의 _BODY_GAIN배를 넘을 때만 쓴다
_BODY_HOLES_PER_BLOB = 6
_BODY_HUGGING_MIN = 0.8
_BODY_THICK_MAX = 2.0
_BODY_GAIN = 1.5
# 크롭 가장자리의 이 몫 이상이 어두우면(톤·검은 칸 위 흰 글자) 몸통이 바탕의 흰 선·가장자리와 이어져 조각만 남으므로 쓰지 않는다
_BODY_DARK_BORDER_MAX = 0.5


def _outlined_bodies(ink):
    """테두리 두른 흰 글자의 몸통(테두리 안 흰 조각)을 글자 잉크로 (0/255). 그런 글이 아니면 None.

    글자 속 빈칸(口·日)은 한 글자에 몇 개뿐이고 잉크 획의 일부만 닿는다 — 글자가 붙어 빈칸이 많은 검은 대사·굵은
    제목은 잉크의 15~65%만 빈칸 가까이 있고, 테두리 글자는 86~98%다.
    """
    h, w = ink.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(255 - ink, connectivity=4)
    min_area = max(6.0, 0.0002 * h * w)
    keep = [i for i in range(1, num)
            if stats[i][4] >= min_area and stats[i][0] > 0 and stats[i][1] > 0
            and stats[i][0] + stats[i][2] < w and stats[i][1] + stats[i][3] < h]
    if len(keep) < _BODY_HOLES_PER_BLOB:
        return None
    _, ink_labels = cv2.connectedComponents(ink, connectivity=8)
    per_blob = {}
    for i in keep:
        x, y, comp_w = (int(v) for v in stats[i][:3])
        # 구멍 맨 윗줄 첫 픽셀의 바로 위는 그 구멍을 둘러싼 잉크다 (구멍은 크롭 테두리에 닿지 않는다)
        first = x + int(np.argmax(labels[y, x:x + comp_w] == i))
        blob = int(ink_labels[y - 1, first])
        if blob:
            per_blob[blob] = per_blob.get(blob, 0) + 1
    if not per_blob or max(per_blob.values()) < _BODY_HOLES_PER_BLOB:
        return None
    holes = np.isin(labels, keep)
    dark = ink > 0
    dark_thick = 2.0 * float(np.percentile(cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 3)[dark], 90))
    hole_thick = 2.0 * float(np.percentile(cv2.distanceTransform(holes.astype(np.uint8), cv2.DIST_L2, 3)[holes], 90))
    to_hole = cv2.distanceTransform((~holes).astype(np.uint8), cv2.DIST_L2, 3)
    if float((to_hole[dark] <= max(2.0, dark_thick)).mean()) < _BODY_HUGGING_MIN or hole_thick > _BODY_THICK_MAX * dark_thick:
        return None
    return (holes * 255).astype(np.uint8)


def estimate_glyph_height(crop_pil, glyphs=None, bars=0, upright=False) -> float:
    """크롭의 메인 텍스트 글자 본체 높이(px)를 추정합니다. 측정 불가 시 0.

    glyphs = 글자 인식이 읽은 글자 수(문장 부호 빼고, 모르면 None) — 1~3자면 바탕 부스러기에 덜 흔들린다.
    bars = 글자 인식이 읽은 막대 부호(장음·줄표·물결·느낌표·세로줄) 수 — 그만큼 가장 길쭉한 조각을 크기 근거에서 뺀다.
    upright = 세로가 가로보다 긴 크롭을 세로쓰기로 본다 — 줄 구조 점수로 고르면 후리가나가 붙은 세로 여러 단을 가로
    줄로 보고 단 사이 간격을 글자 크기로 잰다 (font_relative.measure_weight의 글자 폭도 이 기준으로 단을 잰다).
    """
    gray = np.array(crop_pil.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    if h < 12 or w < 12:
        return 0.0
    ink = _ink_mask(gray)
    size = _estimate_from_ink(ink, glyphs, bars, upright)
    # 테두리 두른 흰 글자는 붙은 테두리를 한 덩어리로 재 2~3배가 된다 — 그대로 쓰면 원문보다 몇 배 크게 식자된다
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bodies = _outlined_bodies(ink) if float((border < 128).mean()) < _BODY_DARK_BORDER_MAX else None
    if bodies is not None:
        body_size = _estimate_from_ink(bodies, glyphs, bars, upright)
        if body_size > 0 and (size <= 0 or size > _BODY_GAIN * body_size):
            return body_size
    return size


def _estimate_from_ink(ink, glyphs=None, bars=0, upright=False) -> float:
    """글자 잉크(0/255)에서 글자 본체 높이 — 줄·단 밴드, 조각 높이, 교차 검증 (estimate_glyph_height)."""
    h, w = ink.shape

    row_profile = ink.sum(axis=1) / 255.0
    col_profile = ink.sum(axis=0) / 255.0
    if h > w * 1.6 or (upright and h > w):
        vertical = True
    elif w > h * 1.6:
        vertical = False
    else:
        vertical = _band_score(col_profile, h) > _band_score(row_profile, w)

    profile, cross_len, axis = (col_profile, h, "x") if vertical else (row_profile, w, "y")
    # 교차 검증에 쓰는 선택 축·반대 축 주요 밴드 (본문 밴드 추정과 따로 잰다)
    ch_majors, _, ch_pitch = _major_bands(profile, cross_len)
    cc = _component_heights(ink, h, w, bars=bars)
    if len(_refined_bands(profile, cross_len)) > len(ch_majors):
        # 주요 밴드 밖 조각이 안쪽 조각보다 한참 작으면 후리가나 줄·컬럼이다 — 본문 글자 크기로 세지 않는다. 후리가나
        # 조각이 많으면 성분 높이 p70이 끌려 내려가 본문 글자를 후리가나 크기로 잴 수 있다. 주요 밴드가 붙은 두 단일 뿐인
        # 크롭은 안팎 조각 크기가 같아 그대로다
        main = _component_heights(ink, h, w, ch_majors, vertical, bars=bars)
        rest = _component_heights(ink, h, w, ch_majors, vertical, inside=False, bars=bars)
        if main and rest and np.median(main) >= _RUBY_PIECE_RATIO * np.median(rest):
            cc = main
    cc_p70 = float(np.percentile(cc, 70)) if len(cc) >= 3 else 0.0
    cc_max = float(max(cc)) if cc else 0.0
    # 붙은 두 단·두 줄 가드용 글자 조각의 긴 변 최대 — 높이만 보면 획 조각으로 쪼개진 가는 글꼴에서 작게 나온다
    num, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    sides = [max(stats[i][2], stats[i][3]) for i in range(1, num)
             if stats[i][4] >= 16 and stats[i][2] < 0.95 * w and stats[i][3] < 0.95 * h]
    piece_max = float(max(sides)) if sides else cc_max
    opp_profile, opp_cross = (row_profile, w) if vertical else (col_profile, h)
    opp_majors, opp_med, _ = _major_bands(opp_profile, opp_cross)

    # 재절단 나눔선이 글자를 가로지르면 나누지 않은 밴드로도 잰다 — 글자 안 획 틈에서 잘게 나눠 작게 잰 것을
    # 바로잡는 장치라 키울 때만 쓴다 (나누지 않아 계산 경로가 바뀌며 작아지는 곳은 원래 값을 둔다)
    plain = _band_estimate(profile, cross_len)
    checked = _band_estimate(profile, cross_len, ink, axis)
    result = _size_from_bands(*plain, ink, h, w, vertical, cross_len, cc, cc_p70, cc_max, piece_max,
                              ch_majors, ch_pitch, opp_majors, opp_med, opp_cross)
    if checked != plain:
        result = max(result, _size_from_bands(*checked, ink, h, w, vertical, cross_len, cc, cc_p70, cc_max, piece_max,
                                              ch_majors, ch_pitch, opp_majors, opp_med, opp_cross))

    # 글자 1~3개짜리 크롭은 줄·단 구조가 없어 밴드가 획 조각을 잰다 — 묶은 글자의 긴 변 (이것도 키울 때만)
    few = _few_glyph_size(ink, h, w, glyphs, bars)
    if few is not None and few > result:
        return few
    return result


def _size_from_bands(band, n_bands, ink, h, w, vertical, cross_len, cc, cc_p70, cc_max, piece_max,
                     ch_majors, ch_pitch, opp_majors, opp_med, opp_cross):
    """본문 밴드(두께 중앙값, 수)에서 글자 크기를 정한다 — 성분 높이와 섞고 반대 축·피치·납작 글자로 교차 검증."""
    # 메가밴드 무효화: 단일 밴드가 축의 75%를 넘으면 줄 구조가 아니라 '구조 없음'
    # (인접 컬럼이 전부 이어진 경우 등) — 성분/반대축 신호에 맡긴다.
    if n_bands == 1 and band > 0.75 * cross_len:
        band, n_bands = 0.0, 0

    if n_bands <= 1 and cc_p70 > 0 and band > cc_p70 * 1.8:
        deskewed = _deskew(ink)
        retry_profile = (deskewed.sum(axis=0) if vertical else deskewed.sum(axis=1)) / 255.0
        band_retry, n_retry = _band_estimate(retry_profile, cross_len)
        if n_retry > n_bands and band_retry > 0:
            band, n_bands = band_retry, n_retry

    def _base_estimate():
        # 규칙적 다단(밴드 ≥3)은 밴드가 가장 강한 신호
        if n_bands >= 3 and band > 0:
            if len(cc) >= 8 and band > cc_p70 * 1.8:
                return float(np.sqrt(band * cc_p70))
            return band
        # 성분이 적은 큰 SFX → 최대 성분 (외곽선 체인 가드: 밴드의 1.8배 이내일 때만)
        if len(cc) <= 4 and cc_max > 0:
            if band <= 0 or cc_max <= band * 1.8:
                if band <= 0:
                    return cc_max
                return float(np.mean((cc_max, band))) if cc_max <= band * 1.6 else cc_max
        if band <= 0:
            return cc_p70 if cc_p70 > 0 else cc_max
        if cc_p70 <= 0:
            return band
        lo, hi = sorted((band, cc_p70))
        if hi > lo * 1.6:
            # 밴드 구조가 뚜렷한데(2줄) 성분이 한참 작고 '개수도 적으면' 손글씨처럼
            # 글자 몇 개가 획 조각으로 쪼개진 것 — 밴드가 글자 크기다.
            # 성분이 많으면(>12) 밴드는 여러 줄이 붙은 블록일 가능성이 높아 cc 신뢰.
            if band > cc_p70 and n_bands >= 2 and len(cc) <= 12:
                return band
            return cc_p70 if len(cc) >= 8 else float(np.sqrt(lo * hi))
        return float(np.mean((band, cc_p70)))

    result = _base_estimate()

    # 전각 정사각 교차 검증: CJK 전각은 정사각이므로 '줄 축의 밴드 두께'와
    # '반대 축의 밴드 중심 피치'는 모두 em이라 서로 일치해야 한다. 축이 잘못
    # 선택되면(세로 글자열의 글자 행을 가로 줄로 오인) 선택 축 밴드는 글자
    # 잉크 높이만 재서 과소평가하는데, 이때 반대 축 밴드 두께가 선택 축
    # 피치와 일치하면 그것이 진짜 em이다. (상향 전용 — 올바른 축 선택에서는
    # 피치=줄간격>em이라 게이트가 자연히 닫히고, 닫히지 않아도 반대축≈결과라 무해)
    if opp_med > 0 and (len(opp_majors) >= 2 or opp_med <= 0.6 * opp_cross):
        # 잉크 커버리지: 반대 축 주요 밴드가 전체 잉크의 절반 이상을 품어야
        # 진짜 텍스트 본체다 (글자 쌍이 붙은 행 하나가 우연히 피치와
        # 일치하는 경우 차단).
        total_ink = float(ink.sum()) + 1e-6
        in_bands = sum(
            float((ink[a:b, :] if vertical else ink[:, a:b]).sum())
            for a, b in opp_majors
        )
        coverage_ok = in_bands / total_ink >= 0.5
        # 물리 제약: 반대축 밴드 두께(=em)는 선택축 피치(=em+간격) '이하'여야
        # 한다. 이를 넘으면 em이 아니라 병합 블록이 우연히 게이트에 든 것이다.
        square_ok = (
            ch_pitch > 0
            and abs(opp_med - ch_pitch) <= 0.30 * ch_pitch
            and opp_med <= ch_pitch * 1.15
        )
        # 누더기 밴드 구제: 선택 축 밴드가 성분 높이보다 한참 작으면(과분할)
        # 균일한 반대 축 밴드(3개 이상)가 더 신뢰할 수 있는 글자 신호다.
        opp_sizes = [e - s for s, e in opp_majors]
        ragged = (
            band > 0 and cc_p70 > 0 and band < cc_p70 * 0.65
            and len(opp_majors) >= 3
            and float(np.std(opp_sizes)) / max(opp_med, 1e-6) < 0.25
        )
        # 최소 이득 1.15×: 반대축≈결과면 새 정보가 없는데 전각-잉크 차이만큼
        # 미세 인플레만 생긴다.
        # 글자 조각이 넉넉한데(8개 이상) 가장 큰 조각 긴 변의 1.35배를 넘으면 붙은 두 단·두 줄을 한 칸으로 잰 것이다
        # (여러 단 말풍선이 글자 크기의 두 배 넘게 잡히는 경우)
        limit = max(result * 2.6, 1.0)
        if len(cc) >= 8:
            limit = min(limit, 1.35 * piece_max)
        if (square_ok or ragged) and coverage_ok and result * 1.15 < opp_med <= limit:
            result = opp_med

    # 줄/컬럼 간 피치 교차 검증 — 전각 em은 한자/가나 잉크 크기와 무관하므로
    # 글자 중심 간격이 잉크 기반 추정보다 크면 (상향으로만) 보정한다.
    # 상향 전용인 이유: 한자 부수가 셀로 갈라지면 피치가 작게 나올 수 있어
    # 하향 보정은 위험하다. 상한(×2.6)은 글자 쌍 병합 인플레 가드.
    if ch_majors:
        pitch, strong = _line_pitch_estimate(ink, ch_majors, vertical)
        # 단일 줄은 세로 컬럼일 때만: 가로 한 줄의 advance는 글자 '폭' 피치라
        # (ょ·っ 등 폭≠높이) 높이 신호로 쓰기엔 부정확하다.
        if pitch > 0 and (strong or (len(ch_majors) == 1 and vertical)):
            if result * 1.25 < pitch <= max(result * 2.6, 1.0):
                result = pitch

    # 납작 글자 보정: ヘ·ー·つ 같은 글자는 잉크 높이가 em의 ~30%라 모든 위 경로가
    # 크게 과소평가한다. 납작 성분이 지배하고 폭/피치 추정이 충분히 크면 교체.
    flat = _flat_glyph_estimate(ink, h, w)
    if flat > result * 1.4:
        return flat
    return result
