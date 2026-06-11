"""잉크 기하 측정으로 원문 글자의 크기와 기울기를 추정합니다.

실측 벤치마크에서 회귀 모델 예측보다 오차가 크게 낮아 크기·각도의
기본 산출 경로로 쓰인다 (모델은 글씨체 분류 전용).

크기 추정 구성:
- OTSU 이진화 + 극성 자동 반전 (흰 글자/검은 배경 대응)
- 축 선택: 종횡비가 애매하면 양축 밴드 구조 점수로 결정
- 투영 밴드: 거친 밴드를 피크의 40%로 재절단해 후리가나·병합 줄 분리
- 연결 성분 높이 p70 (성분 3개 이상일 때만 신뢰)
- 줄/컬럼 피치와 전각 정사각 성질을 이용한 교차 검증 (상향 전용)

주의: 자기상관(주기성) 기반 피치 추정은 글자 주기가 아닌 획 간격을
잡는 경향이 있어 사용하지 않는다.
"""
import cv2
import numpy as np


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


def _refined_bands(profile, cross_len):
    """거친 밴드 내부를 자기 피크 40%로 재절단 (후리가나·병합 줄 분리).

    글자 내부 획 틈으로 인한 과분할(최대 하위밴드 < 원래의 42%)이면 원래 밴드 유지.
    """
    out = []
    for a, b in _coarse_bands(profile, max(2.0, 0.03 * cross_len)):
        segment = profile[a:b]
        sub = _coarse_bands(segment, max(2.0, 0.40 * segment.max()), merge_gap=1)
        if len(sub) >= 2 and max(e - s for s, e in sub) >= 0.42 * (b - a):
            out.extend((a + s, a + e) for s, e in sub)
        else:
            out.append((a, b))
    return out


def _band_estimate(profile, cross_len):
    sizes = [e - s for s, e in _refined_bands(profile, cross_len)]
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


def _component_heights(ink, h, w):
    num, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    heights = []
    # 최소 높이: 작은 크롭에선 상대(10%) 노이즈 컷, 큰 크롭(다줄 박스)에선
    # 12px로 캡 (큰 박스에서 실제 글자까지 걸러지는 것 방지).
    min_h = max(5.0, min(h * 0.10, 12.0))
    for i in range(1, num):
        _, _, comp_w, comp_h, area = stats[i]
        if area < 16 or comp_h < min_h or comp_h > h * 0.95:
            continue
        if comp_w > w * 0.95 or comp_h > comp_w * 8:
            continue
        heights.append(comp_h)
    return heights


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


def estimate_text_angle(crop_pil):
    """텍스트 기울기(°, 렌더러 규약: 양수 = render_rotated_text 양의 방향)를 추정합니다.

    원리: 후보 각도로 회전했을 때 행/열 투영 프로파일이 가장 선명해지는
    (분산 최대) 각도 = 줄/컬럼이 수평·수직으로 정렬된 상태이므로,
    그 디스큐 회전각의 부호 반전이 텍스트 자체의 기울기다.
    글자 1~2개짜리 크롭은 기준선이 없어 부정확하므로, 구조가 부족하면
    None을 반환해 호출부가 폴백하게 한다.

    선명도 이득 게이트: 최적 각도가 0° 대비 충분한 이득(8%)이 없으면
    사실상 수직이거나 신호가 약한 것이므로 0°를 반환한다.
    """
    gray = np.array(crop_pil.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    if h < 24 or w < 24:
        return None
    ink = _ink_mask(gray)
    num, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    n_comps = sum(1 for i in range(1, num) if stats[i][4] >= 16)
    # 글자 1~2개(블롭 몇 개)는 회전하면 어느 각도로든 '정렬'될 수 있어
    # 선명도 게이트가 무력화된다 — 성분 6개 이상을 요구한다.
    if n_comps < 6:
        return None

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
    for angle in range(-40, 41, 2):
        if angle == 0:
            continue
        value = _sharpness(float(angle))
        if value > best_value:
            best_angle, best_value = float(angle), value
    for angle in np.arange(best_angle - 2, best_angle + 2.01, 0.5):
        value = _sharpness(float(angle))
        if value > best_value:
            best_angle, best_value = float(angle), value

    if best_value < base * 1.08:
        return 0.0
    return -best_angle


def estimate_glyph_height(crop_pil) -> float:
    """크롭의 메인 텍스트 글자 본체 높이(px)를 추정합니다. 측정 불가 시 0."""
    gray = np.array(crop_pil.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    if h < 12 or w < 12:
        return 0.0
    ink = _ink_mask(gray)

    row_profile = ink.sum(axis=1) / 255.0
    col_profile = ink.sum(axis=0) / 255.0
    if h > w * 1.6:
        vertical = True
    elif w > h * 1.6:
        vertical = False
    else:
        vertical = _band_score(col_profile, h) > _band_score(row_profile, w)

    profile, cross_len = (col_profile, h) if vertical else (row_profile, w)
    band, n_bands = _band_estimate(profile, cross_len)

    # 메가밴드 무효화: 단일 밴드가 축의 75%를 넘으면 줄 구조가 아니라 '구조 없음'
    # (인접 컬럼이 전부 이어진 경우 등) — 성분/반대축 신호에 맡긴다.
    if n_bands == 1 and band > 0.75 * cross_len:
        band, n_bands = 0.0, 0

    cc = _component_heights(ink, h, w)
    cc_p70 = float(np.percentile(cc, 70)) if len(cc) >= 3 else 0.0

    if n_bands <= 1 and cc_p70 > 0 and band > cc_p70 * 1.8:
        deskewed = _deskew(ink)
        retry_profile = (deskewed.sum(axis=0) if vertical else deskewed.sum(axis=1)) / 255.0
        band_retry, n_retry = _band_estimate(retry_profile, cross_len)
        if n_retry > n_bands and band_retry > 0:
            band, n_bands = band_retry, n_retry

    cc_max = float(max(cc)) if cc else 0.0

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
    ch_majors, _, ch_pitch = _major_bands(profile, cross_len)
    opp_profile, opp_cross = (row_profile, w) if vertical else (col_profile, h)
    opp_majors, opp_med, _ = _major_bands(opp_profile, opp_cross)
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
        # 미세 인플레만 생긴다 (in6_dev_0008에서 +1.06× 잡음 확인).
        if (square_ok or ragged) and coverage_ok and result * 1.15 < opp_med <= max(result * 2.6, 1.0):
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
