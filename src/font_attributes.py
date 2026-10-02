"""원문 글자 모양 실측 — 글자 극성(검은/흰 글자)과 외곽선(유무·두께).

지우기 전 원본 조각과 글자 모양 마스크(comic-text-mask)로 잰다. 마스크는 획보다 몇 픽셀 넉넉한 '글자 영역'이라,
색은 원본 밝기로 가른다(koharu 방식을 흑백 만화에 맞춤). 판단이 어려우면 None(모름)을 돌려 식자가 기본 규칙을
쓰게 한다 — 흰 종이 위의 흰 외곽선처럼 원리상 알 수 없는 경우도 모름이다.

0. 바탕이 흰 종이·검은 면이면 글자는 그 반대색이고 외곽선은 알 수 없다(모름). 단 말풍선 밖 글자는 흰 종이 위에서
   '흰 채움+검은 외곽선' 가설(1의 검사)이 통과하면 검은 외곽선 두른 흰 글자로 본다.
1. 그 밖의 바탕에서는 외곽선 가설 두 가지(검은 채움+흰 테두리 / 흰 채움+검은 테두리)를 각각 검사한다.
   - 테두리 색으로 막힌 채 조각 가장자리에서 닿지 않는 채움 픽셀(둘러싸인 채움)이 글자 영역의 채움 대부분이어야 하고,
   - 채움 둘레를 1px씩 넓혀 가며 본 띠가 테두리 색으로 빈틈없이 이어지다가, 그 바깥에서는 끊겨야(바탕과 구별돼야) 한다.
   - 이어진 폭이 외곽선 두께다.
2. 외곽선이 없으면, 조각 둘레(바탕)보다 글자 영역에 유난히 많은 색을 글자색으로 본다.
"""
import cv2
import numpy as np

_DARK = 90                 # 이 밝기 이하 = 검은 픽셀
_LIGHT = 170               # 이 밝기 이상 = 흰 픽셀
_MIN_FILL_PX = 30          # 둘러싸인 채움 픽셀이 이보다 적으면 외곽선 판단 안 함
_MIN_FILL_COMPONENT_RATIO = 0.06  # 채움 덩어리 최소 크기 (글자 크기 대비 한 변) — 스크린톤 점 제외
_MIN_OUTLINE_PX = 2        # 이보다 얇은 띠는 외곽선으로 보지 않음 (안티에일리어싱·톤 틈)
_BAND_COVERAGE = 0.7       # 글자 영역 안 테두리 색 픽셀 중 채움에 붙은 비율이 이 이상이어야 외곽선
_FILL_BAND_RATIO_MAX = 2.2 # 둘러싸인 채움 두께가 띠 두께의 이 배수를 넘으면 채움이 아니라 글자의 속 빈 곳(口·の)
_MIN_OUTSIDE_PX = 50       # 바탕 표본이 이보다 적으면 판정하지 않는다
_MIN_OUTSIDE_SHARE = 0.15  # 바탕 표본이 조각 넓이의 이 비율에 못 미쳐도 판정하지 않는다
_BAND_SOLID = 0.8          # 채움 둘레 한 줄에서 테두리 색 비율이 이 이상이면 띠가 이어짐
_FIRST_RING_MIN = 0.5      # 첫 줄(안티에일리어싱 섞임)의 테두리 색 최소 비율
_BAND_DROP = 0.2           # 띠 바깥은 이만큼 이상 테두리 색 비율이 떨어져야(바탕과 구별)
_MIN_RING_PX = 30          # 한 둘레의 픽셀이 이보다 적으면 그 둘레는 안 봄
_MAX_OUTLINE_RATIO = 0.35  # 외곽선을 찾는 최대 폭 (글자 크기 대비)
_INK_ENRICHMENT = 0.12     # 글자 영역 안 비율이 둘레보다 이만큼 높아야 그 색을 글자색으로 봄
_INK_MARGIN = 0.15         # 두 색의 늘어난 정도 차이가 이보다 작으면 모름
_PLAIN_BACKGROUND = 0.9    # 바탕 픽셀 중 한 색이 이 비율 이상이면 흰 종이/검은 면
_LIGHT_GUARD_SHARE = 0.15  # 흰 글자 판정 전: 흰색에 둘러싸인 검은 덩어리가 흰 픽셀의 이 비율 이상이면 모름


def _ring_fractions(inside, opposite, max_px):
    """inside 둘레 k번째 줄(k=1..)마다 opposite 비율."""
    dist = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 3)
    fractions = []
    for k in range(1, max_px + 4):
        ring = (dist > k - 1) & (dist <= k)
        if int(ring.sum()) < _MIN_RING_PX:
            break
        fractions.append(float(opposite[ring].mean()))
    return fractions


def _outline_test(fill_cls, stroke_cls, region, font_size, max_px):
    """fill_cls 채움을 stroke_cls 테두리가 둘러싼 글자인가 → (두께 px 또는 None, 둘러싸인 채움 픽셀 수)."""
    # 테두리 색을 지나지 않고 조각 가장자리에서 닿는 픽셀
    num, labels = cv2.connectedComponents((~stroke_cls).astype(np.uint8), connectivity=4)
    border_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    reachable = np.isin(labels, border_labels[border_labels > 0])
    enclosed = (fill_cls & region & ~reachable).astype(np.uint8)
    # 스크린톤 점처럼 작은 덩어리는 채움이 아니다
    min_area = max(12, int((_MIN_FILL_COMPONENT_RATIO * font_size) ** 2))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, connectivity=8)
    keep = np.zeros(num, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    enclosed = keep[labels]
    enclosed_px = int(enclosed.sum())
    if enclosed_px < _MIN_FILL_PX:
        return None, enclosed_px
    fractions = _ring_fractions(enclosed, stroke_cls, max_px)
    # 채움 바로 옆 첫 줄은 안티에일리어싱으로 중간 밝기가 섞여 기준을 낮춘다
    if not fractions or fractions[0] < _FIRST_RING_MIN:
        return None, enclosed_px
    width = 1
    while width < len(fractions) and fractions[width] >= _BAND_SOLID:
        width += 1
    if width < _MIN_OUTLINE_PX or width > max_px:
        return None, enclosed_px
    beyond = fractions[width:width + 3]
    if not beyond or max(beyond) > fractions[width - 1] - _BAND_DROP:
        return None, enclosed_px  # 띠 바깥을 못 봤거나 바탕과 이어진다
    # 채움 덩어리가 띠보다 지나치게 두꺼우면 채움이 아니라 글자의 속 빈 곳이다 — 검은 글자를 흰 채움으로 뒤집지 않게
    num, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed.astype(np.uint8), connectivity=8)
    depth = cv2.distanceTransform(enclosed.astype(np.uint8), cv2.DIST_L2, 3)
    thickness = [2.0 * float(depth[labels == i].max()) for i in range(1, num)]
    if thickness and float(np.median(thickness)) > _FILL_BAND_RATIO_MAX * width:
        return None, enclosed_px
    # 테두리 색 픽셀 대부분이 채움에 바짝 붙어 있어야 외곽선이다 — 검은 글자의 획은 속 빈 곳이 없는 데도 많다
    # 스크린톤의 가는 틈(1~2px)은 빼고 두께 있는 테두리 색만 센다
    solid_stroke = cv2.morphologyEx(stroke_cls.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    stroke_in_region = solid_stroke & region
    near = cv2.distanceTransform((~enclosed).astype(np.uint8), cv2.DIST_L2, 3) <= width + 1.5
    if (stroke_in_region & near).sum() < _BAND_COVERAGE * max(1, stroke_in_region.sum()):
        return None, enclosed_px
    return float(width), enclosed_px


def measure_text_look(crop_rgb, glyph_mask, font_size, freeform=False):
    """원문 글자 모양 {polarity, outline, outline_px, basis} — 모르면 None.

    basis는 극성을 정한 근거: "plain"(한 가지 색 바탕의 반대색) / "outline"(외곽선 확인) / "ink"(글자 영역에 많은 색 —
    덜 믿을 만해 식자에는 쓰지 않는다) / None.
    freeform=True(말풍선 밖 글자)면 흰 종이 바탕에서도 '흰 채움+검은 외곽선' 가설을 한 번 본다. 말풍선 속 글자는 이 가설을 보지 않는다.
    """
    look = {"polarity": None, "outline": None, "outline_px": 0.0, "basis": None}
    if crop_rgb is None or glyph_mask is None or crop_rgb.size == 0 or not glyph_mask.any():
        return look
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    region = cv2.dilate(glyph_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    outside = ~cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    # 글자가 조각을 거의 채워 바탕을 충분히 볼 수 없으면(작은 세로 글자 등) 모름 — 가장자리 몇 줄만 보고 바탕을 정하면
    # 흰 종이 위 검은 글자를 '검은 면 위 흰 글자'로 뒤집는다
    if outside.sum() < max(_MIN_OUTSIDE_PX, _MIN_OUTSIDE_SHARE * outside.size):
        return look
    dark, light = gray <= _DARK, gray >= _LIGHT
    # 바탕이 흰 종이/검은 면인지는 바탕 픽셀 거의 전부가 그 색일 때만 — 스크린톤은 흰 픽셀이 많아도 종이가 아니다
    paper = float(light[outside].mean()) >= _PLAIN_BACKGROUND
    black = float(dark[outside].mean()) >= _PLAIN_BACKGROUND
    max_px = max(3, int(round(_MAX_OUTLINE_RATIO * max(1, int(font_size)))))

    # 바탕이 한 가지 색(흰 종이·검은 면)이면 글자는 그 반대색이고 외곽선은 알 수 없다. 이런 바탕에서 외곽선 가설을
    # 검사하면 글자의 속 빈 곳(口·の)을 채움으로 잘못 봐 흰 글자를 '흰 테두리 두른 검은 글자'로 뒤집는다.
    if paper or black:
        ink = light if black else dark
        if float(ink[region].mean() - ink[outside].mean()) < _INK_ENRICHMENT:
            return look
        # 단, 말풍선 밖 글자는 흰 종이 위라도 검은 외곽선 두른 흰 글자일 수 있다 — 검은 외곽선이 '검은 잉크'로 잡혀 검은
        # 글자로 보기 쉽다(흰 종이 속 검은 띠 위 흰 글자도 같다). 흰 채움 가설이 끝까지(속 빈 곳·테두리 비율 검사 포함)
        # 통과할 때만 흰 글자로 본다 — 흰 종이 위 검은 글자를 뒤집지 않게. 검은 먹칠 바탕의 거꾸로 가설은 흰 글자를
        # 뒤집어 버려 쓰지 않는다
        if freeform and paper:
            width, _ = _outline_test(light, dark, region, font_size, max_px)
            if width is not None:
                return {"polarity": "light", "outline": True, "outline_px": width, "basis": "outline"}
        look["polarity"] = "light" if black else "dark"
        look["basis"] = "plain"
        return look

    # 1. 외곽선 가설 (스크린톤·그림처럼 한 가지 색이 아닌 바탕)
    enclosed_dark_px = 0
    found = []
    for fill_cls, stroke_cls, polarity in ((dark, light, "dark"), (light, dark, "light")):
        width, enclosed_px = _outline_test(fill_cls, stroke_cls, region, font_size, max_px)
        if polarity == "dark":
            enclosed_dark_px = enclosed_px
        if width is not None:
            found.append((polarity, width))
    if len(found) == 2:
        return look  # 두 가설이 다 맞는다 — 어느 쪽이 채움인지 모른다
    if found:
        polarity, width = found[0]
        return {"polarity": polarity, "outline": True, "outline_px": width, "basis": "outline"}

    # 2. 외곽선 없음 — 둘레보다 글자 영역에 유난히 많은 색이 글자색
    enrich_dark = float(dark[region].mean() - dark[outside].mean())
    enrich_light = float(light[region].mean() - light[outside].mean())
    # 두 색이 다 많으면 외곽선을 못 찾은 테두리 글자일 가능성이 커서 모름으로 둔다
    if min(enrich_dark, enrich_light) >= _INK_ENRICHMENT or abs(enrich_dark - enrich_light) < _INK_MARGIN:
        return look
    if max(enrich_dark, enrich_light) < _INK_ENRICHMENT:
        return look
    polarity = "dark" if enrich_dark > enrich_light else "light"
    # 흰 글자로 잘못 보면 지금보다 나빠진다 — 흰색에 둘러싸인 검은 덩어리가 제법 있으면(흰 외곽선 두른 검은 글자일 수
    # 있음) 모름으로 둔다
    if polarity == "light" and enclosed_dark_px >= _LIGHT_GUARD_SHARE * max(1, int((light & region).sum())):
        return look
    look["polarity"] = polarity
    look["basis"] = "ink"  # 외곽선은 '못 찾음'이지 '없음'이 아니므로 None으로 둔다
    return look


# ── 원문 테두리 두께 — 지운 그림을 글자 뒤 바탕으로 삼아 잰다 ─────────────────────────
# 원문의 흰 테두리 두께도 글자 크기처럼 따라 한다. 글자 마스크만 보는 외곽선 검사(_outline_test)는 실제 외곽선 글줄을
# 잡지 못한다. 지운 그림은 글자 뒤 바탕이라, 채움 둘레에서 원문이 그 바탕보다 뚜렷이 밝은(검은 글자)·어두운(흰 글자)
# 띠가 곧 테두리다. 테두리 둘레가 흰 여백·옅은 톤이라 바탕과 가를 수 없으면 모름(기본 테두리)으로 둔다
_RIM_FILL_LUMA = 128          # 채움 경계 — 안티에일리어싱 중간 밝기에서 재야 테두리 폭이 글자 밖으로 보이는 폭과 맞는다
_RIM_FILL_CONTRAST = 25       # 채움은 지운 바탕(5px 평균)보다 이만큼 어둡다(흰 글자는 밝다) — 바탕의 먹칠·톤을 채움으로 안 본다
_RIM_MIN_ROOM = 40            # 바탕이 이만큼도 더 밝아질(어두워질) 수 없으면(흰 종이·먹칠) 그 거리의 테두리는 보이지 않는다
_RIM_COVER = 0.5              # 띠 덮임이 이 아래로 떨어지는 거리 = 채움 둘레 지점별 테두리 폭의 중앙값
_RIM_BEYOND = 0.35            # 테두리 바깥 세 줄은 이 아래여야 — 흰 여백이 바탕으로 이어지면 테두리가 아니다
_RIM_SEARCH_RATIO = 0.35      # 채움에서 이 거리(글자 크기 대비)까지 본다
_RIM_POLARITY_MARGIN = 0.15   # 두 가설이 다 맞으면 띠 덮임이 이만큼 더 큰 쪽을 믿는다


def rim_window_px(font_size):
    """measure_rim에 넘길 조각을 글자 상자 밖으로 넓힐 폭 (찾는 거리 + 여유)."""
    return max(3, int(round(_RIM_SEARCH_RATIO * float(font_size or 20)))) + 3


def _rim_width(cover, max_px):
    """채움에서 1px씩 떨어진 띠의 덮임 목록 → (테두리 폭, 띠 평균 덮임). 테두리가 없거나 잴 수 없으면 None."""
    if len(cover) < 2 or cover[0] is None or cover[1] is None:
        return None  # 채움 바로 옆부터 바탕이 흰 종이·먹칠이다
    start = 0 if cover[0] >= _RIM_COVER else (1 if cover[1] >= _RIM_COVER else None)
    if start is None:
        return None  # 원문이 바탕과 같다 — 테두리 없음
    end = start
    while end + 1 < len(cover) and cover[end + 1] is not None and cover[end + 1] >= _RIM_COVER:
        end += 1
    if end + 1 >= len(cover) or cover[end + 1] is None or end + 1 > max_px:
        return None  # 테두리 끝을 못 봤다 (지운 자리 밖까지 이어짐)
    beyond = [c for c in cover[end + 1:end + 4] if c is not None]
    if beyond and max(beyond) > _RIM_BEYOND:
        return None  # 띠가 바탕으로 이어진다 — 흰 여백이지 테두리가 아니다
    width = end + 0.5 + (cover[end] - _RIM_COVER) / max(1e-6, cover[end] - cover[end + 1])
    return width, float(np.mean(cover[start:end + 1]))


def measure_rim(orig_rgb, erased_rgb, glyph_mask, font_size):
    """원문 테두리 (글자 밖으로 보이는 폭 px, 글자 극성 "dark"/"light") — 잴 수 없으면 None.

    세 조각은 같은 자리·크기(원문, 지운 그림, 글자 마스크). 채움(검은 글자: 원문 128 이하이고 지운 바탕보다 어두움) 둘레를
    1px 거리 띠로 나눠, 지운 자리 안에서 띠마다 '원문 평균 밝기가 지운 바탕 평균보다 밝아진 몫 ÷ 더 밝아질 수 있던 몫'을
    덮임으로 본다(흰 글자는 거울). 평균이라 원문과 지운 그림의 톤 점 위치가 달라도 흔들리지 않는다.
    """
    if orig_rgb is None or erased_rgb is None or glyph_mask is None or not glyph_mask.any():
        return None
    orig = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    erased_gray = cv2.cvtColor(erased_rgb, cv2.COLOR_RGB2GRAY)
    back = cv2.blur(erased_gray, (5, 5)).astype(np.int16)
    erased_gray = erased_gray.astype(np.int16)
    size = float(font_size or 20)
    max_px = max(3, int(round(_RIM_SEARCH_RATIO * size)))
    region = cv2.dilate(glyph_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    changed = orig != erased_gray  # 지운 자리 — 그 밖은 원문 그대로라 뒤 바탕을 모른다
    min_area = max(12, int((_MIN_FILL_COMPONENT_RATIO * size) ** 2))
    found = {}
    for polarity in ("dark", "light"):
        if polarity == "dark":
            fill = (orig <= _RIM_FILL_LUMA) & (back - orig >= _RIM_FILL_CONTRAST) & region
        else:
            fill = (orig >= _RIM_FILL_LUMA) & (orig - back >= _RIM_FILL_CONTRAST) & region
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), connectivity=8)
        keep = np.zeros(num, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        fill = keep[labels]
        if fill.sum() < _MIN_FILL_PX:
            continue
        dist = cv2.distanceTransform((~fill).astype(np.uint8), cv2.DIST_L2, 5)
        cover = []
        for k in range(1, max_px + 5):
            ring = (dist > k - 1) & (dist <= k) & changed
            if ring.sum() < 15:
                break
            o, e = float(orig[ring].mean()), float(erased_gray[ring].mean())
            room = 255.0 - e if polarity == "dark" else e
            cover.append(None if room < _RIM_MIN_ROOM else max(0.0, ((o - e) if polarity == "dark" else (e - o)) / room))
        width = _rim_width(cover, max_px)
        if width is not None:
            found[polarity] = width
    if len(found) == 2:
        if abs(found["dark"][1] - found["light"][1]) < _RIM_POLARITY_MARGIN:
            return None  # 어느 쪽이 채움인지 모른다
        polarity = max(found, key=lambda p: found[p][1])
    elif found:
        polarity = next(iter(found))
    else:
        return None
    return round(found[polarity][0], 2), polarity
