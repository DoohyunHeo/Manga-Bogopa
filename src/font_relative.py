"""책 단위 글씨체 판정 — 글자 조각마다 모양 6분류 모델로 글씨체를 고르고, 굵기는 책 안의 보통체 기준으로 잰다.

- 모양 6분류(보통체·가는명조·손글씨·팝·각진·호러)는 생김새만 본다. 분류마다 저장된 글꼴 설정 키(FONT_MAP)로 옮긴다
  (_SHAPE_TO_FONT_KEY) — 설정에서 고른 글꼴이 그대로 이어진다.
- 가는명조는 확신할 때만(MINCHO_MIN_PROB) 해설 글꼴로, 아니면 보통체. 말풍선 밖 보통체(굵지 않은 평문)도 어두운 바탕의 흰
  글자가 아니면, 꼬리 없는 네모 해설 상자(page_drawer.is_caption_box)의 보통체도 해설 글꼴로 쓴다. 설정 '나레이션체를
  평문체로'(NARRATION_AS_STANDARD)를 켜면 모두 보통체로 그린다.
- 굵기 = 획 단면의 잉크 양 ÷ 글자 폭(세로쓰기 열 폭) — 원문 획만으로 정한다. 책의 보통체 조각 굵기가 두 무리로 깔끔히
  갈리면 그 사이를, 아니면 보통 굵기의 _ONE_GROUP_RATIO배를 문턱으로 그보다 굵은 보통체는 굵은 대사 글꼴(shouting —
  평범한 대사 글꼴의 같은 가족 굵은 글꼴, src/font_family.py)로. 3자 이하·흰 바탕이 아닌 조각(톤·그림·검은 바탕 위)은
  기준에서만 빼고 판정은 한다. 작은 글자(책 보통체 글자 폭의 0.7배 미만이거나 12px 미만)와 검은 바탕 흰 글자는 기준에서
  빼고 판정도 하지 않는다.
- 검은·회색 바탕의 흰 글자는 뒤집어서 모델에 넣는다 (학습 자료에 드물어 손글씨로 보는 일이 잦았다).
- 조각마다 확률·굵기를 출력 폴더의 font_style.json에 둔다 — 식자만 다시·설정 바꾸기는 이 파일만 읽어 몇 초 안에 끝난다.
  대사집(translation_data.json)에는 쓰지 않는다.
"""
import json
import logging
import math
import os
import re

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from src import config
from src.progress import PipelinePhase, ProgressEvent
from src.utils import Letterbox, read_image_bgr, replace_file

logger = logging.getLogger(__name__)

STYLE_FILE = "font_style.json"
_MEASURE_VERSION = 7
_MEASURE_PAD_PX = 8          # 획은 상자를 이만큼 넓힌 조각에서 잰다 (가장자리에 닿는 바탕 덩어리를 가려내려고)
_INNER_INK_MIN_SHARE = 0.3   # 가장자리에 닿지 않는 잉크가 이 비율 이상 남을 때만 바탕 덩어리를 뺀다
_GLYPH_HINT_RATIO = 1.5
_SIZE_PER_GLYPH = 1.18       # 1단계 글자 크기 = 잉크 글자 높이 × 1.18 (src/font_analysis.py)
# 모양 6분류 → 글꼴 설정 키(FONT_MAP)의 여섯 칸: 보통체 → standard, 가는명조 → narration, 손글씨 → handwriting,
# 팝 → pop, 각진 글씨 → angry, 호러 글씨 → scared. 굵은 대사(shouting)는 굵기 판정이 따로 정한다
_SHAPE_TO_FONT_KEY = {"standard": "standard", "mincho": "narration", "handwriting": "handwriting",
                      "pop": "pop", "angular": "angry", "horror": "scared"}
# 가는명조 확률이 이 값 이상일 때만 해설 글꼴로 — 검증 책 실제 조각에서 보통체를 가는명조로 본 비율이
# 0.3% 이하가 되는 가장 낮은 값
MINCHO_MIN_PROB = 0.9206
# 손글씨도 확신할 때만 — 같은 검증 조각에서 보통체를 손글씨로 본 비율이 0.2% 이하가 되는 가장 낮은 값
# 모자라면 보통체(굵기 판정 포함)로
HANDWRITING_MIN_PROB = 0.6
# 검은 바탕을 뒤집어 넣은 흰 글자(채팅·화면 글자)는 손글씨를 거의 확신할 때만 — 뒤집은 조각을 손글씨로 본 것은
# 거의 모두 고딕 화면 글자다
HANDWRITING_MIN_PROB_INVERTED = 0.95
# 팝·각진 글씨·호러도 확신할 때만 — 검증 조각에 이 셋이 너무 적어 재현율을 볼 수 없어
# 하한 0.8로 — 굵은 고딕 대사를 팝으로 낮게(0.6대) 본 것에 속 빈 외곽선 글꼴이 들어가지 않게
SHAPE_MIN_PROB = {"pop": 0.8, "angular": 0.8, "horror": 0.8}
# 책의 보통체 굵기가 두 무리로 깔끔히 갈린다고 볼 간격 — 정규분포 둘의 평균 차가 두 퍼짐의 평균의 이 배 이상
_TWO_GROUPS_SEPARATION = 2.0
_HEAVY_SHARE = (0.08, 0.8)  # 굵은 무리가 이 비율 안일 때만 두 무리 사이를 문턱으로
# 두 무리로 갈리지 않는 책은 보통 굵기의 이 배보다 굵어야 굵게 — 보통 대사체와 중간 굵기 고딕의 굵기가 겹치는 책은
# 가를 빈 자리가 없어, 애매하면 보통으로 둔다
_ONE_GROUP_RATIO = 1.38
_MIN_BASE_ELEMENTS = 15    # 기준을 잡을 보통체 조각이 이보다 적으면 굵기 판정을 하지 않는다
_MIN_PLAIN_BG = 0.9        # 잉크 밖이 흰 바탕인 비율이 이보다 낮으면 톤·그림 위 글씨
_SHORT_CHARS = 3           # 이 글자 수 이하는 굵기를 믿지 않는다
# 작은 글자는 획 폭이 몇 px로 뭉개져 굵기 비율이 부풀려진다 (책 글자 폭 중앙값의 0.7배 미만 조각은
# 보통 크기 조각보다 굵게 재진다 — 작은 말·주석 등)
_MIN_GLYPH_RATIO = 0.7
_MIN_GLYPH_PX = 12
_PUNCT = re.compile(r"[\s・…、。！？!?\-ー―~〜「」『』()（）.,]")
_TRANSFORM = transforms.Compose([Letterbox((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _elements(page):
    return [b.text_element for b in page.speech_bubbles] + list(page.freeform_texts)


def _key(page_name, element):
    return f"{page_name}|{','.join(str(int(round(v))) for v in element.text_box[:4])}"


def _stamp():
    path = config.FONT_STYLE6_MODEL_PATH
    return {"model": os.path.basename(path), "model_mtime": int(os.path.getmtime(path)) if os.path.exists(path) else 0,
            "measure": _MEASURE_VERSION}


# ── 굵기 측정 ──

def _thin(ink):
    """Zhang-Suen 뼈대."""
    image = ink.astype(np.uint8).copy()
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(image, 1)
            p2, p3, p4, p5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            p6, p7, p8, p9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = sum(s.astype(np.int16) for s in seq[:8])
            transitions = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            c1, c2 = (p2 * p4 * p6, p4 * p6 * p8) if step == 0 else (p2 * p4 * p8, p2 * p6 * p8)
            remove = ((image == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
                      & (c1 == 0) & (c2 == 0))
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            return image.astype(bool)


def _runs(profile):
    runs, start = [], None
    for i, value in enumerate(list(profile) + [False]):
        if value and start is None:
            start = i
        if not value and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _darkness_gray(pil):
    """RGB 세 값 가운데 가장 작은 값 — 보라·빨강 같은 색 잉크도 검정처럼 어둡게 잡힌다 (흑백 그림은 밝기와 같다)."""
    return np.asarray(pil.convert("RGB")).min(axis=2)


def _ink(gray):
    """오쓰 문턱으로 가른 잉크와, 밝은 쪽을 잉크로 뒤집었는가 (검은 바탕 흰 글자)."""
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = gray <= threshold  # 순수 흑백이면 문턱이 0이라 '<'로는 잉크가 하나도 안 남는다
    flipped = bool(ink.mean() > 0.5)
    return (~ink if flipped else ink), flipped


def _run_bounds(ink, axis):
    """잉크 칸마다 그 칸이 든 연속 잉크 구간의 [시작, 끝) — axis=1 가로, 0 세로 (잉크 밖 칸의 값은 쓰지 않는다)."""
    a = ink if axis == 1 else ink.T
    index = np.arange(a.shape[1])
    first = a & ~np.pad(a, ((0, 0), (1, 0)))[:, :-1]
    last = a & ~np.pad(a, ((0, 0), (0, 1)))[:, 1:]
    start = np.maximum.accumulate(np.where(first, index, -1), axis=1)
    end = np.minimum.accumulate(np.where(last, index + 1, a.shape[1] + 1)[:, ::-1], axis=1)[:, ::-1]
    return (start, end) if axis == 1 else (start.T, end.T)


def _stroke_thickness(ink, gray, flipped):
    """획 두께(px) — 뼈대 칸마다 가로·세로 단면의 잉크 양 가운데 작은 쪽을, 위아래 10%씩 잘라내고 평균. 못 재면 None.

    단면은 잉크 연속 구간을 양쪽으로 1칸씩 넓혀 반쯤 걸친 칸도 그만큼 센다. 잉크 양은 바탕보다 어두운 정도를 완전한
    검정 기준으로 더한다 — 흐려서 퍼진 획도 잉크 총량은 같아 두께가 그대로다."""
    skeleton = _thin(ink)
    if skeleton.sum() < 10:
        return None
    g = gray.astype(np.float32)
    if flipped:
        g = 255 - g
    paper = float(np.percentile(g[~ink], 75)) if (~ink).any() else 255.0
    dark = np.clip((paper - g) / max(paper, 1.0), 0, 1)
    h, w = ink.shape
    row_start, row_end = _run_bounds(ink, 1)
    col_start, col_end = _run_bounds(ink, 0)
    ys, xs = np.nonzero(skeleton)
    down = np.vstack([np.zeros((1, w), np.float32), np.cumsum(dark, axis=0)])
    across = np.hstack([np.zeros((h, 1), np.float32), np.cumsum(dark, axis=1)])
    top, bottom = np.maximum(col_start[ys, xs] - 1, 0), np.minimum(col_end[ys, xs] + 1, h)
    left, right = np.maximum(row_start[ys, xs] - 1, 0), np.minimum(row_end[ys, xs] + 1, w)
    thickness = np.sort(np.minimum(down[bottom, xs] - down[top, xs], across[ys, right] - across[ys, left]))
    cut = int(len(thickness) * 0.1)
    kept = thickness[cut:len(thickness) - cut]
    return float(kept.mean() if len(kept) else thickness.mean())


def measure_weight(crop_pil, padded_pil=None, glyph_hint=None):
    """(굵기, 흰 바탕 위 글씨인가, 글자 폭 px). 굵기 = 획 두께(_stroke_thickness) ÷ 글자 폭, 못 재면 None.

    padded_pil(상자를 몇 px 넓힌 조각)이 오면 획은 거기서 재고, 조각 가장자리에 닿는 잉크 덩어리(톤·집중선·그림 바탕)는
    뺀다 — 테두리 두른 글자는 테두리가 글자를 바탕과 떼어 놓는다. glyph_hint(1단계가 잰 글자 높이)가 오면 글자 폭이
    그 1.5배를 넘을 때(괄호·후리가나·바탕이 한 덩어리로 잡힘) 그 값을 글자 폭으로 쓴다.
    """
    from src.font_analysis import strip_furigana_column
    gray = _darkness_gray(strip_furigana_column(crop_pil))
    ink, flipped = _ink(gray)
    white_bg = float((gray[~ink] > 200).mean()) if (~ink).any() else 0.0
    plain = not flipped and white_bg >= _MIN_PLAIN_BG
    if padded_pil is not None:
        gray = _darkness_gray(strip_furigana_column(padded_pil))
        ink, flipped = _ink(gray)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        h, w = ink.shape
        inner = np.zeros(count, bool)
        for i in range(1, count):
            x, y, bw, bh, _ = stats[i]
            inner[i] = x > 0 and y > 0 and x + bw < w and y + bh < h
        kept = inner[labels]
        if kept.sum() >= _INNER_INK_MIN_SHARE * ink.sum():
            ink = kept
    thickness = _stroke_thickness(ink, gray, flipped) if ink.sum() >= 30 else None
    if gray.shape[0] < gray.shape[1]:
        ink = ink.T  # 가로쓰기는 돌려서 줄을 열로 본다
    columns = _runs(ink.any(axis=0))
    if not columns or ink.sum() < 30:
        return None, plain, None
    widest = max(b - a for a, b in columns)
    glyph = float(np.median([b - a for a, b in columns if b - a >= 0.62 * widest]))
    if glyph_hint and glyph > _GLYPH_HINT_RATIO * glyph_hint:
        glyph = float(glyph_hint)
    if thickness is None or glyph <= 0:
        return None, plain, glyph
    return thickness / glyph, plain, glyph


_BORDER_RATIO = 0.05       # 바탕 밝기를 볼 조각 테두리 띠 (바깥 5%)
_DARK_BG = 110             # 테두리 밝기 중앙값이 이보다 어두우면 검은 바탕
_LIGHT_BG = 200            # 이 사이면 회색 바탕 — 글자가 바탕보다 밝을 때만 뒤집는다


def to_dark_on_light(crop_pil):
    """검은·회색 바탕의 흰 글자를 흰 바탕 검은 글자로 뒤집는다 (모양 모델은 검은 바탕 글자를 거의 못 봤다).

    (뒤집은 조각, 뒤집었는가). 흰 바탕이거나 글자가 바탕보다 어두우면 그대로 둔다.
    """
    gray = np.asarray(crop_pil.convert("L"))
    h, w = gray.shape
    band = max(1, int(round(min(h, w) * _BORDER_RATIO)))
    border = np.concatenate([gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()])
    background = float(np.median(border))
    invert = background < _DARK_BG
    if not invert and background < _LIGHT_BG:
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright, dark = gray[gray > threshold], gray[gray <= threshold]
        if bright.size and dark.size:
            # 바탕에서 더 먼 쪽이 글자 — 그쪽이 밝으면 흰 글자
            invert = abs(float(bright.mean()) - background) > abs(float(dark.mean()) - background)
    if not invert:
        return crop_pil, False
    return Image.fromarray(255 - np.asarray(crop_pil.convert("RGB"))), True


# ── 책 전체 측정 ──

def _measure(book_pages, input_dir):
    """책 전체 글자 조각의 {열쇠: {p: 분류 확률, w: 굵기, plain: 흰 바탕, short: 3자 이하}}와 분류 이름."""
    from src.font_analysis import _cuda_autocast_context, _resolve_style_name
    from src.model_loader import _load_font_checkpoint

    # 번역 제외된 글자도 잰다 — 책 기준(굵기·글자 폭)이 그 실행의 번역 결과('번역 불가' 답)에 따라 흔들리지 않게
    wanted = [(page, element) for page in book_pages for element in _elements(page)]
    if not wanted:
        return {}, []
    model = _load_font_checkpoint(config.FONT_STYLE6_MODEL_PATH, "style6")
    if model is None:
        return {}, []
    names = [_resolve_style_name(model, i) for i in range(len(model.style_mapping))]
    device = next(model.parameters()).device
    keys, tensors, records, image_cache = [], [], {}, {}
    for page, element in wanted:
        image = page.image_rgb
        if image is None:
            if page.source_page not in image_cache:
                image_cache = {page.source_page: read_image_bgr(os.path.join(input_dir, page.source_page))}
            bgr = image_cache[page.source_page]
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        if image is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in element.text_box[:4])
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0 or min(crop.shape[:2]) < 4:
            continue
        crop_pil = Image.fromarray(crop)
        key = _key(page.source_page, element)
        pad = _MEASURE_PAD_PX
        padded = Image.fromarray(image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad])
        hint = element.font_size / _SIZE_PER_GLYPH if element.font_size else None
        weight, plain, glyph = measure_weight(crop_pil, padded, hint)
        records[key] = {"w": None if weight is None else round(weight, 5), "plain": plain, "g": glyph,
                        "short": len(_PUNCT.sub("", element.original_text or "")) <= _SHORT_CHARS}
        keys.append(key)
        # 학습 때처럼 원래 조각 그대로 (후리가나 열을 떼지 않는다) — 검은·회색 바탕 흰 글자만 뒤집는다
        model_input, records[key]["inv"] = to_dark_on_light(crop_pil)
        tensors.append(_TRANSFORM(model_input.convert("RGB")))
    batch_size = max(1, int(config.FONT_MODEL_BATCH_SIZE))  # 메모리 아끼기(LOW_VRAM_MODE)면 절반
    with torch.inference_mode():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size]).to(device)
            with _cuda_autocast_context():
                probs = torch.softmax(model(batch)["style"].float(), dim=1).cpu().numpy()
            for key, row in zip(keys[i:i + batch_size], probs):
                records[key]["p"] = [round(float(v), 4) for v in row]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {k: r for k, r in records.items() if "p" in r}, names


def load_or_measure(book_pages, output_dir, input_dir, callback=None):
    """출력 폴더의 font_style.json을 읽는다. 없거나 책·모델이 바뀌었으면 책 전체로 다시 재서 저장한다."""
    path = os.path.join(output_dir, STYLE_FILE)
    wanted = {_key(page.source_page, element) for page in book_pages for element in _elements(page)}
    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("stamp") == _stamp() and wanted <= set(saved.get("elements", {})) | set(saved.get("skipped", [])):
            return saved
    except (OSError, ValueError):
        pass
    if callback:
        callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 0, len(book_pages), "책의 글씨체를 판정하는 중",
                               extras={"base_font": True}))
    elements, names = _measure(book_pages, input_dir)
    record = {"stamp": _stamp(), "classes": names, "elements": elements, "skipped": sorted(wanted - set(elements))}
    if elements:
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
            replace_file(tmp, path)  # Windows에서 누가 잠깐 열어 두었으면 기다렸다 다시
        except OSError as exc:
            logger.warning(f"글씨체 판정 기록을 저장하지 못했습니다: {exc}")
    if callback:
        callback(ProgressEvent(PipelinePhase.LOADING_MODELS, len(book_pages), len(book_pages),
                               f"책의 글씨체를 판정했습니다 — 조각 {len(elements)}곳", extras={"base_font": True}))
    return record


def _weighable(r, min_glyph):
    """책 기준을 잡는 데 쓸 조각 — 흰 바탕, 4자 이상, 작은 글자가 아님."""
    return r["w"] is not None and r["plain"] and not r["short"] and (r.get("g") or 0) >= min_glyph


def _judgeable(r, min_glyph):
    """굵기를 판정할 조각 — 짧은 외침과 톤 위 글씨도 판정한다. 기준에서만 뺀다.

    검은 바탕을 뒤집어 넣은 흰 글자와 작은 글자는 획 폭을 믿을 수 없어 판정하지 않는다.
    """
    return r["w"] is not None and not r.get("inv") and (r.get("g") or 0) >= min_glyph


def _two_groups_apart(log_weights, iterations=200):
    """굵기 로그값에 정규분포 둘을 맞춰(EM) 두 무리로 깔끔히 갈리는지 — 평균 차가 두 퍼짐의 평균의
    _TWO_GROUPS_SEPARATION배 이상인가. 한 무리에 긴 꼬리가 붙은 분포는 갈리지 않는다."""
    v = log_weights
    mean = np.percentile(v, [25, 85])
    spread = np.full(2, v.std() * 0.6) + 1e-4
    weight = np.array([0.6, 0.4])
    for _ in range(iterations):
        density = weight * np.exp(-0.5 * ((v[:, None] - mean) / spread) ** 2) / spread
        share = density / np.maximum(density.sum(axis=1, keepdims=True), 1e-300)
        count = share.sum(axis=0) + 1e-9
        weight, mean = count / len(v), (share * v[:, None]).sum(axis=0) / count
        spread = np.sqrt((share * (v[:, None] - mean) ** 2).sum(axis=0) / count) + 1e-4
    return abs(mean[1] - mean[0]) / np.sqrt((spread[0] ** 2 + spread[1] ** 2) / 2) >= _TWO_GROUPS_SEPARATION


def _bold_rule(record, names):
    """책 기준 (굵은 보통체 문턱, 굵기를 잴 최소 글자 폭). 기준 조각이 모자라면 None.

    보통체 조각 굵기의 로그값이 두 무리로 깔끔히 갈리고 굵은 무리가 _HEAVY_SHARE 안이면, 둘로 가를 때 무리 사이 차이가
    가장 커지는 자리(오쓰 방식)의 두 끝값 사이가 문턱이다. 아니면 가는 쪽 절반의 가운데 값 × _ONE_GROUP_RATIO.
    """
    standard = names.index("standard") if "standard" in names else None
    if standard is None:
        return None
    base = [r for r in record["elements"].values() if int(np.argmax(r["p"])) == standard and _weighable(r, 0)]
    if not base:
        return None
    min_glyph = max(_MIN_GLYPH_PX, _MIN_GLYPH_RATIO * float(np.median([r["g"] for r in base])))
    weights = [r["w"] for r in base if _weighable(r, min_glyph)]
    if len(weights) < _MIN_BASE_ELEMENTS:
        return None
    v = np.sort(np.log(weights))
    n = len(v)
    total = np.cumsum(v)
    i = np.arange(1, n)
    cut = int(i[np.argmax(i * (n - i) * (total[:-1] / i - (total[-1] - total[:-1]) / (n - i)) ** 2)])
    if _two_groups_apart(v) and _HEAVY_SHARE[0] <= (n - cut) / n <= _HEAVY_SHARE[1]:
        return float(np.exp((v[cut - 1] + v[cut]) / 2)), min_glyph
    return _ONE_GROUP_RATIO * float(np.exp(np.median(v[: max(1, n // 2)]))), min_glyph


def _choose(r, names, bold_rule, narration_as_standard):
    """조각 하나의 (글꼴 설정 키, 이유)."""
    probs = r["p"]
    shape = names[int(np.argmax(probs))]
    reason = "shape_model"
    handwriting_floor = HANDWRITING_MIN_PROB_INVERTED if r.get("inv") else HANDWRITING_MIN_PROB
    if shape == "handwriting" and probs[names.index("handwriting")] < handwriting_floor:
        shape, reason = "standard", "handwriting_threshold"
    if shape in SHAPE_MIN_PROB and probs[names.index(shape)] < SHAPE_MIN_PROB[shape]:
        # 장식 글씨로 볼 만큼 확신하진 않아도 굵은 제목·외침 글씨인 것은 맞다 — 가는 보통체로 떨어지지 않게 굵은 보통체로
        return "shouting", "shape_threshold_bold"
    if shape == "mincho":
        if narration_as_standard:
            shape, reason = "standard", "mincho_setting"
        elif probs[names.index("mincho")] < MINCHO_MIN_PROB:
            shape, reason = "standard", "mincho_threshold"
    if shape == "standard" and bold_rule is not None and _judgeable(r, bold_rule[1]) and r["w"] > bold_rule[0]:
        return "shouting", "bold_weight"
    return _SHAPE_TO_FONT_KEY.get(shape, "standard"), reason


def apply_book_styles(book_pages, target_pages, output_dir, input_dir, callback=None):
    """식자할 쪽(target_pages)의 글씨체를 책 단위 판정으로 정한다. 판정 모델이 없으면 1단계 판정을 그대로 둔다.

    바꾼 수를 돌려준다. 대사 객체만 메모리에서 고치고 대사집 파일에는 쓰지 않는다.
    """
    record = load_or_measure(book_pages, output_dir, input_dir, callback)
    names = record.get("classes") or []
    elements = record.get("elements") or {}
    if not elements or not names:
        logger.warning("글씨체 판정 모델이 없어 1단계 판정을 그대로 씁니다")
        return 0
    # 굵기 기준은 지금 책의 조각으로만 — 기록에 남은 지운 쪽·옛 좌표 조각이 기준을 흔들지 않게
    current = {_key(page.source_page, element) for page in book_pages for element in _elements(page)}
    bold_rule = _bold_rule({"elements": {k: r for k, r in elements.items() if k in current}}, names)
    narration_as_standard = bool(config.NARRATION_AS_STANDARD)
    changed = 0
    from src.text_filters import dialogue_size
    book_size = dialogue_size([b for page in book_pages for b in page.speech_bubbles])
    for page in target_pages:
        page.book_dialogue_px = book_size  # 말풍선이 적은 쪽의 대사 크기 기준 (식자가 쓴다)
        loose = {id(text) for text in page.freeform_texts}
        for element in _elements(page):
            r = elements.get(_key(page.source_page, element))
            if r is None:
                continue
            style, reason = _choose(r, names, bold_rule, narration_as_standard)
            if style == "standard" and id(element) in loose and not narration_as_standard and not r.get("inv"):
                style = "narration"  # 말풍선 밖 평문(해설·속말)은 해설 글꼴로 — 어두운 바탕의 흰 글자(화면·채팅 글자)는 빼고
            order = np.argsort(r["p"])[::-1][:3]
            element.font_style_raw = names[int(order[0])]
            element.font_style_scores = {names[int(i)]: r["p"][int(i)] for i in order}
            element.font_style_reason = reason
            element.style_inverted = bool(r.get("inv"))  # 어두운 바탕을 뒤집어 잰 흰 글자 (채팅 화면 줄 묶음에 쓴다)
            # 원문 획 굵기 — 짧은 조각은 획이 적어 튀므로 없음으로 (맞닿은 말풍선 굵기 맞추기가 쓴다)
            element.stroke_weight = r.get("w") if not r.get("short") else None
            if element.font_style != style:
                element.font_style = style
                changed += 1
    logger.info("책 단위 글씨체 판정: %d곳을 바꿈 (굵은 보통체 문턱 %s)", changed,
                "없음" if bold_rule is None else f"{bold_rule[0]:.4f}, 글자 폭 {bold_rule[1]:.0f}px 이상만")
    if not narration_as_standard:
        _caption_boxes_in_narration(target_pages, input_dir)
    _match_visible_size(target_pages, elements)
    _raise_tiny_sizes(book_pages, target_pages, elements)
    return changed


def _caption_boxes_in_narration(pages, input_dir):
    """꼬리 없는 네모 해설 상자에 든 보통체 대사를 해설 글꼴로 쓴다. 바꾼 수를 돌려준다."""
    from src.page_drawer import is_caption_box
    changed = 0
    for page in pages:
        candidates = [b for b in page.speech_bubbles
                      if b.text_element.font_style == "standard" and (b.text_element.translated_text or "").strip()]
        image = read_image_bgr(os.path.join(input_dir, page.source_page)) if candidates else None
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for bubble in candidates:
            if is_caption_box(gray, bubble.bubble_box, bubble.text_element.text_box):
                bubble.text_element.font_style = "narration"
                changed += 1
    if changed:
        logger.info("네모 해설 상자 %d곳을 해설 글꼴로 씁니다", changed)
    return changed


# 1단계 글자 크기(잉크 높이 실측)가 원본 글자 폭보다 한참 작게 나오는 세로 대사가 있다(한 글자 대사 등) — 말풍선에
# 아주 작게 그려진다. '크기 ÷ 글자 폭'은 대개 0.8 이상이라,
# 0.65 미만이면 잘못 잰 것으로 보고 책 중앙값 × 글자 폭까지 올린다. 세로 상자(원본 세로쓰기)의 말풍선 대사만.
_TINY_SIZE_TO_GLYPH = 0.65
_TINY_MAX_STACK = 1.3  # 올린 크기로 열마다 쌓은 높이가 원문 상자 높이의 이 배를 넘으면 잘못 잰 글자 폭


# 글꼴마다 같은 크기에서 한글이 차지하는 높이가 다르다 (바른바탕 0.89em, 나눔펜 0.71em) — 1단계 글자 크기는 원문 글자
# 높이 × 1.18이라 em을 덜 채우는 글꼴은 원문보다 작아 보였다. 원문 글자 폭(세로쓰기 열 폭 = 보이는 글자 크기)에서
# 한국어 크기 = 원문 글자 폭 × 0.95 ÷ (그 글꼴의 한글 높이 비)로 맞춘다. 원문 글자 폭을 잘못 잰 조각에 휘둘리지 않게
# 바꾸는 폭은 지금 크기의 1.0~1.3배로 묶는다 (키우기만 한다). 이 뒤의 맞추기(말풍선 여백·줄바꿈·하한)는 그대로라 안 들어가면 줄어든다
_VISIBLE_FILL = 0.95
_VISIBLE_CHANGE = (1.0, 1.3)  # 줄이지는 않는다 — 재 보니 대부분의 조각은 이미 원문 글자 폭만큼 보였다
_VISIBLE_SAMPLE = "한글식자기준본문"
_BODY_RATIO_CACHE = {}


def _hangul_body_ratio(style_name):
    """그 글씨체 글꼴로 한글을 그렸을 때 잉크 높이 ÷ 글자 크기 (글꼴 파일·글씨체마다 한 번 잰다)."""
    from src.text_renderer import DEFAULT_STYLES, measure_character_body_size
    font_path = config.FONT_MAP.get(style_name, config.DEFAULT_FONT_PATH)
    # 같은 가족 굵은 파일이 없으면 굵은 대사도 평범한 대사 글꼴(합성 굵게)이라 파일만으로는 못 가른다
    key = (font_path, style_name)
    if key not in _BODY_RATIO_CACHE:
        from dataclasses import replace
        from src.text_layout import needs_synthetic_bold
        style = DEFAULT_STYLES.get(style_name, DEFAULT_STYLES["standard"])
        if style_name == "shouting" and not needs_synthetic_bold():
            style = replace(style, embolden=False)  # 그릴 때와 같게 — 같은 가족 굵은 파일이면 합성 굵게를 겹치지 않는다
        try:
            _, body = measure_character_body_size(_VISIBLE_SAMPLE, font_path, 100, style)
            _BODY_RATIO_CACHE[key] = body / 100.0 if body > 0 else None
        except Exception:  # noqa: BLE001 — 글꼴을 못 읽으면 보정하지 않는다
            _BODY_RATIO_CACHE[key] = None
    return _BODY_RATIO_CACHE[key]


def _match_visible_size(target_pages, elements):
    """식자할 글자 크기를 원문 글자 폭과 글꼴의 한글 높이에 맞춘다. 바꾼 수를 돌려준다."""
    changed = 0
    for page in target_pages:
        for element in _elements(page):
            r = elements.get(_key(page.source_page, element))
            g = r.get("g") if r else None
            ratio = _hangul_body_ratio(element.font_style)
            if not g or g < _MIN_GLYPH_PX or not ratio or not element.font_size:
                continue
            wanted = _VISIBLE_FILL * g / ratio
            # 식자 키우기 상한의 기준으로 쓴다 (page_drawer._reference_size) — 실행 중 값, 대사집에는 쓰지 않는다
            element.visible_size = wanted
            low, high = _VISIBLE_CHANGE
            new_size = int(round(min(high * element.font_size, max(low * element.font_size, wanted))))
            if new_size == element.font_size:
                continue
            if element.font_char_ratio:
                element.font_char_ratio *= new_size / element.font_size
            element.font_size = new_size
            changed += 1
    if changed:
        logger.info("글꼴의 한글 높이에 맞춰 글자 크기 %d곳을 원문 글자 폭 기준으로 고침", changed)
    return changed


def _raise_tiny_sizes(book_pages, target_pages, elements):
    """식자할 쪽 말풍선 대사의 과소 측정 글자 크기를 책 기준으로 올린다. 올린 수를 돌려준다."""
    def glyph(page, element):
        r = elements.get(_key(page.source_page, element))
        g = r.get("g") if r else None
        x1, y1, x2, y2 = element.text_box[:4]
        return g if g and g >= _MIN_GLYPH_PX and (y2 - y1) >= (x2 - x1) else None

    ratios = [b.text_element.font_size / g for page in book_pages for b in page.speech_bubbles
              if b.text_element.font_size and (g := glyph(page, b.text_element))]
    if len(ratios) < _MIN_BASE_ELEMENTS:
        return 0
    typical = float(np.median(ratios))
    raised = 0
    for page in target_pages:
        for bubble in page.speech_bubbles:
            element = bubble.text_element
            g = glyph(page, element)
            if not g or not element.font_size or element.font_size >= _TINY_SIZE_TO_GLYPH * g:
                continue
            new_size = int(round(typical * g))
            # 글자 폭을 여러 열이 붙은 덩어리로 잘못 잰 경우는 올리지 않는다 — 올린 크기로 원문 글자 수를 열마다
            # 쌓으면 원문 상자 높이를 한참 넘는다
            x1, y1, x2, y2 = element.text_box[:4]
            columns = max(1, int(round((x2 - x1) / g)))
            chars = len(_PUNCT.sub("", element.original_text or ""))
            if math.ceil(chars / columns) * new_size > _TINY_MAX_STACK * (y2 - y1):
                continue
            if element.font_char_ratio:
                element.font_char_ratio *= new_size / element.font_size  # 글자 비율 맞춤도 같은 크기를 겨누게
            element.font_size = new_size
            raised += 1
    raised_free = _raise_tiny_lines(target_pages)
    if raised:
        logger.info("원본 글자 폭보다 한참 작게 잰 글자 크기 %d곳을 책 기준(크기 ÷ 글자 폭 %.2f)까지 올림", raised, typical)
    return raised + raised_free


# 가로로 긴 한 줄 상자의 말풍선 밖 글자(속 빈 외곽선 글자·톤 위 흰 글자)는 1단계가 획만 재서 크기를 한참 작게 잡는다
# 원문 한 글자 칸 = min(상자 높이, 상자 폭 ÷ 글자 수)의 이 배보다 작으면
# 그 칸의 _LINE_FILL배로 올린다 — 여러 줄 상자는 글자 수로 나눈 칸이 작게 나와 올라가지 않는다
_LINE_MIN_ASPECT = 2.5
_LINE_TINY = 0.6
_LINE_FILL = 0.85


def _raise_tiny_lines(target_pages):
    raised = 0
    for page in target_pages:
        for element in page.freeform_texts:
            x1, y1, x2, y2 = element.text_box[:4]
            chars = len(_PUNCT.sub("", element.original_text or ""))
            if chars < 2 or not element.font_size or (x2 - x1) < _LINE_MIN_ASPECT * (y2 - y1):
                continue
            cell = min(y2 - y1, (x2 - x1) / chars)
            if element.font_size >= _LINE_TINY * cell:
                continue
            new_size = int(round(_LINE_FILL * cell))
            if element.font_char_ratio:
                element.font_char_ratio *= new_size / element.font_size
            element.font_size = new_size
            raised += 1
    if raised:
        logger.info("가로 한 줄 상자에서 한참 작게 잰 글자 크기 %d곳을 상자 칸 크기로 올림", raised)
    return raised

