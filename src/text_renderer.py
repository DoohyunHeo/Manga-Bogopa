"""
skia-python 기반 텍스트 렌더링 엔진.
만화 식자 품질의 텍스트 렌더링을 제공합니다.
- 자간 (letter spacing)
- 평체 (horizontal scale, 가로 90% = 세로로 길쭉)
- 행간 (line spacing)
- 스트로크 (외곽선) — 전체 줄 외곽선 → 전체 줄 본체 2-pass로 줄 간 침범 없음
- 글리프 폴백 — 지정 폰트에 없는 문자(♪★♡ 등)는 다른 폰트로 자동 대체
- 세로쓰기 다단(multi-column) — 한 단에 안 들어가면 우→좌로 단을 나눔
- 서브픽셀 안티앨리어싱

Adobe Photoshop과 동일한 HarfBuzz 텍스트 엔진 기반.
"""
import functools
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import skia
from PIL import Image

from src import config

logger = logging.getLogger(__name__)

# 한국어 조판 금칙 (text_wrapping에서 re-export하여 사용)
# 행두 금지: 닫는 괄호류·문장 부호·말줄임표·장식 기호는 줄(또는 세로 단)의 머리에 올 수 없다.
LINE_HEAD_FORBIDDEN = frozenset(")]}〉》」』】〕’”.。,，、!！?？…⋯‥·:;~〜‼⁉♪♬♡♥☆★―—–")
# 행말 금지: 여는 괄호류는 줄 끝에 올 수 없다.
LINE_TAIL_FORBIDDEN = frozenset("([{〈《「『【〔‘“")

# In vertical text, these characters are drawn rotated 90° clockwise so they
# read along the vertical flow. Multi-char tokens (e.g. "!?") are NOT rotated.
VERTICAL_ROTATE_CHARS = frozenset({
    # Dashes and wave marks
    '~', '〜', '-', '—', '─', 'ー', '−', '–',
    # Parentheses and brackets (halfwidth + fullwidth + CJK)
    '(', ')', '[', ']', '{', '}',
    '(', ')', '〈', '〉', '《', '》',
    '「', '」', '『', '』', '【', '】', '〔', '〕',
    # Ellipsis (horizontal three dots → vertical three dots after rotation)
    '⋯', '…',
})

# 세로쓰기에서 공백 한 칸이 차지하는 높이 (글자 크기 대비)
_VERTICAL_SPACE_RATIO = 0.45
# 세로쓰기 단(컬럼) 사이 간격 (글자 크기 대비)
_VERTICAL_COLUMN_GAP_RATIO = 0.18

# 지정 폰트에 글리프가 없을 때 시도할 시스템 폰트 (음표·하트·별 등 기호 보강)
_FALLBACK_SYSTEM_FAMILIES = ("Malgun Gothic", "Segoe UI Symbol", "Yu Gothic", "MS Gothic")


@dataclass
class TextStyle:
    """텍스트 스타일 프리셋."""
    color: Tuple[int, int, int] = (0, 0, 0)
    letter_spacing: float = -0.6       # px, 음수=좁게
    horizontal_scale: float = 0.94     # 장평. 1.0보다 작을수록 세로로 길쭉한 인상
    line_spacing: float = 1.18         # 행간 배율
    stroke_width: float = 0.0          # 외곽선 두께 (0=없음)
    stroke_color: Tuple[int, int, int] = (255, 255, 255)
    embolden: bool = False
    oversample_scale: int = 2
    subpixel: bool = True
    baseline_snap: bool = True
    linear_metrics: bool = True
    hinting: str = "full"


# 스타일별 프리셋 (만화 식자 표준)
DEFAULT_STYLES: Dict[str, TextStyle] = {
    # 일반 대사 굵기는 의도된 디자인 — 임의로 굵게 바꾸지 말 것
    "standard": TextStyle(color=(0, 0, 0), letter_spacing=-0.6, horizontal_scale=0.94, line_spacing=1.18),
    "shouting": TextStyle(
        color=(0, 0, 0), letter_spacing=-0.35, horizontal_scale=0.90, line_spacing=1.08,
        embolden=True, oversample_scale=3
    ),
    "cute": TextStyle(color=(10, 10, 10), letter_spacing=-0.2, horizontal_scale=0.96, line_spacing=1.16),
    "narration": TextStyle(color=(12, 12, 12), letter_spacing=-0.15, horizontal_scale=0.96, line_spacing=1.22),
    "handwriting": TextStyle(color=(10, 10, 10), letter_spacing=0.0, horizontal_scale=0.98, line_spacing=1.12),
    "pop": TextStyle(
        color=(0, 0, 0), letter_spacing=-0.4, horizontal_scale=0.91, line_spacing=1.05,
        embolden=True, oversample_scale=3
    ),
    "angry": TextStyle(
        color=(0, 0, 0), letter_spacing=-0.35, horizontal_scale=0.89, line_spacing=1.06,
        embolden=True, oversample_scale=3
    ),
    "scared": TextStyle(color=(0, 0, 0), letter_spacing=-0.1, horizontal_scale=0.98, line_spacing=1.12),
    "embarrassment": TextStyle(color=(18, 18, 18), letter_spacing=-0.1, horizontal_scale=0.98, line_spacing=1.12),
}

# 말풍선 밖 텍스트 기본 스타일 (외곽선 포함)
FREEFORM_STYLE = TextStyle(
    color=(0, 0, 0), letter_spacing=-0.25, horizontal_scale=0.94,
    line_spacing=1.08, stroke_width=2.0, stroke_color=(255, 255, 255),
    embolden=True, oversample_scale=3,
)


@functools.lru_cache(maxsize=64)
def _load_typeface(font_path: str) -> skia.Typeface:
    """폰트 로드 (캐시됨). "system:<패밀리명>" 형식은 시스템 설치 폰트를 사용."""
    if font_path.startswith("system:"):
        return skia.Typeface(font_path[len("system:"):])
    tf = skia.Typeface.MakeFromFile(font_path)
    if tf is None:
        logger.warning(f"폰트 로드 실패: {font_path}, 기본 폰트 사용")
        tf = skia.Typeface()
    return tf


@functools.lru_cache(maxsize=2048)
def _has_glyph(font_path: str, char: str) -> bool:
    """폰트가 특정 문자의 글리프를 가지고 있는지 확인합니다 (캐시됨)."""
    typeface = _load_typeface(font_path)
    font = skia.Font(typeface, 12)
    glyphs = font.textToGlyphs(char)
    return len(glyphs) > 0 and glyphs[0] != 0


def _fallback_font_paths(primary_path: str) -> Tuple[str, ...]:
    """글리프 폴백 후보: FONT_MAP의 다른 폰트 → 시스템 기호 폰트 순."""
    candidates: List[str] = []
    font_map = getattr(config, "FONT_MAP", {}) or {}
    for key in ("standard", "narration"):
        path = font_map.get(key)
        if path:
            candidates.append(path)
    candidates.extend(path for path in font_map.values() if path)
    candidates.extend(f"system:{family}" for family in _FALLBACK_SYSTEM_FAMILIES)

    seen = {primary_path}
    ordered = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            ordered.append(cand)
    return tuple(ordered)


@functools.lru_cache(maxsize=4096)
def _resolve_char_font(char: str, primary_path: str) -> str:
    """문자를 그릴 폰트를 결정합니다. 지정 폰트 우선, 없으면 폴백 체인."""
    if char.isspace() or _has_glyph(primary_path, char):
        return primary_path
    for cand in _fallback_font_paths(primary_path):
        if _has_glyph(cand, char):
            return cand
    return primary_path


def _any_font_has_glyph(char: str, primary_path: str) -> bool:
    return _resolve_char_font(char, primary_path) != primary_path or _has_glyph(primary_path, char)


def _style_cache_key(style: TextStyle) -> Tuple:
    """Create a hashable key for style-aware caches."""
    if style is None:
        style = DEFAULT_STYLES["standard"]
    return (
        style.color,
        style.letter_spacing,
        style.horizontal_scale,
        style.line_spacing,
        style.stroke_width,
        style.stroke_color,
        style.embolden,
        style.oversample_scale,
        style.subpixel,
        style.baseline_snap,
        style.linear_metrics,
        style.hinting,
    )


def _style_from_cache_key(style_key: Tuple) -> TextStyle:
    return TextStyle(
        color=style_key[0],
        letter_spacing=style_key[1],
        horizontal_scale=style_key[2],
        line_spacing=style_key[3],
        stroke_width=style_key[4],
        stroke_color=style_key[5],
        embolden=style_key[6],
        oversample_scale=style_key[7],
        subpixel=style_key[8],
        baseline_snap=style_key[9],
        linear_metrics=style_key[10],
        hinting=style_key[11],
    )


@functools.lru_cache(maxsize=256)
def _make_font_cached(font_path: str, font_size: int, style_key: Tuple) -> skia.Font:
    """Return a cached skia.Font for measurement and rendering."""
    style = _style_from_cache_key(style_key)
    typeface = _load_typeface(font_path)
    font = skia.Font(typeface, font_size)
    font.setScaleX(style.horizontal_scale)
    font.setSubpixel(style.subpixel)
    font.setBaselineSnap(style.baseline_snap)
    font.setLinearMetrics(style.linear_metrics)
    font.setEmbolden(style.embolden)
    font.setEdging(
        skia.Font.Edging.kSubpixelAntiAlias
        if style.subpixel else skia.Font.Edging.kAntiAlias
    )
    hinting = {
        "none": skia.FontHinting.kNone,
        "slight": skia.FontHinting.kSlight,
        "normal": skia.FontHinting.kNormal,
        "full": skia.FontHinting.kFull,
    }.get(style.hinting, skia.FontHinting.kFull)
    font.setHinting(hinting)
    return font


def replace_unsupported_chars(text: str, font_path: str) -> str:
    """어느 폰트로도 그릴 수 없는 특수문자만 대체합니다.
    ⋯(U+22EF) → ・・・(U+30FB) → ...(ASCII) 순으로 폴백.
    (지정 폰트에 없어도 폴백 폰트가 그릴 수 있으면 그대로 둔다.)"""
    if "⋯" in text and not _any_font_has_glyph("⋯", font_path):
        if _any_font_has_glyph("・", font_path):
            text = text.replace("⋯", "・・・")
        else:
            text = text.replace("⋯", "...")
    if "︙" in text and not _any_font_has_glyph("︙", font_path):
        text = text.replace("︙", "⋮" if _any_font_has_glyph("⋮", font_path) else ":")
    return text


@functools.lru_cache(maxsize=8192)
def _compute_line_runs_cached(line: str, primary_font_path: str, font_size: int, style_key: Tuple):
    """한 줄을 폰트 폴백 런(run) 단위로 분해해 글리프·x좌표·총폭을 계산합니다.

    Returns: (runs, total_width)
        runs: tuple of (font_path, glyphs tuple, x_positions tuple)
    """
    style = _style_from_cache_key(style_key)
    if not line:
        return (), 0.0

    # 문자 → 폰트 매핑 후 연속 구간으로 묶기
    segments: List[Tuple[str, List[str]]] = []
    for char in line:
        resolved = _resolve_char_font(char, primary_font_path)
        if segments and segments[-1][0] == resolved:
            segments[-1][1].append(char)
        else:
            segments.append((resolved, [char]))

    runs = []
    x = 0.0
    glyph_count = 0
    for font_path, chars in segments:
        seg_text = "".join(chars)
        font = _make_font_cached(font_path, font_size, style_key)
        glyphs = font.textToGlyphs(seg_text)
        if len(glyphs) == 0:
            continue
        widths = font.getWidths(glyphs)
        x_positions = []
        for w in widths:
            x_positions.append(x)
            x += w + style.letter_spacing
        glyph_count += len(glyphs)
        runs.append((font_path, tuple(glyphs), tuple(x_positions)))

    total_width = max(0.0, x - style.letter_spacing) if glyph_count else 0.0
    return tuple(runs), total_width


@functools.lru_cache(maxsize=4096)
def _measure_line_cached(text: str, font_path: str, font_size: int, style_key: Tuple) -> float:
    _, width = _compute_line_runs_cached(text, font_path, font_size, style_key)
    return width


@functools.lru_cache(maxsize=4096)
def _measure_text_cached(text: str, font_path: str, font_size: int, style_key: Tuple) -> Tuple[float, float]:
    style = _style_from_cache_key(style_key)
    lines = text.split('\n')
    max_width = 0.0
    for line in lines:
        _, w = _compute_line_runs_cached(line, font_path, font_size, style_key)
        max_width = max(max_width, w)

    line_height = font_size * style.line_spacing
    total_height = line_height * len(lines)
    return max_width, total_height


@functools.lru_cache(maxsize=2048)
def _build_text_layout_cached(text: str, font_path: str, font_size: int, style_key: Tuple):
    style = _style_from_cache_key(style_key)
    line_height = font_size * style.line_spacing
    lines_data = []
    max_width = 0.0

    for line in text.split('\n'):
        runs, line_width = _compute_line_runs_cached(line, font_path, font_size, style_key)
        lines_data.append((runs, line_width))
        max_width = max(max_width, line_width)

    return {
        "font_size": font_size,
        "style_key": style_key,
        "line_height": line_height,
        "text_width": max_width,
        "text_height": line_height * len(lines_data),
        "lines_data": tuple(lines_data),
    }


def measure_line(text: str, font_path: str, font_size: int, style: TextStyle = None) -> float:
    """한 줄 텍스트의 너비를 측정합니다 (자간 + 평체 + 글리프 폴백 적용)."""
    if style is None:
        style = DEFAULT_STYLES["standard"]
    return _measure_line_cached(text, font_path, font_size, _style_cache_key(style))


def measure_text(text: str, font_path: str, font_size: int, style: TextStyle = None) -> Tuple[float, float]:
    """멀티라인 텍스트의 (width, height)를 측정합니다."""
    if style is None:
        style = DEFAULT_STYLES["standard"]
    return _measure_text_cached(text, font_path, font_size, _style_cache_key(style))


def _build_text_layout(text: str, font_path: str, font_size: int, style: TextStyle):
    """줄 단위 렌더링에 필요한 레이아웃 정보를 계산합니다."""
    return _build_text_layout_cached(text, font_path, font_size, _style_cache_key(style))


@functools.lru_cache(maxsize=4096)
def _measure_character_body_cached(text: str, font_path: str, font_size: int, style_key: Tuple) -> Tuple[float, float]:
    normalized_text = replace_unsupported_chars(text, font_path)
    body_widths = []
    body_heights = []

    for char in normalized_text:
        if char.isspace():
            continue
        resolved_path = _resolve_char_font(char, font_path)
        font = _make_font_cached(resolved_path, font_size, style_key)
        glyphs = font.textToGlyphs(char)
        if len(glyphs) == 0:
            continue
        _, bounds = font.getWidthsBounds(glyphs, None)
        for rect in bounds:
            width = float(rect.width())
            height = float(rect.height())
            if width <= 0.0 or height <= 0.0:
                continue
            body_widths.append(width)
            body_heights.append(height)

    if not body_widths or not body_heights:
        return 0.0, 0.0

    return float(np.median(body_widths)), float(np.median(body_heights))


def measure_character_body_size(text: str, font_path: str, font_size: int, style: TextStyle = None) -> Tuple[float, float]:
    if style is None:
        style = DEFAULT_STYLES["standard"]
    return _measure_character_body_cached(text, font_path, font_size, _style_cache_key(style))


def measure_character_body_height_ratio(
    text: str,
    font_path: str,
    font_size: int,
    style: TextStyle = None,
    reference_height: float = 1.0,
) -> float:
    resolved_reference = max(float(reference_height), 1.0)
    _, body_height = measure_character_body_size(text, font_path, font_size, style)
    return body_height / resolved_reference


def _paint_line_runs(canvas: skia.Canvas, runs, x_offset: float, baseline_y: float,
                     font_size: int, style_key: Tuple, style: TextStyle, kind: str):
    """한 줄의 모든 런을 stroke 또는 fill 단일 패스로 그립니다."""
    if kind == "stroke":
        paint = skia.Paint(
            Color=skia.Color(*style.stroke_color),
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=style.stroke_width,
            StrokeJoin=skia.Paint.kRound_Join,
        )
    else:
        paint = skia.Paint(Color=skia.Color(*style.color), AntiAlias=True)

    for font_path, glyphs, x_positions in runs:
        if not glyphs:
            continue
        font = _make_font_cached(font_path, font_size, style_key)
        builder = skia.TextBlobBuilder()
        builder.allocRunPosH(font, list(glyphs), [x + x_offset for x in x_positions], baseline_y)
        canvas.drawTextBlob(builder.make(), 0, 0, paint)


def _render_text_layer(
    text: str,
    font_path: str,
    font_size: int,
    style: TextStyle,
    align: str = "center",
    layout=None,
):
    """텍스트만 들어간 투명 레이어를 생성합니다.

    외곽선이 있으면 모든 줄의 외곽선을 먼저 그리고 본체를 나중에 그려,
    줄 간격이 좁을 때 다음 줄 외곽선이 앞 줄 본체를 깎는 문제를 방지한다.
    """
    oversample = max(1, style.oversample_scale)
    oversampled_style = TextStyle(
        color=style.color,
        letter_spacing=style.letter_spacing * oversample,
        horizontal_scale=style.horizontal_scale,
        line_spacing=style.line_spacing,
        stroke_width=style.stroke_width * oversample,
        stroke_color=style.stroke_color,
        embolden=style.embolden,
        oversample_scale=1,
        subpixel=style.subpixel,
        baseline_snap=style.baseline_snap,
        linear_metrics=style.linear_metrics,
        hinting=style.hinting,
    )
    oversampled_key = _style_cache_key(oversampled_style)
    oversampled_size = font_size * oversample
    if layout is None:
        layout = _build_text_layout_cached(text, font_path, oversampled_size, oversampled_key)
    padding = max(4, int(round(font_size * 0.35)))
    layer_width = max(1, int(np.ceil(layout["text_width"] / oversample)) + padding * 2)
    layer_height = max(1, int(np.ceil(layout["text_height"] / oversample)) + padding * 2)

    surface = skia.Surface(layer_width * oversample, layer_height * oversample)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorTRANSPARENT)

    start_x = padding * oversample
    start_y = padding * oversample
    max_width = layout["text_width"]

    paint_passes = ("stroke", "fill") if oversampled_style.stroke_width > 0 else ("fill",)
    for kind in paint_passes:
        for line_index, (runs, line_width) in enumerate(layout["lines_data"]):
            if not runs:
                continue
            if align == "left":
                x_offset = start_x
            elif align == "right":
                x_offset = start_x + (max_width - line_width)
            else:
                x_offset = start_x + (max_width - line_width) / 2

            baseline_y = start_y + layout["line_height"] * (line_index + 0.82)
            _paint_line_runs(
                canvas, runs, x_offset, baseline_y,
                layout["font_size"], layout["style_key"], oversampled_style, kind,
            )

    snapshot = Image.fromarray(surface.makeImageSnapshot().toarray(), mode="RGBA")
    if oversample > 1:
        snapshot = snapshot.resize((layer_width, layer_height), resample=Image.Resampling.LANCZOS)

    return snapshot, layer_width, layer_height, padding


def render_text_on_image(
    img_pil: Image.Image,
    text: str,
    center_x: float,
    center_y: float,
    font_path: str,
    font_size: int,
    style: TextStyle = None,
    align: str = 'center',
    anchor: str = 'mm',
) -> Tuple[int, int, int, int]:
    """PIL Image 위에 skia로 텍스트를 렌더링합니다.

    Args:
        img_pil: 대상 PIL Image (RGB)
        text: 렌더링할 텍스트 (멀티라인 \n 지원)
        center_x, center_y: 앵커 기준 좌표
        font_path: 폰트 파일 경로
        font_size: 폰트 크기 (px)
        style: TextStyle 프리셋
        align: 'left', 'center', 'right'
        anchor: 'mm' (중앙), 'lm' (좌중), 'rm' (우중) 등

    Returns:
        (x1, y1, x2, y2) 바운딩 박스
    """
    if style is None:
        style = DEFAULT_STYLES["standard"]
    text_width, text_height = measure_text(text, font_path, font_size, style)

    # 앵커 기준 좌표 → 좌상단 좌표 변환
    if 'l' in anchor:
        start_x = center_x
    elif 'r' in anchor:
        start_x = center_x - text_width
    else:  # 'm' or center
        start_x = center_x - text_width / 2

    if anchor.endswith('t') or anchor[0] == 't':
        start_y = center_y
    elif anchor.endswith('b') or anchor[0] == 'b':
        start_y = center_y - text_height
    else:  # 'm'
        start_y = center_y - text_height / 2
    text_layer, _, _, padding = _render_text_layer(text, font_path, font_size, style, align)
    paste_x = int(round(start_x)) - padding
    paste_y = int(round(start_y)) - padding
    img_pil.paste(text_layer, (paste_x, paste_y), text_layer)

    # 바운딩 박스 반환
    bbox = (int(start_x), int(start_y), int(start_x + text_width), int(start_y + text_height))
    return bbox


def _tokenize_vertical_text(text: str) -> list:
    """Split text into vertical-stack slots.

    Consecutive ! and ? merge into one slot so "!?" displays horizontally in a
    single vertical slot (manga convention). Every other char is its own slot.
    """
    if not text:
        return []
    return re.findall(r'[!?]+|.', text, flags=re.DOTALL)


def _should_rotate_token(token: str) -> bool:
    return len(token) == 1 and token in VERTICAL_ROTATE_CHARS


def _vertical_token_size(token: str, font_path: str, font_size: int, style: TextStyle) -> Tuple[float, float, bool]:
    """세로 슬롯 하나의 (width, height, rotate)를 계산합니다. 공백은 갭으로 처리."""
    if token.isspace():
        return 0.0, font_size * _VERTICAL_SPACE_RATIO, False
    w, h = measure_text(token, font_path, font_size, style)
    if _should_rotate_token(token):
        return h, w, True
    return w, h, False


def _layout_vertical_columns(
    text: str,
    font_path: str,
    font_size: int,
    style: TextStyle,
    max_column_height: Optional[float] = None,
    max_columns: Optional[int] = None,
):
    """세로쓰기 단(컬럼) 레이아웃을 계산합니다.

    - '\n'은 강제 단 나눔.
    - max_column_height를 넘으면 새 단으로 넘어간다 (행두 금칙 문자는 이전 단에 붙임).
    - 단은 오른쪽에서 왼쪽으로 읽는 순서로 배치된다 (renderer가 처리).

    Returns: (columns, col_widths, col_heights, gap)
        columns: list[list[(token, w, h, rotate)]]
    """
    gap = font_size * _VERTICAL_COLUMN_GAP_RATIO
    columns: List[List[Tuple[str, float, float, bool]]] = []

    for paragraph in text.split('\n'):
        tokens = _tokenize_vertical_text(paragraph)
        if not tokens:
            continue
        current: List[Tuple[str, float, float, bool]] = []
        current_h = 0.0
        for token in tokens:
            w, h, rotate = _vertical_token_size(token, font_path, font_size, style)
            overflows = (
                max_column_height is not None
                and current
                and current_h + h > max_column_height
            )
            if overflows and token in LINE_HEAD_FORBIDDEN:
                # 행두 금칙: 새 단의 머리에 못 오는 문자는 현재 단 끝에 붙인다.
                overflows = False
            if overflows and token.isspace():
                # 단 머리 공백은 의미가 없으므로 단을 넘기면서 버린다.
                columns.append(current)
                current, current_h = [], 0.0
                continue
            if overflows:
                columns.append(current)
                current, current_h = [], 0.0
            current.append((token, w, h, rotate))
            current_h += h
        if current:
            columns.append(current)

    if max_columns is not None and max_columns > 0 and len(columns) > max_columns:
        # 단 수 상한 초과분은 마지막 단에 이어붙인다 (fits 판정은 호출부 책임).
        merged = columns[:max_columns - 1]
        tail: List[Tuple[str, float, float, bool]] = []
        for col in columns[max_columns - 1:]:
            tail.extend(col)
        merged.append(tail)
        columns = merged

    col_widths = [
        max((w for _, w, _, _ in col if w > 0), default=font_size)
        for col in columns
    ]
    col_heights = [sum(h for _, _, h, _ in col) for col in columns]
    return columns, col_widths, col_heights, gap


def measure_vertical_block(
    text: str,
    font_path: str,
    font_size: int,
    style: TextStyle = None,
    max_column_height: Optional[float] = None,
    max_columns: Optional[int] = None,
) -> Tuple[float, float, int]:
    """세로쓰기 블록의 (총너비, 최대단높이, 단 수)를 계산합니다."""
    if style is None:
        style = DEFAULT_STYLES["standard"]
    columns, col_widths, col_heights, gap = _layout_vertical_columns(
        text, font_path, font_size, style, max_column_height, max_columns
    )
    if not columns:
        return 0.0, 0.0, 0
    total_w = sum(col_widths) + gap * (len(columns) - 1)
    total_h = max(col_heights)
    return total_w, total_h, len(columns)


def measure_vertical_text(
    text: str,
    font_path: str,
    font_size: int,
    style: TextStyle = None,
    max_column_height: Optional[float] = None,
    max_columns: Optional[int] = None,
) -> Tuple[float, float]:
    """세로쓰기 블록의 (width, height)를 측정합니다 (다단 포함)."""
    w, h, _ = measure_vertical_block(text, font_path, font_size, style, max_column_height, max_columns)
    return w, h


def render_vertical_text_on_image(
    img_pil: Image.Image,
    text: str,
    center_x: float,
    center_y: float,
    font_path: str,
    font_size: int,
    style: TextStyle = None,
    max_column_height: Optional[float] = None,
    max_columns: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Draw text as vertical column stacks (right→left), rotating candidates 90° CW.

    Block is centered on (center_x, center_y); columns are top-aligned.
    Returns the bounding box.
    """
    if style is None:
        style = DEFAULT_STYLES["standard"]

    columns, col_widths, col_heights, gap = _layout_vertical_columns(
        text, font_path, font_size, style, max_column_height, max_columns
    )
    if not columns:
        cx, cy = int(center_x), int(center_y)
        return (cx, cy, cx, cy)

    total_w = sum(col_widths) + gap * (len(columns) - 1)
    total_h = max(col_heights)
    top_y = center_y - total_h / 2.0
    cursor_right = center_x + total_w / 2.0

    for col, col_w in zip(columns, col_widths):
        col_center_x = cursor_right - col_w / 2.0
        cur_y = top_y
        for token, slot_w, slot_h, rotate in col:
            if token.isspace():
                cur_y += slot_h
                continue
            layer, _, _, _ = _render_text_layer(token, font_path, font_size, style, align="center")
            if rotate:
                layer = layer.rotate(-90, expand=True, resample=Image.Resampling.BICUBIC)
            paste_x = int(round(col_center_x - layer.width / 2))
            paste_y = int(round(cur_y + slot_h / 2 - layer.height / 2))
            img_pil.paste(layer, (paste_x, paste_y), layer)
            cur_y += slot_h
        cursor_right -= col_w + gap

    x1 = int(round(center_x - total_w / 2))
    y1 = int(round(top_y))
    x2 = int(round(center_x + total_w / 2))
    y2 = int(round(top_y + total_h))
    return (x1, y1, x2, y2)


def render_rotated_text_on_image(
    img_pil: Image.Image,
    text: str,
    center_x: float,
    center_y: float,
    angle: float,
    font_path: str,
    font_size: int,
    style: TextStyle = None,
    align: str = 'center',
) -> Tuple[int, int, int, int]:
    """회전된 텍스트를 렌더링합니다."""
    if style is None:
        style = DEFAULT_STYLES["standard"]
    text_layer, text_w, text_h, _ = _render_text_layer(text, font_path, font_size, style, align)
    rotated = text_layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    # 원본 이미지에 합성
    paste_x = int(center_x - rotated.width / 2)
    paste_y = int(center_y - rotated.height / 2)
    img_pil.paste(rotated, (paste_x, paste_y), rotated)

    return (paste_x, paste_y, paste_x + rotated.width, paste_y + rotated.height)
