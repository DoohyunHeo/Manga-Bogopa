"""
skia-python 기반 텍스트 렌더링 엔진.
만화 식자 품질의 텍스트 렌더링을 제공합니다.
- 자간 (letter spacing)
- 평체 (horizontal scale, 가로 90% = 세로로 길쭉)
- 행간 (line spacing)
- 스트로크 (외곽선)
- 서브픽셀 안티앨리어싱

Adobe Photoshop과 동일한 HarfBuzz 텍스트 엔진 기반.
"""
import functools
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import skia
from PIL import Image

logger = logging.getLogger(__name__)


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
    """폰트 파일 로드 (캐시됨)."""
    tf = skia.Typeface.MakeFromFile(font_path)
    if tf is None:
        logger.warning(f"폰트 로드 실패: {font_path}, 기본 폰트 사용")
        tf = skia.Typeface()
    return tf


@functools.lru_cache(maxsize=256)
def _has_glyph(font_path: str, char: str) -> bool:
    """폰트가 특정 문자의 글리프를 가지고 있는지 확인합니다 (캐시됨)."""
    typeface = _load_typeface(font_path)
    font = skia.Font(typeface, 12)
    glyphs = font.textToGlyphs(char)
    return len(glyphs) > 0 and glyphs[0] != 0


def replace_unsupported_chars(text: str, font_path: str) -> str:
    """폰트가 지원하지 않는 특수문자를 대체합니다.
    ⋯(U+22EF) → ・・・(U+30FB) → ...(ASCII) 순으로 폴백."""
    if "⋯" in text and not _has_glyph(font_path, "⋯"):
        if _has_glyph(font_path, "・"):
            text = text.replace("⋯", "・・・")
        else:
            text = text.replace("⋯", "...")
    if "︙" in text and not _has_glyph(font_path, "︙"):
        text = text.replace("︙", "⋮" if _has_glyph(font_path, "⋮") else ":")
    return text


def _make_font(font_path: str, font_size: int, style: TextStyle) -> skia.Font:
    """skia.Font 객체 생성."""
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


def _compute_glyph_positions(font: skia.Font, text: str, letter_spacing: float) -> Tuple[list, list, float]:
    """글리프 ID + 자간 적용된 x 위치 계산.
    Returns: (glyphs, x_positions, total_width)
    """
    glyphs = font.textToGlyphs(text)
    if len(glyphs) == 0:
        return [], [], 0.0
    widths = font.getWidths(glyphs)
    x_positions = []
    x = 0.0
    for i in range(len(glyphs)):
        x_positions.append(x)
        x += widths[i] + letter_spacing
    total_width = x - letter_spacing  # 마지막 글자 뒤에는 spacing 안 붙임
    return list(glyphs), x_positions, total_width


def measure_line(text: str, font_path: str, font_size: int, style: TextStyle = None) -> float:
    """한 줄 텍스트의 너비를 측정합니다 (자간 + 평체 적용).
    Pillow의 font.getlength() 대체."""
    if style is None:
        style = DEFAULT_STYLES["standard"]
    font = _make_font(font_path, font_size, style)
    _, _, width = _compute_glyph_positions(font, text, style.letter_spacing)
    return width


def measure_text(text: str, font_path: str, font_size: int, style: TextStyle = None) -> Tuple[float, float]:
    """멀티라인 텍스트의 (width, height)를 측정합니다.
    Pillow의 draw.multiline_textbbox() 대체."""
    if style is None:
        style = DEFAULT_STYLES["standard"]
    font = _make_font(font_path, font_size, style)
    lines = text.split('\n')
    max_width = 0.0
    for line in lines:
        _, _, w = _compute_glyph_positions(font, line, style.letter_spacing)
        max_width = max(max_width, w)

    line_height = font_size * style.line_spacing
    total_height = line_height * len(lines)

    return max_width, total_height


def _build_text_layout(text: str, font_path: str, font_size: int, style: TextStyle):
    """줄 단위 렌더링에 필요한 레이아웃 정보를 계산합니다."""
    font = _make_font(font_path, font_size, style)
    line_height = font_size * style.line_spacing
    lines_data = []
    max_width = 0.0

    for line in text.split('\n'):
        glyphs, x_positions, line_width = _compute_glyph_positions(font, line, style.letter_spacing)
        lines_data.append((glyphs, x_positions, line_width))
        max_width = max(max_width, line_width)

    return {
        "font": font,
        "line_height": line_height,
        "text_width": max_width,
        "text_height": line_height * len(lines_data),
        "lines_data": lines_data,
    }


def _paint_text_blob(canvas: skia.Canvas, font: skia.Font, glyphs, positioned_x, baseline_y, style: TextStyle):
    """하나의 줄을 stroke/fill 순으로 그립니다."""
    if style.stroke_width > 0:
        stroke_paint = skia.Paint(
            Color=skia.Color(*style.stroke_color),
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=style.stroke_width,
            StrokeJoin=skia.Paint.kRound_Join,
        )
        builder = skia.TextBlobBuilder()
        builder.allocRunPosH(font, glyphs, positioned_x, baseline_y)
        canvas.drawTextBlob(builder.make(), 0, 0, stroke_paint)

    fill_paint = skia.Paint(Color=skia.Color(*style.color), AntiAlias=True)
    builder = skia.TextBlobBuilder()
    builder.allocRunPosH(font, glyphs, positioned_x, baseline_y)
    canvas.drawTextBlob(builder.make(), 0, 0, fill_paint)


def _render_text_layer(
    text: str,
    font_path: str,
    font_size: int,
    style: TextStyle,
    align: str = "center",
):
    """텍스트만 들어간 투명 레이어를 생성합니다."""
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
    layout = _build_text_layout(text, font_path, font_size * oversample, oversampled_style)
    padding = max(4, int(round(font_size * 0.35)))
    layer_width = max(1, int(np.ceil(layout["text_width"] / oversample)) + padding * 2)
    layer_height = max(1, int(np.ceil(layout["text_height"] / oversample)) + padding * 2)

    surface = skia.Surface(layer_width * oversample, layer_height * oversample)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorTRANSPARENT)

    start_x = padding * oversample
    start_y = padding * oversample
    max_width = layout["text_width"]

    for line_index, (glyphs, x_positions, line_width) in enumerate(layout["lines_data"]):
        if not glyphs:
            continue
        if align == "left":
            x_offset = start_x
        elif align == "right":
            x_offset = start_x + (max_width - line_width)
        else:
            x_offset = start_x + (max_width - line_width) / 2

        baseline_y = start_y + layout["line_height"] * (line_index + 0.82)
        positioned_x = [x + x_offset for x in x_positions]
        _paint_text_blob(canvas, layout["font"], glyphs, positioned_x, baseline_y, oversampled_style)

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
