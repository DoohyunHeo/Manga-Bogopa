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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import skia
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TextStyle:
    """텍스트 스타일 프리셋."""
    color: Tuple[int, int, int] = (76, 76, 76)
    letter_spacing: float = -2.0       # px, 음수=좁게
    horizontal_scale: float = 0.9      # 가로 비율 (0.9 = 평체 90%, 세로로 길쭉)
    line_spacing: float = 1.1          # 행간 배율
    stroke_width: float = 0.0          # 외곽선 두께 (0=없음)
    stroke_color: Tuple[int, int, int] = (255, 255, 255)


# 스타일별 프리셋 (만화 식자 표준)
DEFAULT_STYLES: Dict[str, TextStyle] = {
    "standard":     TextStyle(color=(76, 76, 76), letter_spacing=-2.0, horizontal_scale=0.9, line_spacing=1.1),
    "shouting":     TextStyle(color=(0, 0, 0), letter_spacing=-1.0, horizontal_scale=0.85, line_spacing=1.0),
    "cute":         TextStyle(color=(50, 50, 50), letter_spacing=0.0, horizontal_scale=0.95, line_spacing=1.15),
    "narration":    TextStyle(color=(60, 60, 60), letter_spacing=0.0, horizontal_scale=0.95, line_spacing=1.2),
    "handwriting":  TextStyle(color=(40, 40, 40), letter_spacing=1.0, horizontal_scale=1.0, line_spacing=1.1),
    "pop":          TextStyle(color=(0, 0, 0), letter_spacing=-1.0, horizontal_scale=0.9, line_spacing=1.0),
    "angry":        TextStyle(color=(0, 0, 0), letter_spacing=-1.0, horizontal_scale=0.85, line_spacing=1.0),
    "scared":       TextStyle(color=(0, 0, 0), letter_spacing=0.0, horizontal_scale=1.0, line_spacing=1.1),
    "embarrassment": TextStyle(color=(60, 60, 60), letter_spacing=0.0, horizontal_scale=1.0, line_spacing=1.1),
}

# 말풍선 밖 텍스트 기본 스타일 (외곽선 포함)
FREEFORM_STYLE = TextStyle(
    color=(0, 0, 0), letter_spacing=-1.0, horizontal_scale=0.9,
    line_spacing=1.0, stroke_width=2.0, stroke_color=(255, 255, 255),
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


def _make_font(font_path: str, font_size: int) -> skia.Font:
    """skia.Font 객체 생성."""
    typeface = _load_typeface(font_path)
    font = skia.Font(typeface, font_size)
    font.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
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
    font = _make_font(font_path, font_size)
    _, _, width = _compute_glyph_positions(font, text, style.letter_spacing)
    return width * style.horizontal_scale


def measure_text(text: str, font_path: str, font_size: int, style: TextStyle = None) -> Tuple[float, float]:
    """멀티라인 텍스트의 (width, height)를 측정합니다.
    Pillow의 draw.multiline_textbbox() 대체."""
    if style is None:
        style = DEFAULT_STYLES["standard"]
    font = _make_font(font_path, font_size)
    lines = text.split('\n')
    max_width = 0.0
    for line in lines:
        _, _, w = _compute_glyph_positions(font, line, style.letter_spacing)
        max_width = max(max_width, w)

    line_height = font_size * style.line_spacing
    total_height = line_height * len(lines)

    return max_width * style.horizontal_scale, total_height


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

    img_np = np.array(img_pil.convert('RGBA'))
    h, w = img_np.shape[:2]

    # skia Surface 생성 (RGBA)
    surface = skia.Surface(w, h)
    canvas = surface.getCanvas()

    # 기존 이미지를 skia canvas에 그리기
    skia_image = skia.Image.fromarray(img_np, colorType=skia.kRGBA_8888_ColorType)
    canvas.drawImage(skia_image, 0, 0)

    font = _make_font(font_path, font_size)
    lines = text.split('\n')
    line_height = font_size * style.line_spacing

    # 전체 텍스트 크기 계산
    line_widths = []
    line_glyphs_data = []
    for line in lines:
        glyphs, x_pos, line_w = _compute_glyph_positions(font, line, style.letter_spacing)
        line_widths.append(line_w)
        line_glyphs_data.append((glyphs, x_pos, line_w))

    text_width = max(line_widths) if line_widths else 0
    text_height = line_height * len(lines)
    scaled_width = text_width * style.horizontal_scale

    # 앵커 기준 좌표 → 좌상단 좌표 변환
    if 'l' in anchor:
        start_x = center_x
    elif 'r' in anchor:
        start_x = center_x - scaled_width
    else:  # 'm' or center
        start_x = center_x - scaled_width / 2

    if anchor.endswith('t') or anchor[0] == 't':
        start_y = center_y
    elif anchor.endswith('b') or anchor[0] == 'b':
        start_y = center_y - text_height
    else:  # 'm'
        start_y = center_y - text_height / 2

    # 평체 적용
    canvas.save()
    canvas.scale(style.horizontal_scale, 1.0)

    # 줄별 렌더링
    for line_idx, (glyphs, x_pos, line_w) in enumerate(line_glyphs_data):
        if not glyphs:
            continue

        # 정렬에 따른 x 오프셋
        if align == 'left':
            x_offset = start_x / style.horizontal_scale
        elif align == 'right':
            x_offset = (start_x + scaled_width - line_w * style.horizontal_scale) / style.horizontal_scale
        else:  # center
            x_offset = (start_x + (scaled_width - line_w * style.horizontal_scale) / 2) / style.horizontal_scale

        y = start_y + line_height * (line_idx + 0.8)  # baseline 위치 (0.8 = ascent 비율)

        # x 위치에 오프셋 적용
        positioned_x = [xp + x_offset for xp in x_pos]

        # 스트로크 (외곽선) 먼저
        if style.stroke_width > 0:
            stroke_paint = skia.Paint()
            stroke_paint.setColor(skia.Color(*style.stroke_color))
            stroke_paint.setAntiAlias(True)
            stroke_paint.setStyle(skia.Paint.kStroke_Style)
            stroke_paint.setStrokeWidth(style.stroke_width)
            stroke_paint.setStrokeJoin(skia.Paint.kRound_Join)

            builder = skia.TextBlobBuilder()
            builder.allocRunPosH(font, glyphs, positioned_x, y)
            canvas.drawTextBlob(builder.make(), 0, 0, stroke_paint)

        # 본문 (fill)
        fill_paint = skia.Paint()
        fill_paint.setColor(skia.Color(*style.color))
        fill_paint.setAntiAlias(True)

        builder = skia.TextBlobBuilder()
        builder.allocRunPosH(font, glyphs, positioned_x, y)
        canvas.drawTextBlob(builder.make(), 0, 0, fill_paint)

    canvas.restore()

    # skia → numpy → PIL (RGB로 변환)
    result = surface.makeImageSnapshot().toarray()  # RGBA
    result_rgb = Image.fromarray(result).convert('RGB')
    img_pil.paste(result_rgb, (0, 0))

    # 바운딩 박스 반환
    bbox = (int(start_x), int(start_y), int(start_x + scaled_width), int(start_y + text_height))
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

    # 텍스트 크기 측정
    text_w, text_h = measure_text(text, font_path, font_size, style)
    text_w, text_h = int(text_w) + 20, int(text_h) + 20

    # 임시 투명 이미지에 텍스트 렌더링
    tmp = Image.new('RGBA', (text_w, text_h), (0, 0, 0, 0))
    # RGBA 임시 이미지에 텍스트 렌더링 (배경 투명)
    tmp_rgb = Image.new('RGB', (text_w, text_h), (255, 255, 255))
    render_text_on_image(
        tmp_rgb, text, text_w / 2, text_h / 2,
        font_path, font_size, style, align, 'mm'
    )

    # 알파 채널 생성 (흰색 배경과의 차이로)
    tmp_np = np.array(tmp_rgb)
    alpha = 255 - np.min(tmp_np, axis=2)  # 텍스트가 있는 곳은 알파 255
    tmp_rgba = Image.new('RGBA', (text_w, text_h))
    tmp_rgba.paste(tmp_rgb, (0, 0))
    tmp_rgba.putalpha(Image.fromarray(alpha))

    # 회전
    rotated = tmp_rgba.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    # 원본 이미지에 합성
    paste_x = int(center_x - rotated.width / 2)
    paste_y = int(center_y - rotated.height / 2)
    img_pil.paste(rotated, (paste_x, paste_y), rotated)

    return (paste_x, paste_y, paste_x + rotated.width, paste_y + rotated.height)
