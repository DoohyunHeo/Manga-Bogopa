"""굵은 대사 글꼴 = 평범한 대사 글꼴의 같은 가족 굵은 글꼴.

굵기는 책 기준으로 코드가 정하는 성질이라(src/font_relative.py) 굵은 대사 글꼴을 따로 고르지 않는다 —
config.apply_font_modes가 설정을 읽거나 바꿀 때마다 FONT_MAP["shouting"]을 bold_partner로 다시 정한다.

- 가족 이름은 이름표 16(없으면 1)의 모든 언어 이름, 굵기는 OS/2 usWeightClass(가변 글꼴은 굵기 축 기본값)로 본다.
  기울임·폭이 같은 글꼴만 같은 가족으로 친다.
- 목표 굵기 = max(600, 보통 굵기 + 200)에 가장 가까운, 보통보다 굵은 파일 (거리가 같으면 더 굵은 쪽).
- 가변 글꼴은 목표 굵기의 고정 글꼴을 한 번 만들어 임시 폴더에 둔다 (Noto Serif KR 약 13초, 다음부터는 그 파일).
- 같은 가족 굵은 파일이 없으면 보통 글꼴 그대로 — 굵은 대사 글씨 모양(text_renderer DEFAULT_STYLES["shouting"])의
  합성 굵게(embolden)로 그린다.
"""
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")
# 굵은 대사 목표 굵기 — 강조 대사 획이 보통 대사보다 조금만 굵어 보이게 Pretendard Regular 400의 짝을
# SemiBold 600(획 1.38배)으로 둔다 (합성 굵게 여부는 text_layout.needs_synthetic_bold가 정한다)
_BOLD_WEIGHT = 600     # 굵은 대사 목표 굵기의 하한
_BOLD_STEP = 200       # 보통 굵기보다 이만큼 더 굵은 것을 목표로 (Pretendard Regular 400 → SemiBold 600)
_INSTANCE_DIR = os.path.join(tempfile.gettempdir(), "manga_bogopa_fonts")
_FACES = {}            # (절대 경로, 크기, 수정 시각) → _Face 또는 None (못 읽는 글꼴)


@dataclass(frozen=True)
class BoldFont:
    path: str        # 굵은 대사를 그릴 글꼴 파일 (합성이면 보통 글꼴 그대로, 가변 글꼴이면 만든 고정 글꼴)
    source: str      # 글꼴 폴더에 있는 원래 파일 (화면에 보일 이름)
    weight: int      # 굵기 (합성이면 보통 글꼴의 굵기)
    synthetic: bool  # 같은 가족 굵은 파일이 없어 보통 글꼴을 합성 굵게로 그린다


@dataclass(frozen=True)
class _Face:
    families: frozenset
    weight: int
    italic: bool
    width: int
    axis: Optional[Tuple[float, float]]  # 굵기 축 (최소, 최대) — 가변 글꼴만


def _read_face(path: str) -> Optional[_Face]:
    from fontTools.ttLib import TTFont
    try:
        with TTFont(path, lazy=True, fontNumber=0) as font:
            def family(name_id):
                names = set()
                for record in font["name"].names:
                    if record.nameID == name_id:
                        try:
                            names.add("".join(record.toUnicode().split()).casefold())
                        except UnicodeDecodeError:
                            continue
                return {name for name in names if name}

            families = family(16) or family(1)
            os2 = font["OS/2"] if "OS/2" in font else None
            axis = default = None
            if "fvar" in font:
                for a in font["fvar"].axes:
                    if a.axisTag == "wght":
                        axis, default = (a.minValue, a.maxValue), a.defaultValue
            weight = int(default if default is not None else (os2.usWeightClass if os2 else 400))
            italic = bool((os2.fsSelection & 1) if os2 else 0) or bool(font["head"].macStyle & 2)
            width = int(os2.usWidthClass) if os2 else 5
    except Exception as exc:  # noqa: BLE001 — 못 읽는 글꼴은 후보에서 뺀다
        logger.debug("글꼴 이름표를 읽지 못함 %s: %s", path, exc)
        return None
    return _Face(frozenset(families), weight, italic, width, axis) if families else None


def _face(path: str) -> Optional[_Face]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns)
    if key not in _FACES:
        _FACES[key] = _read_face(path)
    return _FACES[key]


def _candidates(regular_path: str, font_dir: str):
    folders = [font_dir, os.path.dirname(regular_path)]
    seen = set()
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            key = os.path.normcase(os.path.abspath(path))
            if name.lower().endswith(FONT_EXTENSIONS) and key not in seen and os.path.isfile(path):
                seen.add(key)
                yield path


def _static_instance(path: str, weight: int) -> Optional[str]:
    """가변 글꼴의 한 굵기를 고정 글꼴 파일로 (임시 폴더에 한 번 만들어 둔다). 못 만들면 None."""
    stat = os.stat(path)
    digest = hashlib.sha1(f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()[:10]
    stem, ext = os.path.splitext(os.path.basename(path))
    out = os.path.join(_INSTANCE_DIR, f"{stem}-wght{weight}-{digest}{ext}")
    if os.path.exists(out):
        return out
    tmp = f"{out}.{os.getpid()}.tmp"
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer
        logger.info("가변 글꼴 %s의 굵기 %d 글꼴을 만드는 중 (처음 한 번)", os.path.basename(path), weight)
        os.makedirs(_INSTANCE_DIR, exist_ok=True)
        with TTFont(path) as font:
            instancer.instantiateVariableFont(font, {"wght": weight}, inplace=True)
            font.save(tmp)
        os.replace(tmp, out)
        return out
    except Exception as exc:  # noqa: BLE001 — 못 만들면 합성 굵게로 그린다
        logger.warning("가변 글꼴 %s의 굵기 %d 글꼴을 만들지 못해 합성 굵게로 그립니다: %s", path, weight, exc)
        if os.path.exists(tmp):
            os.remove(tmp)
        return None


def bold_partner(regular_path: str, font_dir: str) -> BoldFont:
    """평범한 대사 글꼴(regular_path)의 같은 가족 굵은 글꼴을 글꼴 폴더(와 그 글꼴이 있는 폴더)에서 고른다."""
    regular = _face(regular_path) if regular_path else None
    if regular is None:
        return BoldFont(regular_path, regular_path, 400, True)
    target = max(_BOLD_WEIGHT, regular.weight + _BOLD_STEP)
    best = None
    for path in _candidates(regular_path, font_dir):
        face = _face(path)
        if (face is None or not (face.families & regular.families)
                or face.italic != regular.italic or face.width != regular.width):
            continue
        weight = int(min(max(target, face.axis[0]), face.axis[1])) if face.axis else face.weight
        if weight <= regular.weight:
            continue
        # 목표에 가까운 것 → 더 굵은 것 → 고정 글꼴(만들 필요 없음) → 파일 이름 순
        rank = (abs(weight - target), -weight, face.axis is not None, os.path.basename(path).casefold())
        if best is None or rank < best[0]:
            best = (rank, path, weight, face.axis is not None)
    if best is None:
        return BoldFont(regular_path, regular_path, regular.weight, True)
    _, path, weight, variable = best
    if variable:
        instance = _static_instance(path, weight)
        if instance is None:
            return BoldFont(regular_path, regular_path, regular.weight, True)
        return BoldFont(instance, path, weight, False)
    return BoldFont(path, path, weight, False)
