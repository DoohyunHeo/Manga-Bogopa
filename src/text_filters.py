"""번역하지 않을 글자 거르기 — 기호만 있는 대사, 효과음과 잡지 부속물(워터마크·주소·SNS 아이디·잡지 제호),
그리고 보완 탐지가 그림을 글자로 잘못 잡은 조각(is_doubtful_supplement).

말풍선 밖의 가타카나로만 된 글자는 대부분 효과음이다. 긴 외래어·이름은 지키도록 6자 이하이거나 같은 글자가
되풀이되는 것만 효과음으로 본다. 히라가나 효과음은 짧은 대사 조각과 구별되지 않아 거르지 않는다. 거른 글자는
'번역 제외'가 되어 지우지도 식자하지도 않는다.

효과음·의태어는 번역 지침으로도 '번역 불가'가 되므로, 코드로는 확실한 것만 거른다 — 가타카나 효과음은
글자가 그 쪽 말풍선 대사보다 1.5배 이상 클 때만. 채팅·화면 글자처럼 대사 크기의 가타카나는
번역에 넘긴다.
"""
import re
import unicodedata
from typing import Optional

_KANA = re.compile(r"[぀-ゟ゠-ヿー]")
_KATAKANA = re.compile(r"[゠-ヿー]")
_REPEATED = re.compile(r"(.{1,2})\1{2,}")
_URL = re.compile(r"scan|https?:|www\.|\.com|\.org|\.net")
_SNS_ONLY = re.compile(r"[@#][A-Za-z0-9_.]+")
_MAGAZINE_TITLES = {
    "少年マガジン", "週刊少年マガジン", "少年ジャンプ", "週刊少年ジャンプ", "ジャンプ", "少年サンデー",
    "週刊少年サンデー", "サンデー", "週刊ジャンプ", "週刊マガジン", "週刊サンデー", "wednesdayissunday", "毎週水曜発売", "毎週水曜日発売", "毎週月曜発売",
    "毎週月曜日発売",
}
SFX_MAX_CHARS = 6
SFX_MIN_SIZE_RATIO = 1.5   # 쪽 말풍선 대사 글자 크기(중앙값)의 이 배 이상일 때만 가타카나를 효과음으로
_MIN_REFERENCE_BUBBLES = 3
# 보완 탐지 오탐 거르기 (is_doubtful_supplement)
_KANA_ONLY = re.compile(r"[぀-ゟ゠-ヿ]")
SUPPLEMENT_MIN_SCORE = 0.5
SUPPLEMENT_SHORT_KANA = 2
SUPPLEMENT_MAX_KANA = 4


def dialogue_size(bubbles, fallback: Optional[float] = None) -> Optional[float]:
    """말풍선(SpeechBubble 목록) 대사의 글자 크기 중앙값. 말풍선이 적으면 fallback."""
    sizes = sorted(b.text_element.font_size for b in bubbles if b.text_element.font_size)
    if len(sizes) < _MIN_REFERENCE_BUBBLES:
        return fallback
    mid = len(sizes) // 2
    return float(sizes[mid]) if len(sizes) % 2 else (sizes[mid - 1] + sizes[mid]) / 2.0


def is_doubtful_supplement(text: str, score: float, readings=None) -> bool:
    """보완 탐지(기본 모델이 못 본 자리)가 그림을 글자로 잘못 잡은 것 같으면 True.

    점수가 낮은 보완 조각에서 판독이 가나 몇 자뿐이면 선반 그림·고양이 얼굴 같은 그림 선을 읽어 낸
    경우가 많다. 가나 SUPPLEMENT_SHORT_KANA자 이하는 그것만으로, SUPPLEMENT_MAX_KANA자 이하는 두 판독기
    (readings = (manga-ocr, Hayai))가 한 글자도 같게 읽지 못했을 때 버린다. 기본 모델이 찾은 박스는
    보지 않으므로 기본 모델이 잡은 짧은 대사(え·ん)는 그대로다.
    """
    if score >= SUPPLEMENT_MIN_SCORE:
        return False
    core = [ch for ch in unicodedata.normalize("NFKC", text or "") if ch.isalnum()]
    if not core or len(core) > SUPPLEMENT_MAX_KANA or not all(_KANA_ONLY.match(ch) for ch in core):
        return False
    if len(core) <= SUPPLEMENT_SHORT_KANA:
        return True
    if not readings:
        return False
    first, second = ({ch for ch in unicodedata.normalize("NFKC", r or "") if ch.isalnum()} for r in readings)
    return not (first & second)


def exclusion_reason(text: str, in_bubble: bool, size_ratio: Optional[float] = None) -> Optional[str]:
    """번역하지 않을 글자면 그 이유("기호"/"부속물"/"효과음"), 아니면 None.

    size_ratio = 이 글자 크기 ÷ 쪽 말풍선 대사 크기(dialogue_size). 모르면 효과음으로 보지 않는다.
    """
    norm = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))
    # 가나·한자·영숫자가 하나도 없는 대사(…, !?, ・・・, ♡)는 번역도 식자도 하지 않고 원본 그대로 둔다
    if norm and not any(ch.isalnum() for ch in norm):
        return "기호"
    lower = norm.lower()
    if _URL.search(lower) or _SNS_ONLY.fullmatch(norm):
        return "부속물"
    if "".join(ch for ch in lower if ch.isalnum()) in _MAGAZINE_TITLES:
        return "부속물"
    if not in_bubble and size_ratio is not None and size_ratio >= SFX_MIN_SIZE_RATIO:
        # 문장부호·기호만 빼고 가나·한자·영숫자는 남긴다 — 한자가 섞이면(이름 등) 효과음이 아니다
        core = "".join(ch for ch in norm if _KANA.match(ch) or ch.isalnum())
        if core and all(_KATAKANA.match(ch) for ch in core) and (
                len(core) <= SFX_MAX_CHARS or _REPEATED.search(core)):
            return "효과음"
    return None


# 도안 로고가 여러 조각으로 잡혀 일부만 번역되면 반쪽만 한국어가 된다. 말풍선 밖의 큰 글자(쪽 말풍선 대사의 2배 이상)
# 조각 둘이 같은 줄에 붙어 있고 (세로 겹침 60% 이상, 틈이 작은 쪽 높이의 30% 이하, 크기 비 2배 이내) 한쪽이 번역
# 제외면 무리 전체를 원문으로 둔다.
LOGO_MIN_SIZE_RATIO = 2.0
_LOGO_MIN_OVERLAP = 0.6
_LOGO_MAX_GAP = 0.3
_LOGO_MAX_SIZE_SPREAD = 2.0


def _same_logo_row(a, b):
    """두 상자가 한 줄로 붙어 있는지 (가로 또는 세로로 나란히)."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    for (a1, a2, b1, b2), (c1, c2, d1, d2) in (((ay1, ay2, by1, by2), (ax1, ax2, bx1, bx2)),
                                               ((ax1, ax2, bx1, bx2), (ay1, ay2, by1, by2))):
        across = min(a2 - a1, b2 - b1)
        overlap = min(a2, b2) - max(a1, b1)
        gap = max(d1 - c2, c1 - d2, 0)
        if across > 0 and overlap >= _LOGO_MIN_OVERLAP * across and gap <= _LOGO_MAX_GAP * across:
            return True
    return False


_LATIN_LOGO = re.compile(r"[A-Z][A-Z0-9 ・.&'!-]{3,}")


def split_logo_parts(page, dialogue_size_px):
    """번역된 말풍선 밖 조각 가운데 원문으로 둘 도안 로고 (id 집합).

    번역 제외 조각과 한 로고를 이루는 조각, 그리고 라틴 대문자만으로 된 큰 제목 로고(WITCHWATCH처럼 한 조각이어도).
    """
    from src.data_models import TranslationStatus
    if not dialogue_size_px:
        return set()
    big = [t for t in page.freeform_texts if t.font_size and t.font_size >= LOGO_MIN_SIZE_RATIO * dialogue_size_px]
    excluded = [t for t in big if t.translation_status == TranslationStatus.EXCLUDED]
    keep = set()
    for text in big:
        if text.translation_status != TranslationStatus.TRANSLATED:
            continue
        if _LATIN_LOGO.fullmatch(unicodedata.normalize("NFKC", text.original_text or "").strip()):
            keep.add(id(text))
            continue
        for other in excluded:
            spread = max(text.font_size, other.font_size) / max(1, min(text.font_size, other.font_size))
            if spread <= _LOGO_MAX_SIZE_SPREAD and _same_logo_row(text.text_box, other.text_box):
                keep.add(id(text))
                break
    return keep
