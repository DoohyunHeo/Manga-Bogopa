"""번역 세션들이 알려 준 인명·용어 표기를 모아 한 책의 사전을 만든다.

- 번역 응답마다 이번 묶음에 나온 고유명사(원문 → 한국어 표기)를 같이 받는다
- 한자·가타카나로 쓴 말만 사전에 넣는다 — 이름은 대부분 한자·가타카나이고, 히라가나가 섞인 말은
  お兄ちゃん처럼 문맥 따라 옮김이 달라지는 일반 낱말이 많다
- 사전에 오른 표기는 뒤 요청이 그대로 따른다 — 사전을 받고도 다르게 적어 온 표기는 세지 않는다
- 사전이 비어 있을 때 동시에 돈 요청들이 다르게 옮기면 더 많은 세션이 고른 표기가 사전 표기가 된다. 다만 이름 일부가
  따로 사전에 있으면 그 표기를 담은 쪽을 먼저 고르고, 의심 대사 확인이 고친 표기는 그대로 굳힌다
- 한 인물의 다른 이름(애칭·활동명)은 어느 이름의 변형인지 함께 적어 둔다 — 비슷한 두 이름을 섞지 않게
- 새 요청에는 그 요청에 나오는 사전 표기만 붙이고, 번역이 다 끝나면 진 표기로 쓴 대사를 사전 표기로 고친다
- 대명사(ワシ·オレ)와 그 소리 표기(와시·오레)는 넣지 않는다 — 대명사는 말투로 옮긴다
- 일본 요괴 이름(鬼·天狗 등)은 관례 음역을 처음 요청부터 사전 표기로 붙인다 — 번역이 사전에 올리지 않아도 흔들리지 않게
- 같은 인물의 ちゃん 호칭 꼴(이름만 / 이름 짱)은 책에서 더 많이 쓴 꼴로 맞춘다
- 사전은 출력 폴더의 glossary.json에 둔다 — 이어하기도 같은 표기를 쓴다 (새로 번역하면 새로 만든다)
"""
import difflib
import json
import logging
import re
import threading
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from src.data_models import TranslationStatus
from src.utils import replace_file

logger = logging.getLogger(__name__)

GLOSSARY_FILE = "glossary.json"
_MAX_PROMPT_TERMS = 60   # 한 요청에 붙이는 사전 항목 수 상한
_MAX_TARGET_LEN = 20
_MIN_MIXED_TERM_LEN = 4  # 히라가나가 섞인 용어는 이 글자 수 이상만 사전에 넣는다

_KANJI = "㐀-䶿一-鿿豈-﫿々〆"
_KATAKANA = "ァ-ヺーヽ-ヿㇰ-ㇿｦ-ﾝ"
_KANJI_CHAR = re.compile(f"[{_KANJI}]")
_KATAKANA_CHAR = re.compile(f"[{_KATAKANA}]")
_NAME_LETTER = re.compile(f"[{_KANJI}ァ-ヺㇰ-ㇿｦ-ﾝ]")  # 장음(ー)만으로는 이름으로 치지 않는다
_HIRAGANA = re.compile("[ぁ-ゖゝ-ゟ]")
_JAPANESE = re.compile(f"[ぁ-ゖ{_KANJI}{_KATAKANA}]")
_COUNT_TERM = re.compile(r"[0-9]")  # NFKC 뒤라 전각 숫자도 여기 걸린다

# 이름 뒤에 붙는 호칭 — 사전에는 이름만 둔다
_JP_HONORIFICS = ("ちゃん", "くん", "さん", "さま", "先輩", "先生", "様", "君", "殿", "氏")
_KO_HONORIFICS = ("선생님", "선배", "씨", "군", "짱", "쨩", "님", "양")
# 한자·가타카나로 쓰지만 이름이 아닌 말 (대명사·가족 호칭) — 대명사는 소리가 아니라 말투로 옮긴다
_STOP_SOURCES = {"俺", "僕", "私", "君", "貴様", "貴方", "彼", "彼女", "自分", "兄", "姉", "弟", "妹", "父", "母",
                 "兄貴", "姉貴", "先輩", "先生", "様", "奴", "皆",
                 "儂", "我", "吾", "我輩", "吾輩", "拙者", "某", "小生", "汝", "お主", "御主", "貴殿", "手前", "お前", "御前",
                 "ワシ", "オレ", "ボク", "ワタシ", "ワタクシ", "アタシ", "アタイ", "ウチ", "オイラ", "オラ", "ワイ", "ワガハイ",
                 "オヌシ", "オマエ", "アンタ", "アナタ", "キサマ", "テメエ", "テメー", "テメェ", "ソナタ", "キミ"}
# 대명사를 소리대로 적은 표기 — 원문이 오독(ワシ → フシ)이어도 이 표기로는 사전에 굳히지 않는다
PRONOUN_TARGETS = {"와시", "오레", "보쿠", "와타시", "와타쿠시", "아타시", "아타이", "우치", "오이라", "오라", "와이",
                    "와가하이", "오누시", "오마에", "안타", "아나타", "키사마", "테메", "테메에", "소나타", "키미", "셋샤"}
# 일본 요괴 이름은 음역이 관례 — 번역이 사전에 올리지 않아도 책 첫 요청부터 이 표기로 고정한다
# (요괴·종족을 가리킬 때만 — 鬼ごっこ 같은 낱말은 뜻대로)
_CONVENTIONAL_TERMS = {"鬼": "오니", "天狗": "텐구", "河童": "갓파", "雪女": "유키온나", "座敷童子": "자시키와라시",
                       "座敷わらし": "자시키와라시", "付喪神": "츠쿠모가미", "鵺": "누에", "ろくろ首": "로쿠로쿠비"}
_CONVENTIONAL_EXCEPT = ("鬼ごっこ",)

# 번역문에서 이름 바로 뒤에 올 수 있는 조사·호칭 — 이 밖의 글자가 붙으면 더 긴 낱말의 일부로 본다
_NAME_ENDINGS = ("이에요", "예요", "이었", "였", "이랑", "랑", "이나", "이여", "이라", "라", "이야", "야", "으로", "로",
                 "은", "는", "을", "를", "과", "와", "이", "가", "의", "에게", "에겐", "에서", "에", "께", "도", "만",
                 "한테", "하고", "아", "여", "씨", "군", "짱", "쨩", "님", "선배", "선생", "네", "들", "까지", "부터",
                 "처럼", "보다", "뿐")
# 한 글자 표기는 흔한 낱말과 겹치기 쉬워 조사가 분명할 때만 바꾼다
_SHORT_NAME_ENDINGS = ("이", "가", "은", "는", "을", "를", "의", "에게", "한테", "도", "만", "와", "과", "랑", "이랑",
                       "아", "야", "씨", "군", "짱", "쨩", "님")


def _normalize_source(text) -> str:
    """원문 비교용 — 반각 가나를 전각으로 맞추고 공백·줄바꿈을 뺀다."""
    return "".join(unicodedata.normalize("NFKC", str(text or "")).split())


_KANA_ONLY = re.compile("^[ぁ-ゖァ-ヺー・]+$")
_READING_MARKS = re.compile(r"[\s—·ー・]")  # 띄어쓰기·늘임표·가운뎃점은 소리 비교에서 뺀다 (띄어 쓴 이름과 붙여 쓴 이름을 같게 본다)


def _kana_reading(source: str) -> str:
    """가나로만 쓴 원문을 소리대로 옮긴 한글 (그 밖이면 빈 문자열) — 표 수가 같은 표기를 가를 때 쓴다."""
    if not _KANA_ONLY.match(source or ""):
        return ""
    from src.translator import hangul_for_kana  # translator가 이 모듈을 읽으므로 쓸 때 읽는다
    return _READING_MARKS.sub("", hangul_for_kana(source))


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def _has_final(ch: str) -> bool:
    return _is_hangul(ch) and (ord(ch) - 0xAC00) % 28 != 0


def _final_is_rieul(ch: str) -> bool:
    return _is_hangul(ch) and (ord(ch) - 0xAC00) % 28 == 8


def clean_term(source, target) -> Optional[Tuple[str, str]]:
    """응답의 용어 한 쌍을 사전에 넣을 모양으로 — 한자·가타카나 이름이 아니면 None."""
    source = _normalize_source(source)
    target = " ".join(str(target or "").split())
    for suffix in _JP_HONORIFICS:
        # 奥様처럼 様·殿을 떼면 한 글자만 남는 말은 호칭째 하나의 용어로 둔다 (다른 호칭은 한 글자 이름이어도 뗀다)
        if (source.endswith(suffix) and len(source) > len(suffix)
                and not (len(source) - len(suffix) == 1 and suffix in ("様", "殿"))):
            source = source[:-len(suffix)]
            for ko in _KO_HONORIFICS:
                if target.endswith(ko) and len(target) > len(ko):
                    target = target[:-len(ko)].rstrip()
                    break
            break
    # 히라가나가 섞인 말은 일반 낱말이 많아 빼되, 4자 이상의 이름(곡·작품 이름 등)은 받는다.
    # 숫자로 시작하는 말(수치+일반 명사)은 이름이 아니다 — 사전에 굳으면 문맥에 맞는 말로 못 바꾼다
    if (not source or not target or source in _STOP_SOURCES or target in PRONOUN_TARGETS
            or _COUNT_TERM.match(source)
            or (_HIRAGANA.search(source) and len(source) < _MIN_MIXED_TERM_LEN)
            or not _NAME_LETTER.search(source) or _JAPANESE.search(target)
            or len(target) > _MAX_TARGET_LEN or not any(_is_hangul(ch) for ch in target)):
        return None
    return source, target


def clean_source(source) -> str:
    """원문 표기만 사전 모양으로 — 호칭을 뗀다 (다른 이름이 가리키는 본 이름용)."""
    source = _normalize_source(source)
    for suffix in _JP_HONORIFICS:
        if source.endswith(suffix) and len(source) > len(suffix):
            return source[:-len(suffix)]
    return source


def _script(ch: str) -> str:
    if _KANJI_CHAR.match(ch):
        return "kanji"
    if _KATAKANA_CHAR.match(ch):
        return "katakana"
    return ""


def _appears_as_word(original: str, term: str) -> bool:
    """원문에 term이 더 긴 한자어·가타카나어의 일부가 아니게 나오는지 (田中先輩의 田中는 이름으로 본다)."""
    start = original.find(term)
    while start >= 0:
        end = start + len(term)
        before = original[start - 1] if start else ""
        rest = original[end:]
        joined_before = bool(before) and _script(before) == _script(term[0]) != ""
        joined_after = (bool(rest) and _script(rest[0]) == _script(term[-1]) != ""
                        and not rest.startswith(_JP_HONORIFICS))
        if not joined_before and not joined_after:
            return True
        start = original.find(term, start + 1)
    return False


def _conventional_use(haystack: str, term: str) -> bool:
    """관례 표기 용어가 요괴·종족 이름으로 나오는지 — 더 긴 한자어의 일부(吸血鬼)나 鬼ごっこ 같은 낱말은 뺀다."""
    for word in _CONVENTIONAL_EXCEPT:
        haystack = haystack.replace(word, "")
    return term in haystack and _appears_as_word(haystack, term)


def _fix_particle(rest: str, old: str, new: str) -> str:
    """표기를 바꾼 뒤 바로 뒤 조사를 새 표기의 받침에 맞춘다 (rest = 이름 뒤 글자들)."""
    old_final, new_final = _has_final(old[-1]), _has_final(new[-1])
    if old_final == new_final:
        if old_final and _final_is_rieul(new[-1]) != _final_is_rieul(old[-1]):
            if rest.startswith("으로") and _final_is_rieul(new[-1]):
                return "로" + rest[2:]
            if rest.startswith("로") and _final_is_rieul(old[-1]):
                return "으" + rest
        return rest
    if old_final:  # 받침 있음 → 없음
        for a, b in (("이에요", "예요"), ("이었", "였"), ("으로", "로"), ("은", "는"), ("을", "를"), ("과", "와")):
            if rest.startswith(a):
                return b + rest[len(a):]
        if rest.startswith("이"):
            # 켄이야 → 하루토야 (이다의 '이'는 빠진다), 켄이 → 하루토가
            return rest[1:] if len(rest) > 1 and _is_hangul(rest[1]) else "가" + rest[1:]
        if rest.startswith("아") and not (len(rest) > 1 and _is_hangul(rest[1])):
            return "야" + rest[1:]
        return rest
    # 받침 없음 → 있음 (뜻이 둘로 갈리는 야·나는 그대로 둔다)
    for a, b in (("예요", "이에요"), ("였", "이었"), ("랑", "이랑"), ("라", "이라"), ("는", "은"), ("를", "을"),
                 ("와", "과"), ("가", "이")):
        if rest.startswith(a):
            return b + rest[len(a):]
    if rest.startswith("로") and not _final_is_rieul(new[-1]):
        return "으" + rest
    return rest


def _inside(text: str, at: int, part: str, whole: str) -> bool:
    """text[at:]의 part가 whole 표기의 일부로 쓰인 것인지 (하루 → 하루토에서 '하루토'는 건드리지 않는다)."""
    offset = whole.find(part)
    while offset >= 0:
        if at - offset >= 0 and text[at - offset:at - offset + len(whole)] == whole:
            return True
        offset = whole.find(part, offset + 1)
    return False


def _rendering_spots(text: str, old: str, new: str) -> List[int]:
    """번역문에서 old 표기가 낱말(이름)로 쓰인 자리들."""
    spots = []
    at = text.find(old)
    while at >= 0:
        rest = text[at + len(old):]
        if len(old) == 1:
            # 한 글자는 '할 수'·'난 몰라'처럼 흔한 낱말과 겹친다 — 조사나 문장부호가 바로 붙을 때만
            ends = not rest or (not _is_hangul(rest[0]) and not rest[0].isspace()) or rest.startswith(_SHORT_NAME_ENDINGS)
        else:
            ends = not rest or not _is_hangul(rest[0]) or rest.startswith(_NAME_ENDINGS)
        if ends and not (at and _is_hangul(text[at - 1])) and not _inside(text, at, old, new):
            spots.append(at)
        at = text.find(old, at + 1)
    return spots


def replace_rendering(text: str, old: str, new: str, most: Optional[int] = None) -> str:
    """번역문에서 낱말로 쓰인 old 표기를 new로 바꾸고 조사를 받침에 맞춘다.

    most를 주면 바꿀 자리가 그보다 많을 때 손대지 않는다 (원문의 이름 수보다 많으면 흔한 낱말이 섞였다).
    """
    if not old or old == new:
        return text
    spots = _rendering_spots(text, old, new)
    if most is not None and len(spots) > most:
        return text
    for at in reversed(spots):  # 뒤에서부터 바꿔야 앞 자리가 밀리지 않는다
        text = text[:at] + new + _fix_particle(text[at + len(old):], old, new)
    return text


_CHAN = "ちゃん"
_KO_CHAN = re.compile(r"([가-힣]{1,6})( ?)([짱쨩])")


def _chan_name_run(original: str, at: int) -> str:
    """원문 at 자리(ちゃん) 바로 앞 이름 — 바로 앞 글자와 같은 문자 종류(한자·가타카나·히라가나)가 이어진 만큼."""
    kinds = (_KANJI_CHAR, _KATAKANA_CHAR, _HIRAGANA)
    if at <= 0:
        return ""
    kind = next((k for k in kinds if k.match(original[at - 1])), None)
    if kind is None:
        return ""
    start = at
    while start > 0 and kind.match(original[start - 1]):
        start -= 1
    return original[start:at]


def unify_honorifics(pages) -> List[Tuple[str, str, str]]:
    """같은 인물의 ちゃん 호칭을 책 전체에서 한 꼴로 맞춘다 — 쪽마다 '이름'과 '이름 짱'이 섞이지 않게.

    ちゃん 하나에 '이름 짱' 하나로 옮긴 대사에서 원문 이름과 한국어 이름을 짝짓고, 그 이름이 나오는
    대사들 가운데 더 많이 쓴 꼴(같으면 앞 쪽에서 쓴 꼴)로 나머지를 고친다. 한 대사에 아는 이름이 둘 이상이면 건드리지 않는다.
    고친 대사마다 (원문 이름+ちゃん, 옛 표기, 새 표기)를 돌려준다.
    """
    rows = []
    for page in pages:
        for element in page.text_elements():
            if element.translation_status != TranslationStatus.TRANSLATED or not element.translated_text:
                continue
            original = _normalize_source(element.original_text)
            if _CHAN in original:
                rows.append((element, original))
    # 한국어 이름 → 원문 이름 후보 (ちゃん 하나·짱 하나인 대사에서)
    runs: Dict[str, List[str]] = {}
    spelling: Dict[str, Counter] = {}
    for element, original in rows:
        matches = list(_KO_CHAN.finditer(element.translated_text))
        if original.count(_CHAN) != 1 or len(matches) != 1:
            continue
        run = _chan_name_run(original, original.find(_CHAN))
        if run:
            name = matches[0].group(1)
            runs.setdefault(name, []).append(run)
            spelling.setdefault(name, Counter())[matches[0].group(2) + matches[0].group(3)] += 1
    sources = {}
    for name, found in runs.items():
        shortest = min(found, key=len)
        if all(run.endswith(shortest) for run in found):  # 모든 꼴이 가장 짧은 꼴로 끝나면 그 꼴로 (앞 감탄사가 붙어 읽힌 이름)
            sources[name] = shortest
    changes = []
    for name, source in sources.items():
        # 짱은 이름에 붙여 쓴다 (한국어판 표기) — 띄어 쓰면 식자가 이름과 짱 사이에서 줄을 바꿀 수 있다
        chan = spelling[name].most_common(1)[0][0].strip()
        forms = []  # (대사, 원문, 짱을 붙였는지) — 읽는 순서
        for element, original in rows:
            text = element.translated_text
            if source + _CHAN not in original or sum(1 for other in sources if _rendering_spots(text, other, "")) > 1:
                continue
            if any(name + suffix in text for suffix in (" 짱", "짱", " 쨩", "쨩")):
                spaced = text
                for suffix in (" 짱", " 쨩"):
                    spaced = replace_rendering(spaced, name + suffix, name + suffix.strip())
                if spaced != text:
                    logger.info(f"호칭 붙여 쓰기 ({source}{_CHAN}): {text} → {spaced}")
                    changes.append((source + _CHAN, text, spaced))
                    element.translated_text = spaced
                forms.append((element, original, True))
            elif _rendering_spots(text, name, name + chan):
                forms.append((element, original, False))
        with_chan = sum(1 for form in forms if form[2])
        if not with_chan or with_chan == len(forms):
            continue
        keep = with_chan > len(forms) - with_chan or (with_chan * 2 == len(forms) and forms[0][2])
        for element, original, has_chan in forms:
            if has_chan == keep:
                continue
            text = element.translated_text
            if keep:
                fixed = replace_rendering(text, name, name + chan, most=original.count(source + _CHAN))
            else:
                fixed = text
                for suffix in (" 짱", "짱", " 쨩", "쨩"):
                    fixed = replace_rendering(fixed, name + suffix, name)
            if fixed != text:
                logger.info(f"호칭 꼴 맞춤 ({source}{_CHAN}): {text} → {fixed}")
                changes.append((source + _CHAN, name if keep else name + chan, name + chan if keep else name))
                element.translated_text = fixed
    return changes


class Glossary:
    """한 책의 인명·용어 사전 — 여러 번역 세션이 같이 쓰므로 잠금으로 지킨다."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._lock = threading.Lock()
        # 원문 → {표기: {"sessions": 고른 세션 수, "requests": 알려 온 응답 수, "order": 처음 받은 순서,
        #               "fixed": 의심 대사 확인이 고친 표기면 1}}
        self._votes: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._aliases: Dict[str, str] = {}  # 다른 이름(애칭·활동명) → 본 이름 원문
        self._voters = set()  # (원문, 표기, 세션) — 이번 실행에서 한 세션은 한 표기에 한 표만
        self._order = 0
        self._dirty = False

    @classmethod
    def load(cls, path: str) -> "Glossary":
        """저장해 둔 사전을 읽는다 (없거나 깨졌으면 빈 사전)."""
        glossary = cls(path)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for source, targets in data.get("terms", {}).items():
                glossary._votes[str(source)] = {
                    str(target): {key: int(vote.get(key, 0)) for key in ("sessions", "requests", "order", "fixed")}
                    for target, vote in targets.items()
                }
            glossary._aliases = {str(k): str(v) for k, v in (data.get("aliases") or {}).items()}
        except (OSError, ValueError, AttributeError, TypeError):
            glossary._votes, glossary._aliases = {}, {}
        orders = [vote["order"] for targets in glossary._votes.values() for vote in targets.values()]
        glossary._order = max(orders, default=-1) + 1
        return glossary

    def _winner(self, source: str) -> str:
        """사전 표기 — 의심 대사 확인이 고친 표기, 이름 일부의 사전 표기를 담은 표기, 많은 세션, 많은 응답, 가나 원문이면 소리대로
        옮긴 꼴에 가까운 표기, 먼저 받은 표기 순. 표 수가 같을 때 먼저 받은 표기만 보면 한 세션이 잘못 옮긴 표기가 남을 수 있다."""
        targets = self._votes[source]
        if len(targets) == 1:
            return next(iter(targets))
        parts = [self._winner(other) for other in self._votes
                 if other != source and other in source and len(self._votes[other]) == 1
                 and _appears_as_word(source, other)]
        reading = _kana_reading(source)

        def rank(item):
            target, vote = item
            consistent = sum(1 for part in parts if part in target)
            near = difflib.SequenceMatcher(None, _READING_MARKS.sub("", target), reading).ratio() if reading else 0.0
            return vote.get("fixed", 0), consistent, vote["sessions"], vote["requests"], near, -vote["order"]
        return max(targets.items(), key=rank)[0]

    def record(self, session_key: str, terms, source_texts: Iterable[str], given: Optional[Dict[str, str]] = None) -> int:
        """응답 하나의 용어를 센다. 받아들인 수를 돌려준다.

        원문에 없는 말(모델이 고쳐 적은 표기)은 믿지 않는다. given(요청에 붙여 보낸 사전 표기)과 다르게 적어 온
        표기도 세지 않는다 — 한 번 정한 표기는 뒤 요청이 따른다.
        """
        haystack = "\n".join(_normalize_source(text) for text in source_texts)
        given = given or {}
        taken = 0
        with self._lock:
            for item in terms or []:
                pair = clean_term(item.get("source"), item.get("target")) if isinstance(item, dict) else None
                if not pair or pair[0] not in haystack:
                    continue
                source, target = pair
                if source in given and given[source] != target:
                    logger.info(f"사전 표기와 다르게 적어 온 표기는 세지 않습니다: {source} → {target} (사전 {given[source]})")
                    continue
                alias_of = clean_source(item.get("alias_of"))
                if alias_of and alias_of != source and _NAME_LETTER.search(alias_of):
                    self._aliases.setdefault(source, alias_of)
                vote = self._votes.setdefault(source, {}).get(target)
                if vote is None:
                    vote = self._votes[source][target] = {"sessions": 0, "requests": 0, "order": self._order}
                    self._order += 1
                vote["requests"] += 1
                if (source, target, session_key) not in self._voters:
                    self._voters.add((source, target, session_key))
                    vote["sessions"] += 1
                taken += 1
            self._dirty = self._dirty or taken > 0
        return taken

    def fix(self, source, target) -> bool:
        """의심 대사 확인이 고친 표기를 사전 표기로 굳힌다. 사전 모양이 아니면 False."""
        pair = clean_term(source, target)
        if not pair:
            return False
        source, target = pair
        with self._lock:
            targets = self._votes.setdefault(source, {})
            for vote in targets.values():
                vote.pop("fixed", None)
            vote = targets.setdefault(target, {"sessions": 0, "requests": 0, "order": self._order})
            if vote["order"] == self._order:
                self._order += 1
            vote["fixed"] = 1
            self._dirty = True
        return True

    def terms_for(self, texts: Iterable[str]) -> List[Tuple[str, str]]:
        """이 글들에 나오는 사전 항목 (원문, 사전 표기) — 많은 세션이 쓴 말부터."""
        haystack = "\n".join(_normalize_source(text) for text in texts)
        with self._lock:
            found = [(source, targets) for source, targets in self._votes.items()
                     if source in haystack and _appears_as_word(haystack, source)]
            found.sort(key=lambda item: -sum(vote["sessions"] for vote in item[1].values()))
            terms = [(source, self._winner(source)) for source, _ in found[:_MAX_PROMPT_TERMS]]
            known = {source for source, _ in terms} | set(self._votes)
        return terms + [(source, target) for source, target in _CONVENTIONAL_TERMS.items()
                        if source not in known and _conventional_use(haystack, source)]

    def prompt_lines(self, texts: Iterable[str]) -> List[str]:
        """요청에 붙일 사전 줄 — 다른 이름이면 본 이름과 그 표기를 함께 적는다 (다른 이름 → 표기 · 본 이름(표기)의 다른 이름)."""
        terms = self.terms_for(texts)
        lines = []
        with self._lock:
            for source, target in terms:
                line = f"- {source} → {target}"
                main = self._aliases.get(source)
                if main and main in self._votes:
                    line += f" · {main}({self._winner(main)})의 다른 이름"
                if source not in self._votes and source in _CONVENTIONAL_TERMS:
                    line += " (요괴·종족 이름일 때 — 관례 음역)"
                lines.append(line)
        return lines

    def given_map(self, texts: Iterable[str]) -> Dict[str, str]:
        """요청에 붙인 사전 표기 (원문 → 표기) — 응답을 셀 때 다르게 적어 온 표기를 가린다."""
        return dict(self.terms_for(texts))

    def refresh(self, pages) -> List[Tuple[str, str, str]]:
        """진 표기로 쓴 대사를 사전 표기로 고치고, 인물마다 ちゃん 호칭 꼴을 하나로 맞춘다.
        고친 대사마다 (원문, 옛 표기, 사전 표기)를 돌려준다."""
        with self._lock:
            conflicts = {source: (self._winner(source), sorted(targets, key=len, reverse=True))
                         for source, targets in self._votes.items() if len(targets) > 1}
        changes = self._refresh_terms(pages, conflicts) if conflicts else []
        return changes + unify_honorifics(pages)

    def _refresh_terms(self, pages, conflicts) -> List[Tuple[str, str, str]]:
        changes = []
        for page in pages:
            for element in page.text_elements():
                if element.translation_status != TranslationStatus.TRANSLATED or not element.translated_text:
                    continue
                original = _normalize_source(element.original_text)
                text = element.translated_text
                for source, (winner, renderings) in conflicts.items():
                    # 이미 사전 표기로 쓴 대사는 그대로 — 옛 표기와 같은 글자는 이름이 아닌 흔한 낱말일 수 있다
                    if winner in text or source not in original or not _appears_as_word(original, source):
                        continue
                    for rendering in renderings:
                        if rendering == winner:
                            continue
                        fixed = replace_rendering(text, rendering, winner, most=original.count(source))
                        if fixed != text:
                            changes.append((source, rendering, winner))
                            text = fixed
                element.translated_text = text
        return changes

    def save(self):
        """바뀐 게 있으면 출력 폴더에 저장한다."""
        if not self.path:
            return
        with self._lock:
            if not self._dirty:
                return
            data = {"terms": {source: {target: dict(vote) for target, vote in targets.items()}
                              for source, targets in self._votes.items()},
                    "aliases": dict(self._aliases)}
            self._dirty = False
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            replace_file(tmp, self.path)  # Windows에서 누가 잠깐 열어 두었으면 기다렸다 다시
        except OSError as exc:
            logger.warning(f"인명·용어 사전을 저장하지 못했습니다: {exc}")
