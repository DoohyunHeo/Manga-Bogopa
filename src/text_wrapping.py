"""Text wrapping strategies used by the layout and fitting modules.

Each wrap function has the signature:
    wrap(text, font_path, font_size, style, max_width) -> str

build_wrap_candidates() picks the right combination of strategies based on
text density and bubble aspect ratio.
"""
import re
from functools import lru_cache

# 한국어 조판 금칙은 세로쓰기(단 나눔)에서도 쓰이므로 렌더러에 정의되어 있다.
from src.text_renderer import LINE_HEAD_FORBIDDEN, LINE_TAIL_FORBIDDEN, measure_line

# Shared tall-bubble thresholds (used by wrapping + fitting + style resolvers).
TALL_BUBBLE_RATIO = 1.8
TALL_BUBBLE_MIN_CHARS = 8

# 행두/행말 금칙 위반 벌점 — 줄 선택(DP)과 후보 선택(fitting 스코어러)이
# 같은 강도로 벌점을 매기도록 공유한다.
FORBIDDEN_LINE_PENALTY = 8.0


def visible_len(text: str) -> int:
    """공백을 제외한 글자 수."""
    return len(re.sub(r"\s+", "", text or ""))


# 의미상 별칭: 밀도(줄바꿈 전략 선택)와 길이(줄 미관 판단)를 같은 측정으로 쓴다.
text_density = visible_len


def is_layout_equivalent(source, candidate):
    """candidate가 source(번역 원문)의 글자와 어절 경계를 보존하고 '줄바꿈만' 바꾼 결과인지.

    허용: 어절 사이 공백이 그 자리에서 줄바꿈으로 바뀜, 공백이 없던 자리에
          줄바꿈 추가(글자 단위 줄바꿈), 공백 런이 다른 공백 런으로 바뀜.
    금지: 어절 공백 삭제(붙여쓰기), 공백 삽입, 원문 단락 줄바꿈('\\n') 삭제,
          글자 변경·삭제·추가.
    식자 검수·페이지 통일이 만드는 줄바꿈 대안은 이 검사를 통과해야만 쓰인다.
    """
    s = source or ""
    c = candidate or ""
    i = j = 0
    while i < len(s) or j < len(c):
        if i < len(s) and s[i].isspace():
            had_newline = False
            while i < len(s) and s[i].isspace():
                had_newline = had_newline or s[i] == "\n"
                i += 1
            cand_ws = ""
            while j < len(c) and c[j].isspace():
                cand_ws += c[j]
                j += 1
            if not cand_ws:
                return False  # 어절 공백 삭제
            if had_newline and "\n" not in cand_ws:
                return False  # 원문 단락 줄바꿈 삭제
            continue
        if j < len(c) and c[j].isspace():
            while j < len(c) and c[j].isspace():
                if c[j] != "\n":
                    return False  # 공백 삽입 (줄바꿈 추가만 허용)
                j += 1
            continue
        if i >= len(s) or j >= len(c) or s[i] != c[j]:
            return False
        i += 1
        j += 1
    return True


def count_midword_breaks(source_text, wrapped_text):
    """줄 경계 중 원문의 공백(어절 경계)과 일치하지 않는 곳의 수.

    공백을 제거한 누적 글자 수로 비교하므로 줄바꿈 과정에서 공백이
    소비/복원되어도 정확하다. 어절 안 활용 꼬리 앞('감개무량/해요'·'수고/했어')은 세지 않는다 — 좁은
    말풍선에서 끊어도 자연스러운 자리다. 그 자리는 count_tail_breaks가 센다.
    """
    word_break_positions = set()
    n = 0
    for ch in source_text or "":
        if ch in (' ', '\n'):
            word_break_positions.add(n)
        else:
            n += 1
    word_break_positions |= _tail_start_offsets(source_text)
    breaks = 0
    cum = 0
    lines = [line for line in (wrapped_text or "").split('\n') if line.strip()]
    for line in lines[:-1]:
        cum += len(line.replace(' ', ''))
        if cum not in word_break_positions:
            breaks += 1
    return breaks


def count_tail_breaks(source_text, wrapped_text):
    """줄 경계 가운데 어절 안 활용 꼬리 앞에서 끊은 곳의 수 ('감개무량/해요'는 1)."""
    tails = _tail_start_offsets(source_text)
    if not tails:
        return 0
    breaks = 0
    cum = 0
    lines = [line for line in (wrapped_text or "").split('\n') if line.strip()]
    for line in lines[:-1]:
        cum += len(line.replace(' ', ''))
        breaks += cum in tails
    return breaks


def _tail_start_offsets(source_text):
    """활용 꼬리가 시작하는 자리들 — 공백을 뺀 누적 글자 수 (count_midword_breaks와 같은 셈)."""
    starts = set()
    offset = 0
    for paragraph in (source_text or "").split("\n"):
        nonspace_index = []
        count = 0
        for ch in paragraph:
            nonspace_index.append(count)
            if ch != ' ':
                count += 1
        for start in _verb_tail_clusters(paragraph):
            starts.add(offset + nonspace_index[start])
        offset += count
    return starts


def count_single_char_lines(text):
    """보이는 글자가 한 개뿐인 줄의 수 ('더', '게' 처럼 한 글자만 남은 줄)."""
    return sum(1 for line in (text or "").split("\n") if visible_len(line) == 1)


def count_lone_middle_lines(text):
    """첫 줄·끝줄을 뺀 가운데 줄 가운데 보이는 글자가 한 개뿐인 줄의 수 ('싫어, / 안 / 돌려줘!'는 1)."""
    lines = [line for line in (text or "").split("\n") if line.strip()]
    return sum(1 for line in lines[1:-1] if visible_len(line) == 1)


def worsens_line_breaks(source, base_text, candidate_text):
    """candidate가 base보다 단어 중간 줄바꿈·어미 앞 줄바꿈·한 글자 줄을 늘리는지.

    검수·통일이 만드는 줄바꿈 대안은 기존 식자 규칙(단어 중간 줄바꿈은 마지막 수단)을
    우회하면 안 되므로, 이 검사에 걸리는 대안은 쓰지 않는다.
    """
    return (count_midword_breaks(source, candidate_text) > count_midword_breaks(source, base_text)
            or count_unnatural_breaks(source, candidate_text) > count_unnatural_breaks(source, base_text)
            or count_single_char_lines(candidate_text) > count_single_char_lines(base_text))


def wrap_text_korean(text, font_path, font_size, style, max_width):
    lines = []
    for paragraph in text.split('\n'):
        words = paragraph.split()
        if not words:
            continue

        current_line = ""
        for word in words:
            candidate = word if not current_line else f"{current_line} {word}"
            if measure_line(candidate, font_path, font_size, style) <= max_width:
                current_line = candidate
                continue

            if current_line:
                lines.append(current_line)
                current_line = ""

            current_line = word

        if current_line:
            lines.append(current_line)

    return "\n".join(lines)


def _line_layout_penalty(line_text, line_width, max_width, target_ratio, is_last_line):
    stripped = line_text.strip()
    if not stripped:
        return 100.0

    width_ratio = line_width / max(max_width, 1)
    desired_ratio = 0.68 if is_last_line else target_ratio
    penalty = 0.0

    if width_ratio < desired_ratio:
        penalty += (desired_ratio - width_ratio) * 18.0
    else:
        penalty += (width_ratio - desired_ratio) * 7.0

    if stripped[0] in LINE_HEAD_FORBIDDEN:
        penalty += FORBIDDEN_LINE_PENALTY
    if stripped[-1] in LINE_TAIL_FORBIDDEN:
        penalty += FORBIDDEN_LINE_PENALTY
    if is_last_line and visible_len(stripped) <= 2:
        penalty += 5.0

    return penalty


def wrap_text_balanced(text, font_path, font_size, style, max_width, target_lines):
    wrapped_paragraphs = []

    for paragraph in text.split('\n'):
        words = paragraph.split()
        if not words:
            continue

        if len(words) <= 1 or target_lines <= 1:
            wrapped_paragraphs.append(wrap_text_korean(paragraph, font_path, font_size, style, max_width))
            continue

        tokens = words
        line_count = min(target_lines, len(tokens))
        target_ratio = 0.86 if line_count <= 2 else 0.8
        token_count = len(tokens)
        widths = {}

        for start in range(token_count):
            line_text = ""
            for end in range(start, token_count):
                line_text = tokens[end] if not line_text else f"{line_text} {tokens[end]}"
                widths[(start, end + 1)] = (line_text, measure_line(line_text, font_path, font_size, style))

        inf = float("inf")
        dp = [[inf] * (line_count + 1) for _ in range(token_count + 1)]
        prev = [[None] * (line_count + 1) for _ in range(token_count + 1)]
        dp[0][0] = 0.0

        for start in range(token_count):
            for used_lines in range(line_count):
                if dp[start][used_lines] == inf:
                    continue

                remaining_lines = line_count - used_lines
                max_end = token_count - (remaining_lines - 1)
                for end in range(start + 1, max_end + 1):
                    line_text, line_width = widths[(start, end)]
                    penalty = _line_layout_penalty(
                        line_text,
                        line_width,
                        max_width,
                        target_ratio,
                        used_lines == line_count - 1,
                    )
                    total = dp[start][used_lines] + penalty
                    if total < dp[end][used_lines + 1]:
                        dp[end][used_lines + 1] = total
                        prev[end][used_lines + 1] = start

        if dp[token_count][line_count] == inf:
            wrapped_paragraphs.append(wrap_text_korean(paragraph, font_path, font_size, style, max_width))
            continue

        lines = []
        end = token_count
        used_lines = line_count
        while used_lines > 0:
            start = prev[end][used_lines]
            if start is None:
                break
            line_text, _ = widths[(start, end)]
            lines.append(line_text)
            end = start
            used_lines -= 1

        lines.reverse()
        wrapped_paragraphs.append("\n".join(lines) if lines else wrap_text_korean(paragraph, font_path, font_size, style, max_width))

    return "\n".join(wrapped_paragraphs)


def _make_balanced_wrap(target_lines):
    def _wrap(text, font_path, font_size, style, max_width):
        return wrap_text_balanced(text, font_path, font_size, style, max_width, target_lines)

    _wrap.__name__ = f"balanced_{target_lines}"
    return _wrap


def wrap_text(text, font_path, font_size, style, max_width):
    """Default space-separated wrap with punctuation-aware token handling."""
    lines = []
    for paragraph in text.split('\n'):
        words = re.findall(r'(·+|[!?]+|\S+)', paragraph)
        if not words:
            continue
        current_line = words[0]
        for word in words[1:]:
            joiner = "" if re.match(r'^(·+|[!?⋯]+)$', word) else " "
            if measure_line(current_line + joiner + word, font_path, font_size, style) <= max_width:
                current_line += joiner + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return "\n".join(lines)


# 강제 개행 시에도 직전 글자에 붙어 함께 움직여야 하는 기호들 (~~!! 등 연속 포함)
_TRAILING_STICKY = frozenset(")]}〉》」』】〕’”.。,，、!！?？…⋯‥·~〜‼⁉♪♬♡♥☆★―—–-:;")
# 다음 줄로 함께 딸려가면 읽기 편한 한국어 어미/조사 단위 (뒤에 한글이 아닌
# 문자(구두점·공백·끝)가 올 때만 한 덩어리로 묶는다 — '이야기'의 '이야' 오인 방지)
# 선어말+종결 조합·인용·감탄 계열 포함. '이지'(페이지·스테이지)·'이다'
# (사이다·보이다)처럼 일반 단어 끝과 오인되기 쉬운 항목은 제외.
# 긴 단위가 먼저 매칭되도록 길이 내림차순 정렬 (라고요 > 라고, 습니까 > 니까).
_KO_ENDING_UNITS = tuple(sorted({
    # 격식체/존댓말
    "십시오", "습니다", "입니다", "됩니다", "합니다", "습니까", "잖습니까",
    "답니다", "랍니다",
    # 의문/반문
    "인가", "인지", "건가", "건지", "는가", "은가", "느냐", "더냐", "거냐", "거니",
    "을까", "까요", "나요", "을래", "든가", "든지", "던가", "던지",
    "었냐", "았냐", "였냐", "겠냐", "었니", "았니", "였니", "겠니", "잖니", "잖냐",
    # 평서/감탄
    "이야", "이죠", "이네", "이군", "거야", "거지", "거든", "구나", "구만", "구먼",
    "는군", "더군", "로군", "로구나", "로구만", "는걸", "은걸", "을걸",
    "걸요", "는걸요", "은걸요", "을걸요", "군요", "네요",
    "었어", "았어", "였어", "겠어", "었지", "았지", "였지", "겠지",
    "었네", "았네", "였네", "겠네", "한다", "했다", "겠다", "단다", "란다",
    # 인용/전달
    "라고", "다고", "냐고", "자고", "라고요", "다고요", "냐고요", "자고요",
    "대요", "래요", "라니", "라며", "다며", "냐며", "자며", "다니", "냐니", "자니",
    "더라", "더라고", "더라구", "더라고요", "더라구요",
    "라더라", "다더라", "래더라", "대더라",
    # 연결/종결 공통
    "라면", "다면", "냐면", "자면", "면서", "니까", "라니까", "다니까", "냐니까",
    "자니까", "는데", "은데", "던데", "지만", "잖아", "잖아요",
    "세요", "어요", "아요", "여요", "예요", "에요", "해요", "가요", "게요",
    "니다", "시오", "지요", "는데요", "은데요", "던데요", "거든요",
    "면서요", "다면서", "다면서요", "을게", "을게요",
}, key=len, reverse=True))
_KO_ENDING_UNIT_SET = frozenset(_KO_ENDING_UNITS)


def _is_hangul(ch):
    return '가' <= ch <= '힣'


# ── 어절 안 활용 꼬리 ────────────────────────────────────────────────────────
# 단어 중간에서 끊어야 할 때 자연스러운 자리는 동사가 시작되는 곳이다:
# 체언 + '하다·되다·있다·없다'(수고/했어, 안녕/하세요, 재미/있어), '-아/어' 뒤
# 보조용언(귀여워/졌네, 구해/주셨다, 지워/버렸어, 해/봤어), 체언 + '이다'(스포츠맨/이구나). 그 자리부터 어절 끝까지를
# 한 덩어리로 묶어, 어미만 다음 줄로 넘어가는 줄바꿈('수고했/어!')을 막는다.
_LIGHT_VERB_SYLLABLES = frozenset("하한할함합해했되된될됨됩돼됐있없")
_AUX_VERB_HEADS = (
    "버리", "버린", "버릴", "버림", "버립", "버려", "버렸",
    "드리", "드린", "드릴", "드림", "드립", "드려", "드렸",
    "주", "준", "줄", "줌", "줍", "줘", "줬",
    "보", "본", "볼", "봄", "봅", "봐", "봤",
    "두", "둔", "둘", "둠", "둡", "둬", "뒀",
    "내", "낸", "낼", "냄", "냅", "냈",
)
# '지다'는 조사 '까지'(앞까지는)·'가지다'(가지고)·'마지막'과 헷갈리기 쉬워 따로 좁게 본다
_AUX_JI_HEADS = ("지", "진", "질", "짐", "져", "졌")
# '-아/어 가다·오다'(쫓아/가요, 들어/와요) — 꼬리가 한 글자면 조사 '가·와'(여자가·사과와)와 같아 두 글자 이상만
_AUX_GO_HEADS = ("가", "간", "갈", "감", "갑", "갔", "와", "왔", "온", "올", "옴", "옵")
# '-아/어'로 끝난 음절의 모음 (받침 없음): ㅏ ㅐ ㅓ ㅕ ㅘ ㅙ ㅝ — 먹어·구해·보여·귀여워
_A_EO_MEDIALS = frozenset((0, 1, 4, 6, 9, 10, 14))
# 이 음절들로만 된 어절은 감탄사로 보고 꼬리를 찾지 않는다 (아하하하·으헤헤)
_INTERJECTION_SYLLABLES = frozenset("아어오우으이에야와워하히헤호후흐크캬꺄푸악앗윽욱잇읏엣흑헉")
# 이보다 긴 꼬리는 끝의 어미 단위 앞까지만 묶는다 (좁은 말풍선에서 못 들어가는 덩어리 방지)
_VERB_TAIL_MAX_CHARS = 4
# 받침 있는 체언 + '이다' 활용 세 글자 이상('스포츠맨/이구나'·'사람/이냐고') — 좁은
# 말풍선에서 끊어도 자연스러운 자리. 두 글자 꼬리(이야·이다)와 ㅇ받침 뒤('고양이'·'꼬맹이'의 '이')는 '이'로 끝나는 낱말과
# 헷갈려 뺀다. '인·일'로 시작하는 꼬리(인데·인가·일까)는 그런 혼동이 없어 두 글자도 넣고 받침도 가리지 않는다
# (친구/인데·동생/인데)
_COPULA_TAILS = frozenset((
    "인데", "인데도", "인가", "인지", "인걸", "일까", "일걸", "일지",
    "이구나", "이구만", "이로구나", "이었어", "이었지", "이었다", "이었네", "이었군", "이었냐", "이었니", "이었는데",
    "이었으니", "이었어요", "이었지만", "이었잖아", "이었구나", "이라고", "이라서", "이라면", "이라니", "이라며",
    "이라니까", "이라고요", "이라든가", "이라든지", "이냐고", "이냐고요", "이니까", "이니까요", "이잖아", "이잖아요",
    "이에요", "입니다", "입니까", "이거든", "이거든요", "이더라", "이더라고", "이던데", "이겠지", "이겠어", "이겠네",
    "인가요", "인데요", "인걸요", "일걸요", "일까요", "일지도", "인지도",
))
_FINAL_NG = 21  # 받침 ㅇ의 번호


def _final_consonant(ch):
    """한글 음절의 받침 번호 (없거나 한글이 아니면 0)."""
    return (ord(ch) - 0xAC00) % 28 if _is_hangul(ch) else 0


def _ends_with_a_eo(ch):
    if not _is_hangul(ch):
        return False
    code = ord(ch) - 0xAC00
    return code % 28 == 0 and (code // 28) % 21 in _A_EO_MEDIALS


def _verb_tail_starts(word):
    """한글 어절 안에서 활용 꼬리가 시작할 수 있는 위치들 (앞에서부터)."""
    if len(word) < 2 or all(ch in _INTERJECTION_SYLLABLES for ch in word):
        return
    for k in range(1, len(word)):
        ch = word[k]
        if k >= 2 and ch in _LIGHT_VERB_SYLLABLES:
            if word[k + 1:k + 2] in _LIGHT_VERB_SYLLABLES:
                continue  # '투합하고'의 '합'처럼 체언 끝 글자 — 동사는 다음 글자부터
            if word[k - 1] == "당" and k >= 3:
                yield k - 1  # 습격/당하면
            yield k
        elif k >= 2 and ch == "시" and word[k + 1:k + 2] in ("키", "켜", "켰", "킨", "킬"):
            yield k  # 감동/시켰다
        elif k >= 2 and word[k:] in _COPULA_TAILS and (word[k] in "인일"
                                                         or _final_consonant(word[k - 1]) not in (0, _FINAL_NG)):
            yield k  # 스포츠맨/이구나 · 사람/인데
        elif _ends_with_a_eo(word[k - 1]) and (k >= 2 or word[0] in ("해", "돼")):
            if word[k:] not in _KO_ENDING_UNIT_SET and (  # '하지만'의 '지만'처럼 어미 자체인 꼬리는 제외
                    word.startswith(_AUX_VERB_HEADS, k)
                    or (word[k - 1] != "까" and word.startswith(_AUX_JI_HEADS, k))):
                yield k
            elif len(word) - k >= 2 and word.startswith(_AUX_GO_HEADS, k):
                yield k  # 쫓아/가요 — '가요'는 어미 목록에도 있지만 받침 없는 '-아/어' 뒤라 '가다'다


def _verb_tail_clusters(paragraph):
    """문단 안 활용 꼬리의 {시작 위치: 한 덩어리로 묶을 길이}."""
    tails = {}
    i, n = 0, len(paragraph)
    while i < n:
        if not _is_hangul(paragraph[i]):
            i += 1
            continue
        j = i
        while j < n and _is_hangul(paragraph[j]):
            j += 1
        word = paragraph[i:j]
        for start in _verb_tail_starts(word):
            tail = word[start:]
            if len(tail) <= _VERB_TAIL_MAX_CHARS:
                tails[i + start] = len(tail)
                break
            ending = next((u for u in _KO_ENDING_UNITS if tail.endswith(u) and len(u) < len(tail)), None)
            if ending and len(tail) - len(ending) <= _VERB_TAIL_MAX_CHARS:
                tails[i + start] = len(tail) - len(ending)  # 체포/하겠/습니다
                break
        i = j
    return tails


def count_unnatural_breaks(source_text, wrapped_text):
    """활용 꼬리 안(어미 앞 등)에서 끊긴 줄 경계 수 — '수고했/어!'는 1, '수고/했어!'는 0.

    위치는 count_midword_breaks와 같이 공백을 뺀 누적 글자 수로 비교한다.
    """
    regions = []
    offset = 0
    for paragraph in (source_text or "").split("\n"):
        nonspace_index = []
        count = 0
        for ch in paragraph:
            nonspace_index.append(count)
            if ch != ' ':
                count += 1
        for start, length in _verb_tail_clusters(paragraph).items():
            begin = offset + nonspace_index[start]
            regions.append((begin, begin + length))
        offset += count
    if not regions:
        return 0
    breaks = 0
    cum = 0
    lines = [line for line in (wrapped_text or "").split('\n') if line.strip()]
    for line in lines[:-1]:
        cum += len(line.replace(' ', ''))
        breaks += any(begin < cum < end for begin, end in regions)
    return breaks


# 따옴표/괄호 쌍: 감싸인 스팬은 통째로 한 클러스터 (강제 개행에서 보호)
_BRACKET_PAIRS = {
    '「': '」', '『': '』', '【': '】', '〔': '〕', '〈': '〉', '《': '》',
    '(': ')', '(': ')', '[': ']', '[': ']', '{': '}',
    '"': '"', '“': '”', "'": "'", '‘': '’', '｢': '｣',
}
# 이보다 길면 묶지 않는다 (불가분 덩어리가 길수록 폰트가 그만큼 줄어들므로,
# 짧은 인용 명사만 보호하고 긴 인용문은 일반 규칙에 맡긴다)
_QUOTED_CLUSTER_MAX_CHARS = 8


@lru_cache(maxsize=4096)
def _cached_wrap_clusters(paragraph, verb_tails):
    """강제 개행 시에도 쪼개면 안 되는 최소 단위(클러스터)의 튜플로 분해합니다.

    - 활용 꼬리(수고/했어, 귀여워/졌네)는 한 덩어리 — 단어 중간 줄바꿈이 꼬리 앞에서 난다
      (verb_tails=False면 묶지 않는다)
    - 한국어 어미 단위(인가/이야/라고/십시오 등)+뒤따르는 기호는 한 덩어리
    - 기호 연속(~~!! 등)은 직전 클러스터에 붙는다 (기호가 줄에 걸쳐 갈라지지 않게)
    - 여는 괄호는 다음 클러스터에 붙는다 (행말 금칙)
    공백은 별도 토큰으로 남겨 어절 경계 정보를 보존한다.
    """
    # 피팅이 같은 문단을 폭·크기별로 수십 번 줄바꿈하므로 분해 결과를 재사용한다.
    n = len(paragraph)
    tails = _verb_tail_clusters(paragraph) if verb_tails else {}
    units = []
    i = 0
    while i < n:
        ch = paragraph[i]
        if ch == ' ':
            units.append(' ')
            i += 1
            continue
        if i in tails:
            units.append(paragraph[i:i + tails[i]])
            i += tails[i]
            continue
        matched = None
        if _is_hangul(ch):
            for unit in _KO_ENDING_UNITS:
                length = len(unit)
                if paragraph.startswith(unit, i) and (i + length >= n or not _is_hangul(paragraph[i + length])):
                    matched = unit
                    break
        if matched:
            units.append(matched)
            i += len(matched)
        else:
            units.append(ch)
            i += 1

    # 따옴표/괄호 쌍 묶기: 쌍으로 감싸인 짧은 스팬은 통째로 한 클러스터로
    # 만들어 줄바꿈이 인용구 내부를 가르지 않게 한다.
    paired = []
    i = 0
    while i < len(units):
        unit = units[i]
        closer = _BRACKET_PAIRS.get(unit) if len(unit) == 1 else None
        if closer:
            j = i + 1
            span_chars = 0
            found = -1
            while j < len(units):
                if units[j] == closer:
                    found = j
                    break
                span_chars += len(units[j].replace(' ', ''))
                if span_chars > _QUOTED_CLUSTER_MAX_CHARS:
                    break
                j += 1
            if found > 0:
                paired.append(''.join(units[i:found + 1]))
                i = found + 1
                continue
        paired.append(unit)
        i += 1
    units = paired

    clusters = []
    for unit in units:
        if unit != ' ' and all(c in _TRAILING_STICKY for c in unit) and clusters and clusters[-1] != ' ':
            clusters[-1] += unit  # 기호는 직전 글자에 글루
        elif clusters and clusters[-1] != ' ' and clusters[-1][-1] in LINE_TAIL_FORBIDDEN:
            clusters[-1] += unit  # 여는 괄호 뒤에 글루
        else:
            clusters.append(unit)
    return tuple(clusters)


def wrap_text_chars(text, font_path, font_size, style, max_width):
    """클러스터 단위 강제 줄바꿈 (금칙·기호 연속·어미 단위·활용 꼬리 보존).

    공백 단위 전략들은 긴 단어를 쪼개지 못해 그 단어가 폭에 들어갈 때까지
    폰트를 줄인다. 이 전략은 어디서든 줄을 바꾸는 탈출구 후보를 내되 쪼개면 안 되는
    단위는 클러스터로 묶는다. 단어 중간 줄바꿈은 마지막 수단이라 피팅 스코어러가
    한 곳마다 크게 깎고(text_fitting.MIDWORD_BREAK_PENALTY), 어절이 한 줄에 안 들어갈 때만 이긴다.
    """
    return _wrap_by_clusters(text, font_path, font_size, style, max_width, verb_tails=True)


def wrap_text_chars_loose(text, font_path, font_size, style, max_width):
    """활용 꼬리를 묶지 않는 클러스터 단위 강제 줄바꿈.

    꼬리를 묶으면 좁은 말풍선에서 줄이 늘어 글씨가 작아질 때가 있어 후보로 함께 둔다.
    꼬리 안에서 끊은 줄바꿈은 스코어러가 한 번 더 깎아, 크기 이득이 클 때만 이긴다.
    """
    return _wrap_by_clusters(text, font_path, font_size, style, max_width, verb_tails=False)


def _wrap_by_clusters(text, font_path, font_size, style, max_width, verb_tails):
    lines = []
    for paragraph in text.split('\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        cur = []  # 줄 누적을 클러스터 리스트로 유지 (문자열이면 클러스터
                  # 내부 공백을 lookback이 잘라버린다)
        for cluster in _cached_wrap_clusters(paragraph, verb_tails):
            if cluster == ' ' and not cur:
                continue
            candidate = ''.join(cur) + cluster
            if not cur or measure_line(candidate.rstrip(), font_path, font_size, style) <= max_width:
                cur.append(cluster)
                continue
            text_now = ''.join(cur)
            stripped = text_now.rstrip()
            # 줄 끝에 공백을 단 채 넘어왔으면 cluster는 새 단어의 시작 — carry와
            # cluster 사이의 공백을 복원해야 단어 사이가 붙지 않는다.
            had_trailing_space = len(text_now) > len(stripped)
            carry = []
            # 공백 lookback: 줄 끝 3자 이내에 '클러스터 경계' 공백이 있으면
            # 단어 경계에서 끊는다 (클러스터 내부 공백은 후보가 아님)
            k = len(cur) - 1
            tail_chars = 0
            while k >= 0 and cur[k] != ' ':
                tail_chars += len(cur[k].replace(' ', ''))
                if tail_chars > 3:
                    break
                k -= 1
            if 1 <= k < len(cur) - 1 and cur[k] == ' ' and 1 <= tail_chars <= 3:
                carry = cur[k + 1:]
                line = ''.join(cur[:k]).rstrip()
            elif cluster[0] in LINE_HEAD_FORBIDDEN and len(cur) >= 2:
                # 안전망: 글루를 빠져나온 행두 금지 문자는 직전 클러스터와 함께 내린다
                carry = [cur[-1]]
                line = ''.join(cur[:-1]).rstrip()
            else:
                line = stripped
            if line:
                lines.append(line)
            sep = [' '] if (carry and had_trailing_space and cluster != ' ') else []
            cur = carry + sep + ([] if cluster == ' ' else [cluster])
        tail = ''.join(cur).strip()
        if tail:
            lines.append(tail)
    return "\n".join(lines)


def aggressive_wrap(text, font_path, font_size, style, max_width):
    """Ignore paragraph breaks and fill lines greedily, breaking only at spaces.

    Always one of the candidates (build_wrap_candidates); the fitting scorer compares it with the others.
    """
    raw_lines = text.replace('\n', ' ').split(' ')
    lines = []
    current = ""
    for word in raw_lines:
        if not word:
            continue
        if not current:
            current = word
        elif measure_line(current + " " + word, font_path, font_size, style) <= max_width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


# 뒷말을 꾸미는 한 글자 관형사 — 줄 끝에 홀로 남으면 뜻이 끊겨 읽힌다('두 / 분은'·'한 / 명'·'그 / 애')
DETERMINERS = ("한", "두", "세", "네", "몇", "첫", "이", "그", "저", "새", "헌", "각", "온", "딴", "옛")
_DETERMINER_LEAD = "…⋯.,!?~「『（(\"' "


def is_determiner(word):
    return word.lstrip(_DETERMINER_LEAD) in DETERMINERS


def wrap_keep_determiners(text, font_path, font_size, style, max_width):
    """띄어쓰기에서만 끊되, 줄 끝에 남을 한 글자 관형사는 다음 줄로 넘긴다 ('다음은 / 네 차례다')."""
    lines = []
    for paragraph in text.split("\n"):
        current = []
        for word in (w for w in paragraph.split(" ") if w):
            if current and measure_line(" ".join(current + [word]), font_path, font_size, style) > max_width:
                carry = [current.pop()] if len(current) > 1 and is_determiner(current[-1]) else []
                lines.append(" ".join(current))
                current = carry
            current.append(word)
        if current:
            lines.append(" ".join(current))
    return "\n".join(lines)


# 문장 부호로 끝난 어절 — 그 뒤에서 줄을 바꾼다
PHRASE_END_MARKS = ",.!?…⋯~〜—！？"


def wrap_text_phrases(text, font_path, font_size, style, max_width):
    """문장 부호로 끝난 어절마다 줄을 새로 시작하고, 그 사이는 띄어쓰기에서 채워 끊는다 ('아니, / 같이 안 갈 / 건데?')."""
    phrases = []
    for paragraph in text.split("\n"):
        current = []
        for word in paragraph.split():
            current.append(word)
            if word[-1] in PHRASE_END_MARKS:
                phrases.append(" ".join(current))
                current = []
        if current:
            phrases.append(" ".join(current))
    return "\n".join(wrap_text_korean(phrase, font_path, font_size, style, max_width) for phrase in phrases)


def build_wrap_candidates(text, bubble_ratio):
    """Pick an ordered list of wrap strategies for this text + bubble shape."""
    candidates = [
        wrap_text_korean,
        wrap_text,
        aggressive_wrap,
        wrap_keep_determiners,
        # 마지막 순서 = 동점이면 단어 경계 줄바꿈 우선 (스코어러가 strictly-greater만 교체)
        wrap_text_chars,
        wrap_text_chars_loose,
    ]
    if any(word[-1] in PHRASE_END_MARKS for word in text.split()[:-1]):
        candidates.insert(1, wrap_text_phrases)

    density = text_density(text)
    if " " in text and bubble_ratio <= 1.0 and density >= 8:
        line_targets = [2]
        if density >= 14 or bubble_ratio <= 0.75:
            line_targets.append(3)
        if density >= 24 and bubble_ratio <= 0.58:
            line_targets.append(4)
        candidates = [_make_balanced_wrap(target_lines) for target_lines in line_targets] + candidates

    return candidates
