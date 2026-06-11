"""Text wrapping strategies used by the layout and fitting modules.

Each wrap function has the signature:
    wrap(text, font_path, font_size, style, max_width) -> str

build_wrap_candidates() picks the right combination of strategies based on
text density and bubble aspect ratio.
"""
import re

# 한국어 조판 금칙은 세로쓰기(단 나눔)에서도 쓰이므로 렌더러에 정의되어 있다.
from src.text_renderer import LINE_HEAD_FORBIDDEN, LINE_TAIL_FORBIDDEN, measure_line

# Shared tall-bubble thresholds (used by wrapping + fitting + style resolvers).
TALL_BUBBLE_RATIO = 1.8
TALL_BUBBLE_MIN_CHARS = 8


def text_density(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


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
        penalty += 8.0
    if stripped[-1] in LINE_TAIL_FORBIDDEN:
        penalty += 8.0
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
_TRAILING_STICKY = frozenset(")]}〉》」』】〕’”.。,，、!！?？…⋯‥·~〜‼⁉♪♬♡♥☆★―—–:;")
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
    "니다", "시오",
}, key=len, reverse=True))


def _is_hangul(ch):
    return '가' <= ch <= '힣'


# 따옴표/괄호 쌍: 감싸인 스팬은 통째로 한 클러스터 (강제 개행에서 보호)
_BRACKET_PAIRS = {
    '「': '」', '『': '』', '【': '】', '〔': '〕', '〈': '〉', '《': '》',
    '(': ')', '(': ')', '[': ']', '[': ']', '{': '}',
    '"': '"', '“': '”', "'": "'", '‘': '’', '｢': '｣',
}
# 이보다 길면 묶지 않는다 (불가분 덩어리가 길수록 폰트가 그만큼 줄어들므로,
# 짧은 인용 명사만 보호하고 긴 인용문은 일반 규칙에 맡긴다)
_QUOTED_CLUSTER_MAX_CHARS = 8


def _split_wrap_clusters(paragraph):
    """강제 개행 시에도 쪼개면 안 되는 최소 단위(클러스터)로 분해합니다.

    - 한국어 어미 단위(인가/이야/라고/십시오 등)+뒤따르는 기호는 한 덩어리
    - 기호 연속(~~!! 등)은 직전 클러스터에 붙는다 (기호가 줄에 걸쳐 갈라지지 않게)
    - 여는 괄호는 다음 클러스터에 붙는다 (행말 금칙)
    공백은 별도 토큰으로 남겨 어절 경계 정보를 보존한다.
    """
    n = len(paragraph)
    units = []
    i = 0
    while i < n:
        ch = paragraph[i]
        if ch == ' ':
            units.append(' ')
            i += 1
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
    return clusters


def wrap_text_chars(text, font_path, font_size, style, max_width):
    """클러스터 단위 강제 줄바꿈 (금칙·기호 연속·어미 단위 보존).

    공백 단위 전략들은 긴 단어를 쪼개지 못해 그 단어가 폭에 들어갈 때까지
    폰트를 줄인다. 한국어 만화 식자는 단어 중간 줄바꿈이 표준 관행이므로, 어디서든
    줄을 바꾸는 탈출구 후보를 제공하되 쪼개면 안 되는 단위는 클러스터로 묶는다.
    미관 비교는 피팅 스코어러가 담당한다.
    """
    lines = []
    for paragraph in text.split('\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        cur = []  # 줄 누적을 클러스터 리스트로 유지 (문자열이면 클러스터
                  # 내부 공백을 lookback이 잘라버린다)
        for cluster in _split_wrap_clusters(paragraph):
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
    """Ignore paragraphs; break on every space when other strategies fail."""
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


def build_wrap_candidates(text, bubble_ratio):
    """Pick an ordered list of wrap strategies for this text + bubble shape."""
    candidates = [
        wrap_text_korean,
        wrap_text,
        aggressive_wrap,
        # 마지막 순서 = 동점이면 단어 경계 줄바꿈 우선 (스코어러가 strictly-greater만 교체)
        wrap_text_chars,
    ]

    density = text_density(text)
    if " " in text and bubble_ratio <= 1.0 and density >= 8:
        line_targets = [2]
        if density >= 14 or bubble_ratio <= 0.75:
            line_targets.append(3)
        if density >= 24 and bubble_ratio <= 0.58:
            line_targets.append(4)
        candidates = [_make_balanced_wrap(target_lines) for target_lines in line_targets] + candidates

    return candidates
