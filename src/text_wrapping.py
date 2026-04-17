"""Text wrapping strategies used by the layout and fitting modules.

Each wrap function has the signature:
    wrap(text, font_path, font_size, style, max_width) -> str

build_wrap_candidates() picks the right combination of strategies based on
text density and bubble aspect ratio.
"""
import re

from src.text_renderer import measure_line


LINE_HEAD_FORBIDDEN = set(")]}ã€‰ã€‹ã€ã€ã€‘ã€ã€‚ï¼Œï¼ï¼Ÿâ€¦â‹¯:;")
LINE_TAIL_FORBIDDEN = set("([<{ã€ˆã€Šã€Œã€Žã€")

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
        words = re.findall(r'(Â·+|[!?]+|\S+)', paragraph)
        if not words:
            continue
        current_line = words[0]
        for word in words[1:]:
            joiner = "" if re.match(r'^(Â·+|[!?â‹¯]+)$', word) else " "
            if measure_line(current_line + joiner + word, font_path, font_size, style) <= max_width:
                current_line += joiner + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
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
