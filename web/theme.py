"""Manga-Bogopa 시각 테마.

디자인 컨셉 — "만화 원고 위에서 일하는 도구":
- UI 섹션은 만화 원고의 컷(panel): 먹선 테두리 + 인쇄 오프셋 그림자
- 종이(#F7F6F2)와 먹(#17171B)이 기본, 액센트는 만화 제작 현장에서 밑그림에 쓰는
  논포토 블루(non-photo blue) 계열 — 주제에서 끌어온 색
- 제호(마스트헤드)는 만화 단행본 로고타입처럼: Black Han Sans + 망점(스크린톤) 스트립
- 시그니처는 마스트헤드 하나로 제한하고 나머지는 절제 (frontend-design 스킬 지침)
"""
from gradio.themes import Base
from gradio.themes.utils import colors, fonts, sizes


class _SafeFont(fonts.Font):
    """Gradio 6.11의 내장 테마 비교(__eq__에 str이 들어오는 버그)를 견디는 Font."""

    def __eq__(self, other):
        try:
            return super().__eq__(other)
        except AttributeError:
            return False

    __hash__ = fonts.Font.__hash__


class _SafeGoogleFont(fonts.GoogleFont):
    def __eq__(self, other):
        try:
            return super().__eq__(other)
        except AttributeError:
            return False

    __hash__ = fonts.GoogleFont.__hash__

# 논포토 블루 (만화 원고 밑그림 청색) 스케일
_NONPHOTO_BLUE = colors.Color(
    name="nonphoto_blue",
    c50="#F0F9FD", c100="#DDF0F9", c200="#BFE3F3", c300="#93CFE9",
    c400="#5BB4DC", c500="#2D96C8", c600="#1E7FB0", c700="#1A6890",
    c800="#1B5573", c900="#1C475F", c950="#122E3E",
)

# 먹/스크린톤 그레이 (살짝 차가운 무채색)
_INK_GRAY = colors.Color(
    name="ink_gray",
    c50="#F7F6F2", c100="#EFEEE9", c200="#E3E1DA", c300="#CFCDC6",
    c400="#A3A2A6", c500="#75757D", c600="#55555C", c700="#3A3A40",
    c800="#27272C", c900="#17171B", c950="#0E0E11",
)


def build_manga_theme() -> Base:
    theme = Base(
        primary_hue=_NONPHOTO_BLUE,
        secondary_hue=_NONPHOTO_BLUE,
        neutral_hue=_INK_GRAY,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=[
            _SafeGoogleFont("IBM Plex Sans KR"),
            _SafeFont("Pretendard"),
            _SafeFont("Malgun Gothic"),
            _SafeFont("system-ui"),
            _SafeFont("sans-serif"),
        ],
        font_mono=[
            _SafeGoogleFont("IBM Plex Mono"),
            _SafeFont("Consolas"),
            _SafeFont("monospace"),
        ],
    )
    theme.set(
        # 종이
        body_background_fill="#F7F6F2",
        body_background_fill_dark="#101013",
        body_text_color="#17171B",
        body_text_color_dark="#EFEEE9",
        # 컷(panel) — 흰 원고지 + 먹선
        block_background_fill="#FFFFFF",
        block_background_fill_dark="#16161A",
        block_border_color="#17171B",
        block_border_color_dark="#3A3A40",
        block_border_width="1px",
        block_shadow="none",
        block_label_text_color="#55555C",
        block_label_text_color_dark="#A3A2A6",
        block_title_text_color="#17171B",
        block_title_text_color_dark="#EFEEE9",
        block_title_text_weight="700",
        # 입력
        input_background_fill="#FFFFFF",
        input_background_fill_dark="#1C1C21",
        input_border_width="1px",
        input_border_color="#CFCDC6",
        input_border_color_dark="#3A3A40",
        input_border_color_focus="#2D96C8",
        input_border_color_focus_dark="#5BB4DC",
        # 주 버튼 = 먹 (종이 글자), 호버에 논포토 블루
        button_primary_background_fill="#17171B",
        button_primary_background_fill_hover="#1E7FB0",
        button_primary_text_color="#F7F6F2",
        button_primary_background_fill_dark="#EFEEE9",
        button_primary_background_fill_hover_dark="#5BB4DC",
        button_primary_text_color_dark="#17171B",
        # 보조 버튼 = 종이 위 먹선
        button_secondary_background_fill="#FFFFFF",
        button_secondary_background_fill_hover="#DDF0F9",
        button_secondary_text_color="#17171B",
        button_secondary_background_fill_dark="#27272C",
        button_secondary_background_fill_hover_dark="#1C475F",
        button_secondary_text_color_dark="#EFEEE9",
        # 포인트 컨트롤
        slider_color="#2D96C8",
        slider_color_dark="#5BB4DC",
        checkbox_background_color_selected="#2D96C8",
        checkbox_background_color_selected_dark="#2D96C8",
        loader_color="#2D96C8",
        link_text_color="#1E7FB0",
        link_text_color_dark="#5BB4DC",
    )
    return theme


# 마스트헤드: 만화 단행본 제호 + 망점(스크린톤) 스트립 (이 페이지의 시그니처)
HEADER_HTML = """
<div id="masthead">
  <div class="masthead-inner">
    <div class="masthead-eyebrow">일본 만화 → 한국어 자동 번역·식자</div>
    <div class="masthead-title">만화보고파</div>
    <div class="masthead-sub">MANGA-BOGOPA</div>
  </div>
  <div class="halftone-strip" aria-hidden="true"></div>
</div>
"""

# 판권장(colophon) 풍 푸터 — 조용한 마무리
FOOTER_HTML = """
<div id="colophon">
  탐지 · 인식 · 번역 · 지우기 · 식자 — 다섯 공정이 이 화면 안에서 끝납니다.
</div>
"""

MANGA_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Black+Han+Sans&display=swap');

:root {
  --mb-paper: #F7F6F2;
  --mb-ink: #17171B;
  --mb-tone: #75757D;
  --mb-line: #E3E1DA;
  --mb-blue: #2D96C8;
  --mb-blue-deep: #1E7FB0;
  --mb-offset: #E3E1DA;
}
.dark {
  --mb-paper: #101013;
  --mb-ink: #EFEEE9;
  --mb-tone: #A3A2A6;
  --mb-line: #3A3A40;
  --mb-offset: #000000;
}

/* ── 마스트헤드 ─────────────────────────────────────────── */
#masthead {
  border: 2px solid var(--mb-ink);
  background: var(--background-fill-primary, #FFFFFF);
  margin-bottom: 18px;
  box-shadow: 4px 4px 0 var(--mb-offset);
}
#masthead .masthead-inner { padding: 18px 22px 10px 22px; }
#masthead .masthead-eyebrow {
  font-size: 12px;
  letter-spacing: 0.18em;
  color: var(--mb-blue-deep);
  font-weight: 600;
  margin-bottom: 2px;
}
#masthead .masthead-title {
  font-family: 'Black Han Sans', var(--font), sans-serif;
  font-size: 44px;
  line-height: 1.05;
  color: var(--mb-ink);
  letter-spacing: 0.01em;
}
#masthead .masthead-sub {
  font-size: 11px;
  letter-spacing: 0.42em;
  color: var(--mb-tone);
  margin-top: 2px;
}
/* 스크린톤(망점) 스트립 — 왼쪽은 촘촘, 오른쪽으로 갈수록 성김 */
#masthead .halftone-strip {
  height: 14px;
  border-top: 2px solid var(--mb-ink);
  background-image: radial-gradient(circle, var(--mb-ink) 1.1px, transparent 1.2px);
  background-size: 6px 6px;
  -webkit-mask-image: linear-gradient(90deg, #000 0%, rgba(0,0,0,.55) 55%, rgba(0,0,0,.12) 100%);
  mask-image: linear-gradient(90deg, #000 0%, rgba(0,0,0,.55) 55%, rgba(0,0,0,.12) 100%);
}

/* ── 컷(panel) 카드 ─────────────────────────────────────── */
.panel {
  border: 1.5px solid var(--mb-ink) !important;
  border-radius: 2px !important;
  box-shadow: 3px 3px 0 var(--mb-offset) !important;
  background: var(--block-background-fill) !important;
  padding: 14px !important;
}
/* Group이 자식 사이 경계선을 만들려고 border 색을 배경으로 깔아 두는데,
   컨테이너 없는 HTML/Markdown 뒤로 그게 비쳐 보인다. 패널 안에서는 흰 배경으로 통일. */
.panel .styler { background: var(--block-background-fill) !important; }
.panel-label-wrap { background: transparent !important; }

/* 컷 머리표 — 괘선 + 청색 틱 */
.panel-label {
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 0.14em;
  color: var(--mb-ink);
  border-left: 3px solid var(--mb-blue);
  padding-left: 8px;
  margin: 2px 0 10px 0;
}

/* ── 탭 ─────────────────────────────────────────────────── */
[role="tablist"] {
  border-bottom: 2px solid var(--mb-ink) !important;
  background: transparent !important;
}
button[role="tab"] {
  font-weight: 700 !important;
  letter-spacing: 0.08em;
  border-radius: 0 !important;
  background: transparent !important;
}
button[role="tab"][aria-selected="true"] {
  color: var(--mb-blue-deep) !important;
  border-bottom: 3px solid var(--mb-blue) !important;
}

/* ── 보조 버튼: 종이 위 먹선 ─────────────────────────────── */
button.secondary {
  border: 1px solid var(--mb-ink) !important;
  border-radius: 2px !important;
}

/* ── 컨테이너 없는 입력(폴더 경로 등)에도 테두리 ───────────── */
.panel input[type="text"]:not([class*="svelte-checkbox"]),
.panel textarea {
  border: 1px solid var(--mb-line) !important;
  border-radius: 2px !important;
  background: var(--block-background-fill) !important;
}
.panel input[type="text"]:focus, .panel textarea:focus {
  border-color: var(--mb-blue) !important;
}

/* ── 아코디언: 조용한 괘선 목록 ──────────────────────────── */
.accordion {
  border: 1px solid var(--mb-line) !important;
  border-radius: 2px !important;
  box-shadow: none !important;
}
.accordion > button { font-weight: 600 !important; }
/* 패널 카드 안에서는 더 조용하게 */
.panel .accordion { border: 1px solid var(--mb-line) !important; }

/* 파일 다운로드 박스는 컴팩트하게 */
.panel [data-testid="file"] { min-height: 0 !important; }

/* ── 실행 버튼: 컷 테두리 + 오프셋 (마스트헤드와 호응) ────── */
#run-button {
  border: 2px solid var(--mb-ink) !important;
  border-radius: 2px !important;
  box-shadow: 3px 3px 0 var(--mb-offset) !important;
  font-weight: 800 !important;
  font-size: 16px !important;
  letter-spacing: 0.1em;
  transition: transform .06s ease, box-shadow .06s ease;
}
#run-button:active {
  transform: translate(2px, 2px);
  box-shadow: 1px 1px 0 var(--mb-offset) !important;
}

/* ── 로그: 식자 작업대의 모노스페이스 전표 ────────────────── */
#log-box textarea {
  font-family: var(--font-mono) !important;
  font-size: 12.5px !important;
  line-height: 1.55 !important;
  background: var(--mb-paper) !important;
  border: 1px solid var(--mb-line) !important;
}

/* ── 갤러리 ──────────────────────────────────────────────── */
#result-gallery .grid-wrap { background: var(--mb-paper); }
#result-gallery .thumbnail-item {
  border: 1px solid var(--mb-ink) !important;
  border-radius: 0 !important;
}

/* ── 판권장 푸터 ─────────────────────────────────────────── */
#colophon {
  margin-top: 22px;
  padding-top: 10px;
  border-top: 1px solid var(--mb-line);
  color: var(--mb-tone);
  font-size: 11.5px;
  letter-spacing: 0.08em;
  text-align: center;
}

/* 모션 절제 */
@media (prefers-reduced-motion: reduce) {
  #run-button { transition: none; }
}
"""
