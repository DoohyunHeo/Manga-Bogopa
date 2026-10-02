# Text Rendering Spec

## Goal

This project does not aim to mimic Japanese vertical typesetting literally.
The goal is to make Korean dialogue feel close to community-accepted manga scanlation quality:

- readable on mobile first
- visually centered and balanced inside speech bubbles
- slightly tall and compact in narrow bubbles
- never amateur-looking due to blurry raster text or overly thin strokes

This spec intentionally excludes bubble expansion or redraw-driven layout changes.
The current focus is text rendering quality and text block shaping only.

## Community Rules Adopted

Sources reviewed from Korean communities consistently converged on these rules:

- Long Korean text should stay horizontal whenever possible.
- Very narrow bubbles should not force unreadable vertical Korean.
- Dialogue text should remain centered and visually balanced.
- Tiny bubbles need thicker-looking strokes and clearer rendering.
- Over-compressed spacing looks amateur; only small tracking adjustments are acceptable.
- Readers now consume most pages on phones, so minimum readable size matters.
- Line breaks should occur on whitespace boundaries; splitting a word is a penalized last resort,
  used only when no whole-word layout fits (see Mid-Word Line Breaks).

## Rendering Principles

### 1. Dialogue First, Effect Second

Regular dialogue is optimized for readability before style.
Aggressive styling is reserved for shouting, impact text, and freeform text.

### 2. Tall Impression Without Full Vertical Typesetting

For tall speech bubbles, Korean text should create a vertically elongated silhouette by:

- slightly reducing horizontal scale
- using slightly tighter letter spacing
- allowing more lines
- slightly increasing line spacing

This preserves Korean readability while matching the visual rhythm of Japanese bubble shapes.

### 3. Small Bubble Protection

When a bubble is small, the renderer should prefer clarity over density:

- reduce spacing compression
- avoid over-condensing width
- increase render oversampling
- allow faux bold or heavier visual weight

### 4. Black-First Dialogue

Default dialogue fill should be black or near-black.
Gray dialogue text is avoided for the main bubble text because it reduces legibility and often looks weak in scanlation output.

### 5. Oversampled Vector Rendering

Text must be rendered at higher resolution and then downsampled.
Small-font render quality is a bigger priority than marginal speed savings.

## Bubble Categories

### Standard Bubble

- default horizontal Korean dialogue
- centered alignment
- moderate condensed scale
- moderate line spacing

### Tall Bubble

Conditions:

- bubble height / width >= `TALL_BUBBLE_RATIO`
- visible text density >= `TALL_BUBBLE_MIN_CHARS`

Policy:

- preserve horizontal Korean text
- condense slightly
- increase line density vertically
- build a taller text silhouette

### Small Bubble

Conditions:

- min target dimension <= 70px, or
- measured source font size <= `MIN_READABLE_TEXT_SIZE`

Policy:

- relax tracking compression
- widen horizontal scale slightly if needed for clarity
- embolden
- increase oversample scale

### Freeform Text

- keep white stroke by default
- raise oversample for small text
- avoid over-condensing more than bubble text

## Current Render Profiles

### Standard Dialogue

- black fill
- mild negative tracking
- condensed horizontal scale
- moderate line spacing

### Shouting / Angry / Pop

- black fill
- stronger visual weight
- more aggressive oversampling
- tighter line spacing

### Narration / Cute / Handwriting

- kept darker than before
- lighter compression than dialogue

### Freeform

- black fill with white stroke
- oversampled rendering

## Quality Constraints

The renderer should avoid the following:

- blurry scaled-up bitmap text
- excessively thin strokes inside tiny bubbles
- severe tracking compression
- visibly inconsistent font sizes between nearby bubbles
- gray weak-looking dialogue text
- forced one-character-per-line Korean unless explicitly intended
- mid-word line breaks, except as a penalized last resort (see Mid-Word Line Breaks)

## Panel-Border Attachment

Text that visually leans against a panel border should be aligned toward that
border, not centered, because the border is a strong vertical anchor and
centering creates a disconnected gap.

Detection (`src/line_detector.py`):

- morphology-based vertical-line detection (OTSU binarize + vertical opening)
- bubble: scan `BUBBLE_ATTACHMENT_EDGE_RATIO` of the bubble width on each side
- freeform: scan `FREEFORM_ATTACHMENT_SEARCH_PX` outside each side of the text
  box; at page edges, scan as much context as is available
- both require the detected vertical structure to span at least
  `*_MIN_LENGTH_RATIO` of the region height to count as attached
- when both sides trigger, the stronger side wins; ties fall back to centered

Layout response:

- bubble attachment: anchor text at the attached edge, respecting
  `BUBBLE_EDGE_SAFE_MARGIN` and pulling by `ATTACHED_BUBBLE_TEXT_MARGIN`
- freeform attachment: anchor text at the original detection box edge, moved
  inward when a panel line is closer than the widest outline the text can get
  plus 1 px (`text_layout._attached_margin`), preserved through overlap adjustment
- vertical layout ignores attachment (it always centers)

## Vertical Fallback Policy

Speech bubbles are horizontal by default. `page_drawer` builds two final plans
per bubble (fitted, then grown when the text leaves the bubble mostly empty) —
horizontal, and one-column vertical fitted to the bubble target area or the
original text-box shape, whichever allows the larger size — and uses the
vertical one only when `text_layout.bubble_vertical_rescue` approves:

1. the horizontal size is below 0.6 × the vertical size, or
2. the horizontal size is below `max(MIN_READABLE_TEXT_SIZE, 0.65 × measured
   source size)` and the vertical size is at least 1.1 × larger, or
3. a short unspaced word (6 letters or fewer) cannot fit one horizontal line
   and must be split, and the vertical size is not smaller.

Text outside bubbles enters vertical layout via `text_layout._fit_text`:

1. **Extreme aspect override** — if the text box aspect is at least
   `VERTICAL_FORCE_ASPECT_RATIO` (default 6:1), vertical is forced regardless
   of the measured source size or whitespace.
2. **Tall column box** — for a box at least 2.5 × taller than wide whose
   horizontal layout wraps into several lines or spreads past 1.2 × the box
   width, one column is used when it fits at 0.9 × the horizontal size or more
   (0.6 × when the horizontal layout spreads) and at `MIN_READABLE_TEXT_SIZE`
   or more.
3. **Shrink-based fallback** — for a box at least 4 × taller than wide, if
   horizontal fitting ends up below `FONT_SHRINK_THRESHOLD_RATIO × measured
   source size`, the horizontal result is discarded and vertical is fitted
   from the measured source size.

## Vertical Layout: Columns

- **Bubble text** is always a single column: it goes vertical only when the
  whole translation fits in one column (and passes the rescue test above).
  If it cannot, the bubble stays horizontal — never multi-column, never an
  overflowing column.
- **Text outside bubbles**: vertical stays vertical (narrow SFX boxes make
  horizontal nonsensical) and shrinks within the tolerance caps if needed. It
  renders as one column, except long vertical captions: when the box is at
  least 4 × taller than wide and the one-column size is below 0.6 × the page
  dialogue size, `page_drawer._multi_column_caption` tries up to 8 columns,
  right to left, inside the original box width, and uses them when the text
  gets at least 1.2 × larger (never above the measured source size).
- spaces inside vertical text become a 0.35 em vertical gap
- fitting and rendering share the same column-height budget
  (`TextRenderPlan.vertical_column_height`, plus `vertical_columns` for the
  column count) so the measured block matches what is drawn

## Freeform Horizontal Overflow

Detected freeform boxes hug the original Japanese text tightly, which
starves longer Korean translations. Horizontal freeform fitting therefore
allows the text block of a box at least as wide as tall to exceed the detected
box width by `FREEFORM_BOX_OVERFLOW_RATIO` (default 0.2 = 20%), spreading
evenly around the box center (wide boxes stay within 1.1 × the box width and
keep the box height; widening stops at panel borders and the page edge).
Vertical freeform keeps the strict box width.

## Mid-Word Line Breaks (단어 중간 줄바꿈)

단어 중간 줄바꿈은 글자 단위 전략(`text_wrapping.wrap_text_chars`·`wrap_text_chars_loose`)에서만
나오는 마지막 수단이다. 피팅 점수에서 한 곳당 -40점(글씨 10px, `text_fitting.MIDWORD_BREAK_PENALTY`)을
깎고, 그래도 끊긴 배치가 이기면 원문 크기의 0.65배까지 줄여 어절 경계만 쓰는 배치를 찾는다
(`text_fitting._whole_word_fallback`). 활용 꼬리 앞에서 끊은 줄바꿈('수고/했어')은 단어 중간으로 세지
않고 한 곳당 -12점(글씨 3px)만 깎는다. 끊는 자리는 다음 덩어리 규칙으로 정한다:

- 쪼개지 않는 덩어리: 기호 연속(`~~!!`, `-!`), 짧은 괄호 인용, 어미 단위
  (습니다·네요·는데요 등)
- 활용 꼬리: 체언 + '하다·되다·있다·없다'(수고/했어, 안녕/하세요, 재미/있어,
  습격/당하면)와 '-아/어' 뒤 보조용언(귀여워/졌네, 구해/주셨다, 지워/버렸어,
  해/봤어). 동사가 시작되는 자리부터 어절 끝까지를 한 덩어리로 묶어 꼬리 앞에서
  끊기게 한다. 5자 이상 꼬리는 끝의 어미 단위 앞까지만 묶는다(대답/하겠/습니다).
  감탄사(아하하하), 어미 자체('하지만'의 '지만'), 조사 '까지'의 '지'는 제외한다.
- 꼬리를 묶지 않는 전략(`wrap_text_chars_loose`)도 후보로 둔다 — 꼬리를 묶으면 좁은
  말풍선에서 줄이 늘어 글씨가 작아질 때가 있다. 꼬리 안(어미 앞)에서 끊은 줄바꿈('수고했/어!')은
  단어 중간 끊김 감점에 한 곳당 -6점을 더 깎는다.

실제 176쪽 측정: 어미 앞 줄바꿈 51곳 → 7곳(남은 곳은 자연스럽게 끊으면 글씨가
2px 이상 작아지는 경우), 전체 단어 중간 줄바꿈 수는 거의 같음(309 → 307),
글씨가 작아진 곳 2곳(붙임표 '-'를 앞 글자에 붙여 '-!'가 줄 머리에 오지 않게 한 결과).

## Glyph Fallback

Characters missing from the assigned font (♪ ★ ♡ ㊙ etc.) are rendered
through a per-character fallback chain: assigned font → `FONT_MAP` fonts
(standard/narration first) → system symbol fonts (Malgun Gothic, Segoe UI
Symbol, ...). Measurement and drawing both honor the fallback, so symbol
runs do not disappear or render as tofu. `replace_unsupported_chars`
only rewrites a character when *no* candidate font can draw it, and only on a
working copy of the element (`page_drawer._prepare_element`), so the saved
translation never changes.

## Stroke Two-Pass

Outlined text (freeform style) paints all line strokes first, then all
fills. With tight line spacing this prevents the next line's stroke from
carving into the previous line's fill.

## Freeform Style Consolidation

For free-text items (narration, SFX, ambient text) the "standard" style does
not apply — those items are treated as narration:

- first stage (`font_analysis.measure_font_properties`): free-text starts as
  `narration`, bubble text as `standard`
- book-level pass (`font_relative.apply_book_styles`): plain free-text that the
  shape model calls `standard` is set in the narration font, except white text
  on a dark background (screens, chat) and when `NARRATION_AS_STANDARD` is on

## Source Size Tolerance Cap

Font fitting is bounded by the measured source size (`element.font_size`: the
ink-measured size from `glyph_metrics`, adjusted by the book-level pass in
`font_relative`):

- upper cap: `source × MODEL_FONT_SIZE_CEILING_RATIO` (hard cap on search)
- lower bound: `source × MODEL_FONT_SIZE_FLOOR_RATIO` (soft penalty, not a
  hard floor, so translations that genuinely need smaller text still fit)
- both ratios are fixed code values (0.8 and 1.2)
- the per-wrap size search uses bisection over the monotone "fits" predicate
  (largest fitting size), so candidate evaluation costs O(log range) instead
  of a 1px-step linear scan

## Implemented Now

The current implementation includes:

- Skia font rendering with subpixel AA and font-level horizontal scaling
- oversampled transparent text-layer rendering followed by downsampling
- transparent text compositing instead of full-image redraw per bubble
- two-pass stroke/fill painting for outlined multi-line text
- per-character glyph fallback across FONT_MAP + system symbol fonts
- darker dialogue defaults with thicker-looking small-text rendering
- small-bubble readability overrides
- tall-bubble condensed silhouette overrides
- Korean-aware wrapping before fallback wrapping strategies
- whitespace-only balanced wrapping for wide bubbles
- split penalties for forbidden line heads and forbidden line tails
  (sets shared by horizontal wrapping and vertical column breaking)
- candidate scoring based on font size, overflow, fill ratio, orphan lines, and tall-bubble silhouette balance
- source-size tolerance caps on both horizontal and vertical fit paths
- bisection-based size search inside each wrap candidate
- single-column vertical layout for bubbles (they stay horizontal when one
  column cannot fit), several columns only for long vertical captions outside
  bubbles, with column-head 금칙 handling
- freeform horizontal overflow allowance (`FREEFORM_BOX_OVERFLOW_RATIO`)
- panel-border attachment detection and aligned rendering for bubble + freeform text
- entry to vertical layout: the rescue test for bubbles; extreme aspect override,
  tall column box and shrink fallback for text outside bubbles
- free-text items judged `standard` are set as narration with no confidence rule
  (exceptions in Freeform Style Consolidation)

## Module Layout

- `src/text_renderer.py` — skia-backed drawing, `measure_*` primitives,
  glyph fallback, vertical column layout, 금칙 character sets
- `src/text_wrapping.py` — wrap strategies (Korean-aware, balanced, aggressive)
- `src/text_fitting.py` — horizontal + vertical font-size search with scoring
- `src/text_layout.py` — style resolution, vertical decisions, plan assembly
- `src/line_detector.py` — panel-border detection for attachment alignment
- `src/page_drawer.py` — page-level draw orchestration (plan -> consistency -> QA -> composite)
- `src/typeset_qa.py` — 실제 글자 레이어 알파 기반 검수·제한 보정·진단 리포트
- `src/page_consistency.py` — 페이지 단위 대사 글씨 크기 통일 계획

## Render QA and Bounded Repair (식자 검수·제한 보정)

모듈: `src/typeset_qa.py`, 진입: `page_drawer.render_page_text()`
(`draw_text_on_image()`는 이미지만 돌려주는 얇은 껍데기).

검사 대상은 합성 직전의 **실제 글자 레이어**(외곽선·회전·세로쓰기 반영)의
알파 채널이다. 레이어의 투명 여백은 잉크로 세지 않는다 (알파 24 미만은
리샘플 번짐으로 보고 제외). 확정 위반 세 가지를 픽셀 수로 센다:

- `clipped_px` — 페이지 밖으로 잘리는 잉크
- `outside_px` — 말풍선 **안전 영역** 밖의 잉크 (말풍선 텍스트만)
- `collision_px` — 먼저 확정된 다른 글자 잉크와 실제로 겹치는 픽셀 (마스크 AND)

안전 영역은 두 가지를 함께 본다:

- 사각 안전 영역 — 탐지된 말풍선 사각형을 `min(BUBBLE_EDGE_SAFE_MARGIN,
  피팅 여백)`만큼 안으로 들인 사각형(붙은 쪽 배치 포함). 말풍선 모양 따라 맞추지 않은
  글자는 글자 크기의 0.5배(`page_drawer._TEXT_MARGIN_RATIO`)까지 더 좁힌다.
- 말풍선 안쪽 흰 영역 — 지운 쪽에서 글자 자리를 품은 흰 덩어리(구멍은 메움)의 테두리까지
  거리 지도(`page_drawer._bubble_interior`). 테두리에서 글자 크기의 0.2배
  (`page_drawer._OUTLINE_MARGIN_RATIO`)보다 가까운 잉크도 벗어난 것으로 센다 — 가시·곡선
  테두리와 말풍선 안으로 들어온 그림까지 지킨다. 흰 덩어리가 말풍선 상자 넓이의 30%보다
  작으면(검은·톤 말풍선) 사각 영역만 본다.

보정 순서 (요소별·결정적):

1. 배치 이동 — 재렌더 없이 오프셋만 바꿈. 경계 안으로 밀어넣기, 충돌 상대
   (최대 2개) 기준 상/하/좌/우 회피. 붙은 쪽(attachment) 정렬은 안쪽
   방향으로만 움직인다. 이동 상한 = 자기 잉크 크기.
2. 줄바꿈 대안 — 같은 크기에서 기존 줄바꿈 전략 x 폭 비율 조합 중 최대 6개.
   `text_wrapping.is_layout_equivalent`를 통과한 것만 쓴다 (어절 공백
   삭제·삽입, 단락 줄바꿈 삭제, 글자 변경 금지). 연속 공백·탭이 있는 번역은
   줄바꿈 대안을 만들지 않는다. 검수 단계도 같은 검사를 원문 기준으로 한 번 더 한다.
   기준 계획보다 단어 중간 줄바꿈이나 한 글자 줄이 늘어나는 대안도 쓰지 않는다
   (`text_wrapping.worsens_line_breaks`) — 작은 위반을 피하려고 '단어 중간
   줄바꿈은 마지막 수단' 규칙을 우회하지 않게 하려는 것이며, 크기 축소·외톨이
   끝줄 단계의 대안에도 같은 검사가 걸린다.
3. 크기 축소 — 1px씩 최대 4단계, `원문 크기 x MODEL_FONT_SIZE_FLOOR_RATIO`
   아래로는 내려가지 않는다.

후보 채택 규칙: 세 위반 항목 중 **어느 것도 늘지 않으면서** 합이 줄어들
때만 채택 (겹침을 줄이려고 잘림을 새로 만드는 맞바꿈 금지). 첫 후보가
0이 되면 멈추고, 끝까지 0이 안 되면 항목별로 나빠지지 않은 최선을 남긴다.

확정 위반이 없을 때만 같은 크기의 '보기 좋게' 단계가 이어진다:

- 외톨이 끝줄(한 글자만 남은 끝줄) 제거 — 피팅 점수가 기준의 -4점 이내인
  줄바꿈 대안 최대 3개. 원문에 `\n`이 있으면 손대지 않는다. '가자'·'먹어' 같은
  두 글자 완결 단어는 한 줄로 서도 자연스러워 외톨이로 보지 않는다.
- 시각적 중앙 — 말풍선 피팅 목표 영역 안에서 잉크 좌우/상하 여백이 같아지게
  정수 이동 (글자 크기의 6% 이하 치우침은 무시, 25% 초과는 배치 문제로 보고
  손대지 않음). 붙은 쪽 정렬 축은 움직이지 않는다. 프리텍스트에는 적용하지 않는다.

렌더 횟수 상한: 요소당 `1 + 6 + 4 + 3`. 위반이 없는 요소는 합성용 렌더
1회만 쓰므로 추가 비용은 알파 분석(작은 numpy 연산)뿐이다.
합성은 확정 후 한 번만 하므로 기각된 후보가 흔적을 남기지 않는다.

진단: `PageTypesetReport` (요소별 전/후 계획, 기준·최종 위반 픽셀, 전략,
렌더 횟수, 통일 조정 목록). 미해결은 `[typeset-qa] ... 미해결` WARNING,
보정·통일은 INFO 로그.

## Page-Level Dialogue Consistency (페이지 대사 크기 통일)

모듈: `src/page_consistency.py`. 말풍선 계획이 끝난 뒤, 검수 전에 한 번 돈다.

- 대상: 말풍선 안 가로쓰기·비회전이고 스타일이 `standard/scared`인 대사.
  외침·강조·해설·손글씨·세로·회전은 비교하지 않는다.
- 비교 기준은 포인트 크기가 아니라 **겉보기 글자 몸통 높이** — 고정 기준
  문자열을 그 계획의 글꼴·스타일·크기로 잰 `measure_character_body_size`
  값이라 글꼴이 달라도 비교된다. (굵기는 비교·조정하지 않는다.)
- 표본(비교할 대사)이 4개 이상일 때만 돈다.
- 다수 무리 맞추기: 인접 비 1.22 이내로 이어지는 다수 무리가 60% 이상이고 무리
  안 최대/최소 비가 1.35 이내면 합의로 본다. 무리 중앙값 기준 ±4% 이내는 무시,
  그 밖은 요소별 최대 12%(`page_consistency.MAX_ADJUST_RATIO`)와 원문 크기 허용
  오차 창 안에서만 이동. 정수 크기로 내릴 때는 현재 크기 쪽으로 버려 상한을
  넘지 않는다.
- 띠 맞추기(`_band_adjustments`): 그 뒤(합의가 없어도) 무리 밖 대사까지 쪽 중앙
  몸통 높이의 ±15% 띠 끝으로 당긴다 — 원문 크기 허용 오차 창 안에서, 몸통이 쪽
  중앙의 0.7배보다 작은 대사는 창 위 끝을 `MAX_FONT_SIZE`까지 연다. 원문 크기가
  쪽 대사들의 원문 크기 중앙의 1.25배 이상인 줄(해설 상자처럼 일부러 크게 쓴 줄)은
  끌어내리지 않는다.
- 새 크기에서 기존 전략으로 다시 줄바꿈하고, 목표 영역에 안 들어가거나
  단어 중간 줄바꿈·한 글자 줄이 늘어나면 그 크기는 쓰지 않는다.
  이어서 실제 알파로 재검증해 잘림·안전 영역 이탈·다른 말풍선 글자와의
  겹침이 조정 전보다 늘면 그 조정도 버린다. 검수 단계는 통일된 계획을 기준으로
  다시 검사한다.
- 프리텍스트 측정 크기 통일은 `page_structure.harmonize_freeform_sizes`가 따로
  맡고, 이 단계는 말풍선 대사에만 적용된다.

## Same-Size Groups (맞닿은 말풍선·같은 칸 평문)

모듈: `src/page_drawer.py`. 쪽 대사 크기 통일 뒤, 검수 전에 돈다.

- 맞닿은 말풍선(`_unify_attached_sizes`): 상자 틈 6px 안으로 맞닿거나 겹치고
  마주 보는 길이가 작은 쪽의 0.3배 이상인 말풍선끼리 묶는다. 무리마다 쪽 보통
  대사 크기 쪽으로, 모두 들어가는 가장 큰 크기 하나로 맞춘다(키우는 쪽은 알파로
  재검증). 검수가 무리 하나만 더 줄이면 나머지도 그 크기로 다시 검수한다.
- 같은 칸 말풍선 밖 평문(`_panel_narration_sizes`): `reading_order.find_panels`로
  나눈 같은 칸의 보통체·해설 글을 묶어, 가로 글은 가로 글끼리·세로 글은 세로
  글끼리 가장 작은 계획 크기로 맞춘다.
- 둘 다 글씨체(보통·해설·굵은 대사)가 같고 원문 글자 크기(글자 폭으로 잰 보이는
  크기)가 1.2배 안인 글만 한 무리가 된다 — 크기 순으로 가장 작은 글의 1.2배까지.

## Unresolved Cases (해결하지 않는 것)

- 안쪽 흰 영역을 잴 수 없는 말풍선(검은·톤 말풍선)의 실제 윤곽·그림 침범 판단 — 사각 경계만 본다
- 정책 하한까지 줄여도 안전 영역에 못 넣는 긴 번역 — 미해결로 기록만
- 붙은 쪽 정렬 때문에 안쪽 이동만 허용되어 못 피하는 겹침
- 프리텍스트끼리·프리텍스트 대 말풍선 글자 겹침 중 크기 하한에 걸린 경우
- 프리텍스트의 시각적 중앙 정렬 (위쪽 맞춤이 의도된 배치라 적용하지 않음)

## Remaining Work

- 활용 꼬리가 아닌 곳의 단어 중간 줄바꿈은 아직 자리 선호가 없다 — 이름·명사
  한가운데('에스페/리아', '아이스크/림')나 조사 앞('선생님/은', '모두에/게')
- 페이지 통일이 크게 쓴 강조 대사(외침 스타일로 분류되지 않은 것)를 페이지
  중앙값 쪽으로 줄이는 경우가 있다
- 굵기(weight) 비교는 하지 않음 — 필요하면 글꼴별 잉크 밀도 측정 추가 검토
