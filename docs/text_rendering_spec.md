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
- Line breaks should occur on whitespace boundaries only; words must not be split apart.

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

- min target dimension <= 70px
- predicted font size <= `MIN_READABLE_TEXT_SIZE`

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
- mid-word line breaks

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
- freeform attachment: anchor text at the original detection box edge (pulled
  by `FREEFORM_ATTACHMENT_TEXT_MARGIN`), preserved through overlap adjustment
- vertical layout ignores attachment (it always centers)

## Vertical Fallback Policy

Vertical layout is entered via one of three paths:

1. **Natural vertical** — `_is_vertical()` approves based on
   `VERTICAL_TEXT_THRESHOLD` (default 4:1 aspect) AND
   `MIN_READABLE_TEXT_SIZE` AND no whitespace in the translation.
2. **Extreme aspect override** — if the text box aspect is at least
   `VERTICAL_FORCE_ASPECT_RATIO` (default 6:1), vertical is forced regardless
   of predicted font size or whitespace.
3. **Shrink-based fallback** — if horizontal fitting ends up below
   `FONT_SHRINK_THRESHOLD_RATIO * element.font_size`, the horizontal result is
   discarded and vertical is retried from the original predicted size.

## Freeform Style Consolidation

For free-text items (narration, SFX, ambient text) the "standard" style does
not apply — those items are treated as narration. The style chooser
(`extractor._choose_style_name`) enforces this in one place:

- `standard` prediction on free-text → rewritten to `narration`
- any prediction below `FREEFORM_STYLE_MIN_CONFIDENCE` (default 0.70) on
  free-text → rewritten to `narration`
- bubble text uses the generic confidence fallback to `standard`

## Model Size Tolerance Cap

Font fitting is bounded by the model's predicted size:

- upper cap: `predicted × MODEL_FONT_SIZE_CEILING_RATIO` (hard cap on search)
- lower bound: `predicted × MODEL_FONT_SIZE_FLOOR_RATIO` (soft penalty, not a
  hard floor, so translations that genuinely need smaller text still fit)
- the user-facing setting is `MODEL_FONT_SIZE_TOLERANCE` (symmetric ±), which
  derives both ratios via `apply_font_modes()`
- applies equally to legacy, char-ratio, and stroke-ratio fitting paths

## Implemented Now

The current implementation includes:

- Skia font rendering with subpixel AA and font-level horizontal scaling
- oversampled transparent text-layer rendering followed by downsampling
- transparent text compositing instead of full-image redraw per bubble
- darker dialogue defaults with thicker-looking small-text rendering
- small-bubble readability overrides
- tall-bubble condensed silhouette overrides
- Korean-aware wrapping before fallback wrapping strategies
- whitespace-only balanced wrapping for wide bubbles
- split penalties for Korean particles, forbidden line heads, and forbidden line tails
- candidate scoring based on font size, overflow, fill ratio, orphan lines, and tall-bubble silhouette balance
- model-prediction size tolerance caps on both horizontal and vertical fit paths
- panel-border attachment detection and aligned rendering for bubble + freeform text
- triple entry to vertical layout: natural aspect, extreme aspect override, shrink fallback
- freeform items auto-coerced to narration when style confidence is low

## Module Layout

- `src/text_renderer.py` — skia-backed drawing + `measure_*` primitives
- `src/text_wrapping.py` — wrap strategies (Korean-aware, balanced, aggressive)
- `src/text_fitting.py` — horizontal + vertical font-size search with scoring
- `src/text_layout.py` — style resolution, vertical decisions, plan assembly
- `src/line_detector.py` — panel-border detection for attachment alignment
- `src/page_drawer.py` — page-level draw orchestration

## Remaining Work

The next implementation steps should add:

- page-level consistency scoring so neighboring bubbles do not diverge too much in apparent weight
- regression image set for before/after visual comparison
