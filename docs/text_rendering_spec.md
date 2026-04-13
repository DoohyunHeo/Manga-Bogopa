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
- a preferred lower bound of `ceil(model_predicted_font_size * 0.9)` before relaxed fallback search

## Remaining Work

The next implementation steps should add:

- per-role render profile selection outside the current `font_style` mapping
- stronger font-family policy based on official-license or community-approved presets
- regression image set for before/after visual comparison
- page-level consistency scoring so neighboring bubbles do not diverge too much in apparent weight
