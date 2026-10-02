"""식자 결과 검수(QA)와 범위가 정해진 자동 보정.

합성 직전의 실제 글자 레이어(외곽선 포함, 회전·세로쓰기 반영)의 알파 채널을 재서
다음 세 가지 '확정 위반'을 픽셀 단위로 센다. 레이어의 투명 여백(padding)은
잉크로 세지 않는다.

- clipped   : 이미지 밖으로 잘리는 잉크
- outside   : 말풍선의 사각 안전 영역 밖이거나, 말풍선 안쪽 흰 영역의 테두리까지 거리 지도에서 테두리에
              글자 크기의 interior_margin_ratio배보다 가까운(흰 영역 밖 포함) 잉크 (말풍선 텍스트만)
- collision : 먼저 확정된 다른 글자 잉크와 실제로 겹치는 잉크

보정은 합성 전에 후보 계획(TextRenderPlan) 단위로만 시도하므로, 기각된 후보가
이미지에 흔적(ghost)을 남기지 않는다. 시도 순서와 상한:

1. 배치 이동   — 재렌더 없이 오프셋만 바꿔 평가 (붙은 쪽 정렬은 안쪽 이동만 허용)
2. 줄바꿈 대안 — 같은 크기에서 서로 다른 줄바꿈 최대 MAX_WRAP_ALTERNATIVES개
3. 크기 축소   — 정책 하한(원문 크기 x 허용 오차) 위에서 최대 MAX_SIZE_STEPS단계

후보는 세 위반 항목 중 어느 것도 늘지 않으면서 합이 '엄격히 줄어들 때만' 채택한다
(겹침을 줄이려고 잘림을 새로 만드는 맞바꿈 금지). 확정 위반이 없어진 뒤에만
같은 크기의 '보기 좋게' 단계가 따라온다: 외톨이 끝줄 제거(줄바꿈 대안),
사각 목표 영역 안 잉크 여백 균형(시각적 중앙 정렬, 이동만). 둘 다 확정 위반을
새로 만들면 채택하지 않는다.

말풍선의 실제 모양은 안쪽 흰 영역을 잴 수 있을 때만 본다 (가시·곡선 테두리와 말풍선 안으로 들어온 그림 포함) —
흰 영역을 못 잰 말풍선(검은·톤 말풍선)은 사각 경계만 본다.
해결되지 않은 위반은 구조화된 리포트와 로그로 남긴다.
"""
import logging
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src import config
from src.text_layout import TextRenderPlan
from src.text_renderer import (
    compose_rotated_text_layer,
    compose_text_layer,
    compose_vertical_text_layer,
)
from src.text_wrapping import is_layout_equivalent, worsens_line_breaks

logger = logging.getLogger(__name__)

# 리샘플(LANCZOS/BICUBIC) 번짐으로 생기는 옅은 알파는 잉크로 세지 않는다.
INK_ALPHA_THRESHOLD = 24
# 보정 시도 상한 — 요소 하나당 렌더 횟수는 최대 1 + MAX_WRAP_ALTERNATIVES + MAX_SIZE_STEPS.
MAX_WRAP_ALTERNATIVES = 6
MAX_SIZE_STEPS = 4
# 확정 위반이 없을 때 시도하는 '보기 좋게' 단계 상한
MAX_SOFT_ALTERNATIVES = 3
# 시각적 중앙 정렬 이동 상한 (글자 크기 대비) — 그 이상 치우쳐 보이면 배치 문제이므로 손대지 않음
OPTICAL_CENTER_MAX_RATIO = 0.25
# 시각적 중앙 정렬 최소 이동 (글자 크기 대비) — 이보다 작은 치우침은 건드리지 않음
OPTICAL_CENTER_DEADBAND_RATIO = 0.06
# 배치 이동 후보를 만들 때 고려하는 충돌 상대 수 (렌더 없음, 평가만)
MAX_COLLIDERS_FOR_SHIFT = 2

Rect = Tuple[int, int, int, int]
PlanFactory = Callable[[TextRenderPlan], Sequence[TextRenderPlan]]


@dataclass(frozen=True)
class RenderedText:
    """계획 하나를 실제로 렌더한 결과 — 합성용 레이어와 잉크 마스크를 함께 든다."""
    plan: TextRenderPlan
    layer: Optional[Image.Image] = field(compare=False, repr=False)
    paste_x: int = 0
    paste_y: int = 0
    bbox: Rect = (0, 0, 0, 0)
    ink_mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    ink_offset: Tuple[int, int] = (0, 0)  # 레이어 좌표계 안에서 잉크 마스크의 원점

    @property
    def ink_bbox(self) -> Optional[Rect]:
        """실제 잉크(알파 임계 이상)의 페이지 좌표 사각형. 잉크가 없으면 None."""
        if self.ink_mask is None:
            return None
        ox, oy = self.ink_offset
        h, w = self.ink_mask.shape
        x = self.paste_x + ox
        y = self.paste_y + oy
        return (x, y, x + w, y + h)

    def shifted(self, dx: int, dy: int) -> "RenderedText":
        """정수 픽셀만큼 옮긴 사본 (재렌더 없음 — 정수 이동이라 결과 픽셀이 같다)."""
        dx, dy = int(dx), int(dy)
        if dx == 0 and dy == 0:
            return self
        x1, y1, x2, y2 = self.bbox
        return replace(
            self,
            plan=replace(self.plan, center_x=self.plan.center_x + dx, center_y=self.plan.center_y + dy),
            paste_x=self.paste_x + dx,
            paste_y=self.paste_y + dy,
            bbox=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
        )


def _ink_mask(layer: Image.Image):
    alpha = np.asarray(layer.getchannel("A"))
    ink = alpha >= INK_ALPHA_THRESHOLD
    if not ink.any():
        return None, (0, 0)
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return np.ascontiguousarray(ink[y0:y1, x0:x1]), (x0, y0)


def render_plan(plan: TextRenderPlan) -> RenderedText:
    """계획을 합성 없이 렌더한다 (page_drawer의 그리기 분기와 같은 규약)."""
    if plan.vertical:
        layer, paste_x, paste_y, bbox = compose_vertical_text_layer(
            plan.text, plan.center_x, plan.center_y, plan.font_path, plan.font_size, plan.style,
            max_column_height=plan.vertical_column_height, max_columns=plan.vertical_columns,
        )
    elif abs(plan.angle) > config.MIN_ROTATION_ANGLE:
        layer, paste_x, paste_y, bbox = compose_rotated_text_layer(
            plan.text, plan.center_x, plan.center_y, plan.angle,
            plan.font_path, plan.font_size, plan.style, plan.align,
        )
    else:
        layer, paste_x, paste_y, bbox = compose_text_layer(
            plan.text, plan.center_x, plan.center_y,
            plan.font_path, plan.font_size, plan.style, plan.align, plan.anchor,
        )
    mask, offset = (None, (0, 0))
    if layer is not None:
        mask, offset = _ink_mask(layer)
    return RenderedText(plan=plan, layer=layer, paste_x=paste_x, paste_y=paste_y, bbox=bbox,
                        ink_mask=mask, ink_offset=offset)


def composite(img_pil: Image.Image, rendered: RenderedText) -> Rect:
    """확정된 렌더 결과를 이미지에 합성하고 기존 규약의 bbox를 돌려준다."""
    if rendered.layer is not None:
        img_pil.paste(rendered.layer, (rendered.paste_x, rendered.paste_y), rendered.layer)
    return rendered.bbox


# ── 검사 ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QAContext:
    img_size: Tuple[int, int]
    safe_rect: Optional[Rect] = None      # 말풍선 사각 안전 영역 — 벗어나면 확정 위반
    optical_rect: Optional[Rect] = None   # 잉크 여백 균형을 맞출 사각 영역 (None이면 안 함)
    occupied: Tuple[RenderedText, ...] = ()
    # 말풍선 안쪽 흰 영역의 테두리까지 거리 지도 (거리, 지도의 쪽 좌표 x, y) — 테두리에서 글자 크기의
    # interior_margin_ratio배보다 가까운 잉크도 안전 영역을 벗어난 것으로 센다 (None이면 사각 영역만)
    interior: Optional[tuple] = field(default=None, compare=False, hash=False)
    interior_margin_ratio: float = 0.0


@dataclass(frozen=True)
class QAResult:
    clipped_px: int = 0
    outside_px: int = 0
    collision_px: int = 0

    @property
    def hard(self) -> int:
        return self.clipped_px + self.outside_px + self.collision_px

    @property
    def ok(self) -> bool:
        return self.hard == 0


def improves(new: "QAResult", old: "QAResult") -> bool:
    """항목별로 하나도 늘지 않고 합이 줄어들 때만 '개선'. (겹침->잘림 맞바꿈 금지)"""
    return (
        new.clipped_px <= old.clipped_px
        and new.outside_px <= old.outside_px
        and new.collision_px <= old.collision_px
        and new.hard < old.hard
    )


def _pixels_outside(mask: np.ndarray, ox: int, oy: int, rect: Rect) -> int:
    total = int(mask.sum())
    h, w = mask.shape
    x1 = max(0, rect[0] - ox)
    y1 = max(0, rect[1] - oy)
    x2 = min(w, rect[2] - ox)
    y2 = min(h, rect[3] - oy)
    if x2 <= x1 or y2 <= y1:
        return total
    return total - int(mask[y1:y2, x1:x2].sum())


def _pixels_off_shape(mask: np.ndarray, ox: int, oy: int, rect: Optional[Rect], interior, margin: float) -> int:
    """사각 영역 밖이거나 안쪽 흰 영역의 테두리에서 margin보다 가까운(영역 밖 포함) 잉크 픽셀 수."""
    ys, xs = np.nonzero(mask)
    px, py = xs + ox, ys + oy
    off = np.zeros(len(xs), dtype=bool)
    if rect is not None:
        off |= (px < rect[0]) | (px >= rect[2]) | (py < rect[1]) | (py >= rect[3])
    dist, x0, y0 = interior
    qx, qy = px - x0, py - y0
    inside = (qx >= 0) & (qx < dist.shape[1]) & (qy >= 0) & (qy < dist.shape[0])
    near = np.ones(len(xs), dtype=bool)
    near[inside] = dist[qy[inside], qx[inside]] < margin
    return int((off | near).sum())


def overlap_pixels(a: RenderedText, b: RenderedText) -> int:
    """두 렌더 결과의 잉크가 실제로 겹치는 픽셀 수 (사각형 교차 후 마스크 AND)."""
    ra, rb = a.ink_bbox, b.ink_bbox
    if ra is None or rb is None:
        return 0
    x1, y1 = max(ra[0], rb[0]), max(ra[1], rb[1])
    x2, y2 = min(ra[2], rb[2]), min(ra[3], rb[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    ma = a.ink_mask[y1 - ra[1]:y2 - ra[1], x1 - ra[0]:x2 - ra[0]]
    mb = b.ink_mask[y1 - rb[1]:y2 - rb[1], x1 - rb[0]:x2 - rb[0]]
    return int(np.logical_and(ma, mb).sum())


def evaluate(rendered: RenderedText, ctx: QAContext) -> QAResult:
    if rendered.ink_mask is None:
        return QAResult()
    ox, oy = rendered.ink_bbox[:2]
    mask = rendered.ink_mask
    width, height = ctx.img_size
    clipped = _pixels_outside(mask, ox, oy, (0, 0, int(width), int(height)))
    if ctx.interior is not None:
        margin = max(1.0, ctx.interior_margin_ratio * float(rendered.plan.font_size or 0))
        outside = _pixels_off_shape(mask, ox, oy, ctx.safe_rect, ctx.interior, margin)
    else:
        outside = _pixels_outside(mask, ox, oy, ctx.safe_rect) if ctx.safe_rect else 0
    collision = sum(overlap_pixels(rendered, other) for other in ctx.occupied)
    return QAResult(clipped_px=clipped, outside_px=outside, collision_px=collision)


# ── 배치 이동 ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShiftPolicy:
    """배치 이동 허용 규칙 — 붙은 쪽(attachment) 정렬을 보존한다. 위아래 이동은 늘 허용한다.

    horizontal: free | inward-left (오른쪽으로만) | inward-right (왼쪽으로만)
    """
    horizontal: str = "free"

    @classmethod
    def for_alignment(cls, align: str, attached: bool) -> "ShiftPolicy":
        if not attached:
            return cls()
        if align == "left":
            return cls(horizontal="inward-left")
        if align == "right":
            return cls(horizontal="inward-right")
        return cls()

    def clamp(self, dx: int, dy: int, cap: int) -> Tuple[int, int]:
        if self.horizontal == "inward-left" and dx < 0:
            dx = 0
        elif self.horizontal == "inward-right" and dx > 0:
            dx = 0
        dx = max(-cap, min(cap, int(dx)))
        dy = max(-cap, min(cap, int(dy)))
        return dx, dy


def _bound_fix(ink: Rect, bound: Rect) -> Tuple[int, int]:
    """잉크 사각형을 경계 안으로 넣는 최소 이동. 경계보다 크면 중앙 정렬."""
    def axis(i1, i2, b1, b2):
        if i2 - i1 > b2 - b1:
            return int(round(((b1 + b2) - (i1 + i2)) / 2))
        if i1 < b1:
            return b1 - i1
        if i2 > b2:
            return b2 - i2
        return 0

    return axis(ink[0], ink[2], bound[0], bound[2]), axis(ink[1], ink[3], bound[1], bound[3])


def _bound_rect(ctx: QAContext) -> Rect:
    width, height = ctx.img_size
    rect = (0, 0, int(width), int(height))
    if ctx.safe_rect:
        s = ctx.safe_rect
        rect = (max(rect[0], s[0]), max(rect[1], s[1]), min(rect[2], s[2]), min(rect[3], s[3]))
    return rect


def placement_candidates(rendered: RenderedText, ctx: QAContext, result: QAResult,
                         policy: ShiftPolicy) -> List[RenderedText]:
    """재렌더 없이 시도할 배치 이동 후보들 (결정적 순서, 중복 제거)."""
    ink = rendered.ink_bbox
    if ink is None:
        return []
    cap = max(ink[2] - ink[0], ink[3] - ink[1], 1)
    bound = _bound_rect(ctx)
    shifts: List[Tuple[int, int]] = []

    def add(dx, dy):
        dx, dy = policy.clamp(dx, dy, cap)
        if (dx, dy) != (0, 0) and (dx, dy) not in shifts:
            shifts.append((dx, dy))

    if result.clipped_px or result.outside_px:
        add(*_bound_fix(ink, bound))

    if result.collision_px:
        colliders = sorted(
            ((overlap_pixels(rendered, other), idx, other) for idx, other in enumerate(ctx.occupied)),
            key=lambda item: (-item[0], item[1]),
        )
        for pixels, _, other in colliders[:MAX_COLLIDERS_FOR_SHIFT]:
            if pixels <= 0:
                continue
            ob = other.ink_bbox
            moves = [
                (0, ob[1] - ink[3]),  # 위로
                (0, ob[3] - ink[1]),  # 아래로
                (ob[0] - ink[2], 0),  # 왼쪽으로
                (ob[2] - ink[0], 0),  # 오른쪽으로
            ]
            for dx, dy in sorted(moves, key=lambda m: abs(m[0]) + abs(m[1])):
                add(dx, dy)
                moved = (ink[0] + dx, ink[1] + dy, ink[2] + dx, ink[3] + dy)
                fx, fy = _bound_fix(moved, bound)
                if fx or fy:
                    add(dx + fx, dy + fy)

    return [rendered.shifted(dx, dy) for dx, dy in shifts]


def _best_placement(rendered: RenderedText, result: QAResult, ctx: QAContext, policy: ShiftPolicy):
    """현 위치와 이동 후보 중 확정 위반이 가장 적은 것 (동점이면 앞선 후보 유지)."""
    best, best_result = rendered, result
    if result.ok:
        return best, best_result
    for candidate in placement_candidates(rendered, ctx, result, policy):
        candidate_result = evaluate(candidate, ctx)
        if improves(candidate_result, best_result):
            best, best_result = candidate, candidate_result
            if best_result.ok:
                break
    return best, best_result


def is_safe_alternative(base: TextRenderPlan, alternative: TextRenderPlan, source_text: str) -> bool:
    """대안 계획이 글자·어절 경계·글꼴·스타일·세로쓰기·회전을 그대로 두고
    줄바꿈/크기/위치만 바꿨는지.

    source_text(번역 원문)를 기준으로 단락 줄바꿈까지 보존을 요구한다. 어절 공백
    삭제·삽입과 글자 변경은 거절한다. base보다 단어 중간 줄바꿈이나 한 글자 줄이
    늘어나는 대안도 거절한다.
    """
    return (
        is_layout_equivalent(source_text, alternative.text)
        and not worsens_line_breaks(source_text, base.text, alternative.text)
        and alternative.font_path == base.font_path
        # 말풍선 밖 글자 테두리 두께는 원문 테두리를 따라 배치 크기마다 다시 정하므로(page_drawer._follow_rim) 비교에서 뺀다
        and replace(alternative.style, stroke_width=base.style.stroke_width) == base.style
        and alternative.vertical == base.vertical
        and alternative.angle == base.angle
        and alternative.align == base.align
        and alternative.anchor == base.anchor
    )


def optical_center_shift(rendered: RenderedText, rect: Rect, policy: ShiftPolicy) -> RenderedText:
    """사각 영역 안에서 잉크 좌우/상하 여백이 같아지도록 정수 이동한 사본 (상한 안에서만).

    붙은 쪽 정렬(policy)이 막는 축은 움직이지 않는다. 이동량이 상한을 넘으면
    배치 자체의 문제로 보고 손대지 않는다 (0 이동).
    """
    ink = rendered.ink_bbox
    if ink is None:
        return rendered
    cap = max(1, int(round(rendered.plan.font_size * OPTICAL_CENTER_MAX_RATIO)))
    deadband = max(1, int(round(rendered.plan.font_size * OPTICAL_CENTER_DEADBAND_RATIO)))
    dx = int(round(((rect[0] + rect[2]) - (ink[0] + ink[2])) / 2))
    dy = int(round(((rect[1] + rect[3]) - (ink[1] + ink[3])) / 2))
    if abs(dx) > cap or abs(dx) <= deadband:
        dx = 0
    if abs(dy) > cap or abs(dy) <= deadband:
        dy = 0
    dx, dy = policy.clamp(dx, dy, cap)
    return rendered.shifted(dx, dy)


# ── 보정 ────────────────────────────────────────────────────────────────────

@dataclass
class RepairOutcome:
    rendered: RenderedText
    baseline: QAResult
    final: QAResult
    renders: int
    strategy: str  # none | 적용 단계들을 '+'로 이은 문자열 (placement/wrap/size/orphan/optical)


def repair(
    plan: TextRenderPlan,
    ctx: QAContext,
    source_text: str,
    policy: ShiftPolicy = ShiftPolicy(),
    wrap_alternatives: Optional[PlanFactory] = None,
    size_alternatives: Optional[PlanFactory] = None,
    soft_wrap_alternatives: Optional[PlanFactory] = None,
    base_rendered: Optional[RenderedText] = None,
) -> RepairOutcome:
    """계획을 검수하고 제한된 순서로 보정 후보를 시도한다. source_text는 번역 원문 — 줄바꿈·크기 대안은
    그 글자·어절 경계·단락 줄바꿈을 그대로 지킬 때만 쓴다 (is_safe_alternative).

    렌더 횟수 상한: 1 + MAX_WRAP_ALTERNATIVES + MAX_SIZE_STEPS + MAX_SOFT_ALTERNATIVES.
    확정 위반 단계는 0이 되는 첫 후보에서 멈추고, 끝까지 0이 안 되면 항목별로
    나빠지지 않은 최선 후보를 남긴다. 확정 위반이 없을 때만 같은 크기의
    '보기 좋게' 단계(외톨이 끝줄 -> 시각적 중앙)를 한 번씩 시도한다.
    """
    base = base_rendered if (base_rendered is not None and base_rendered.plan == plan) else render_plan(plan)
    base_result = evaluate(base, ctx)
    renders = 1
    best, best_result = base, base_result
    strategies: List[str] = []

    if not best_result.ok:
        placed, placed_result = _best_placement(base, base_result, ctx, policy)
        if improves(placed_result, best_result):
            best, best_result = placed, placed_result
            strategies.append("placement")

        stages = (
            ("wrap", wrap_alternatives, MAX_WRAP_ALTERNATIVES),
            ("size", size_alternatives, MAX_SIZE_STEPS),
        )
        for stage_name, factory, limit in stages:
            if best_result.ok:
                break
            if factory is None:
                continue
            for alternative in list(factory(plan))[:limit]:
                if not is_safe_alternative(plan, alternative, source_text):
                    continue
                rendered = render_plan(alternative)
                renders += 1
                rendered, result = _best_placement(rendered, evaluate(rendered, ctx), ctx, policy)
                if improves(result, best_result):
                    best, best_result = rendered, result
                    if stage_name not in strategies:
                        strategies.append(stage_name)
                if best_result.ok:
                    break

    if best_result.ok:
        # 보기 좋게 1: 외톨이 끝줄 — 같은 크기 줄바꿈 대안 (확정 위반이 생기면 기각)
        if soft_wrap_alternatives is not None:
            for alternative in list(soft_wrap_alternatives(best.plan))[:MAX_SOFT_ALTERNATIVES]:
                if not is_safe_alternative(best.plan, alternative, source_text):
                    continue
                rendered = render_plan(alternative)
                renders += 1
                result = evaluate(rendered, ctx)
                if result.ok:
                    best, best_result = rendered, result
                    strategies.append("orphan")
                    break
        # 보기 좋게 2: 사각 영역 안 잉크 여백 균형 (이동만, 재렌더 없음)
        if ctx.optical_rect is not None:
            centered = optical_center_shift(best, ctx.optical_rect, policy)
            if centered is not best:
                result = evaluate(centered, ctx)
                if result.ok:
                    best, best_result = centered, result
                    strategies.append("optical")

    return RepairOutcome(best, base_result, best_result, renders, "+".join(strategies) or "none")


# ── 리포트 ──────────────────────────────────────────────────────────────────

@dataclass
class ElementReport:
    kind: str            # bubble | freeform
    index: int
    text_preview: str
    strategy: str
    renders: int
    baseline: QAResult
    final: QAResult
    plan_before: TextRenderPlan
    plan_after: TextRenderPlan

    @property
    def unresolved(self) -> bool:
        return not self.final.ok

    @property
    def shift(self) -> Tuple[int, int]:
        return (
            int(round(self.plan_after.center_x - self.plan_before.center_x)),
            int(round(self.plan_after.center_y - self.plan_before.center_y)),
        )


@dataclass
class PageTypesetReport:
    """페이지 한 장의 검수·보정·통일 결과 (draw_text_on_image 반환값과 별도)."""
    elements: List[ElementReport] = field(default_factory=list)
    consistency: list = field(default_factory=list)  # page_consistency.ConsistencyAdjustment

    def log(self, page_name: str = ""):
        for adj in self.consistency:
            logger.info(
                "[page-consistency] %s bubble#%d '%s' 크기 %d->%d (글자몸통 %.1f->%.1f, 기준 %.1f)",
                page_name, adj.key, adj.text_preview, adj.size_before, adj.size_after,
                adj.body_before, adj.body_after, adj.cluster_center,
            )
        for e in self.elements:
            if e.strategy == "none" and e.final.ok:
                continue
            dx, dy = e.shift
            if e.unresolved:
                logger.warning(
                    "[typeset-qa] %s %s#%d '%s' 미해결: 잘림=%dpx 말풍선밖=%dpx 겹침=%dpx "
                    "(처음 %dpx -> 최선 %dpx, 렌더 %d회, 전략=%s, 크기 %d->%d, 이동 (%d,%d))",
                    page_name, e.kind, e.index, e.text_preview,
                    e.final.clipped_px, e.final.outside_px, e.final.collision_px,
                    e.baseline.hard, e.final.hard, e.renders, e.strategy,
                    e.plan_before.font_size, e.plan_after.font_size, dx, dy,
                )
            else:
                logger.info(
                    "[typeset-qa] %s %s#%d '%s' 보정=%s: 위반 %dpx -> 0 (렌더 %d회, 크기 %d->%d, 이동 (%d,%d))",
                    page_name, e.kind, e.index, e.text_preview, e.strategy,
                    e.baseline.hard, e.renders,
                    e.plan_before.font_size, e.plan_after.font_size, dx, dy,
                )
