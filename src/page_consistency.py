"""페이지 단위 대사 글씨 크기 통일 계획.

같은 페이지의 '평범한 대사' 말풍선들은 원작에서 같은 크기로 식자되는데,
크롭별 원문 크기 측정 잡음 때문에 번역본이 조금씩 들쭉날쭉해진다. 이 모듈은
피팅이 끝난 계획(TextRenderPlan)들을 모아, 포인트 크기가 아니라 '보이는 글자
몸통 높이'(measure_character_body_size — 글꼴이 달라도 비교 가능)를 기준으로
다수 무리를 찾고, 그 중앙값으로 제한된 범위 안에서만 끌어당긴다.

규칙:
- 대상: 말풍선 안 가로쓰기·비회전·대사 스타일(DIALOGUE_STYLES)만, 표본 MIN_SAMPLES개 이상일 때
- 다수 무리가 CLUSTER_MAJORITY 이상이고 무리 안 최대/최소 몸통 비가 CLUSTER_MAX_SPREAD 이내면 합의로 보고,
  무리를 그 중앙값 쪽으로 당긴다 — 요소별 ±MAX_ADJUST_RATIO, 그리고 원문 크기 허용 오차 창 안
- 그 뒤(합의가 없어도) 무리 밖 대사까지 쪽 중앙 몸통 높이의 ±BAND_RATIO 띠 끝으로 당긴다 (_band_adjustments) —
  원문 크기 허용 오차 창 안에서, 몸통이 쪽 중앙의 BAND_HARD_FLOOR배보다 작은 대사는 창 위 끝을 MAX_FONT_SIZE까지
  연다. 원문 크기가 쪽 대사들의 원문 크기 중앙의 BAND_KEEP_BIG배 이상인 줄은 끌어내리지 않는다
- 키운 글이 목표 영역에 안 들어가면 그 요소의 조정은 포기 (이후 검수가 재검증)
글꼴·스타일(굵기 포함)·회전·세로쓰기는 바꾸지 않는다. 줄바꿈은 새 크기에서
기존 전략으로 다시 고른다.
"""
import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import config
from src.text_layout import TextRenderPlan
from src.text_renderer import measure_character_body_size

logger = logging.getLogger(__name__)

# 서로 비교 가능한 '평범한 대사' 스타일. 외침·강조(shouting/angry/pop)와
# 비대사(narration/handwriting)는 의도된 크기 차이가 크므로 제외.
DIALOGUE_STYLES = frozenset({"standard", "scared"})
MIN_SAMPLES = 4
CLUSTER_GAP_RATIO = 1.22      # 정렬 후 인접 몸통 높이 비가 이 이내면 같은 무리
CLUSTER_MAJORITY = 0.6        # 다수 무리가 표본의 이 비율 이상이어야 합의
CLUSTER_MAX_SPREAD = 1.35     # 무리 안 최대/최소 비가 이보다 크면 점진적 차이로 보고 포기
DEADBAND_RATIO = 0.04         # 기준과 이 비율 이내 차이는 잡음으로 보고 건드리지 않음
MAX_ADJUST_RATIO = 0.12       # 다수 무리로 맞출 때 요소 하나의 크기를 이 비율 넘게 키우거나 줄이지 않는다
# 다수 무리를 못 찾거나 무리 밖에 있는 보통 대사도 쪽 중앙 몸통 높이의 ±15% 안으로 모은다 — 같은 역할 대사끼리
# 크기 차이가 커 보이지 않게. 키운 글이 안 들어가면 덜 키운다
BAND_MIN_SAMPLES = 3
BAND_RATIO = 0.15
# 이보다 작으면 원문 크기 허용 창을 넘어서라도 띠 아래 끝까지 키운다 — 1단계가 크기를 작게 잰 대사도 너무 작게 그리지 않게
BAND_HARD_FLOOR = 0.7
# 원문 크기가 쪽 대사들의 원문 크기 중앙보다 이 배 이상 큰 줄은 띠 위 끝으로 끌어내리지 않는다 — 해설 상자처럼 원문에서
# 일부러 크게 쓴 줄이다
BAND_KEEP_BIG = 1.25
# 겉보기 글자 크기를 잴 기준 문자열. 실제 대사 대신 고정 문자열을 쓰는 이유:
# 글자 몸통 높이는 어떤 음절이 들어 있느냐에 따라 같은 글꼴·크기에서도 10~15%
# 흔들리므로, 글꼴·스타일·크기만의 함수가 되도록 고정한다 (글꼴 간 비교 가능).
BODY_REFERENCE_TEXT = "한글식자기준본문"


@dataclass(frozen=True)
class ConsistencyMember:
    key: int
    plan: TextRenderPlan
    style_name: str
    predicted_size: float


@dataclass(frozen=True)
class ConsistencyAdjustment:
    key: int
    text_preview: str
    size_before: int
    size_after: int
    body_before: float
    body_after: float
    cluster_center: float


RewrapFn = Callable[[ConsistencyMember, int], Optional[TextRenderPlan]]


def apparent_body_height(plan: TextRenderPlan) -> float:
    """그 계획의 글꼴·스타일·크기로 기준 문자열을 그렸을 때 글자 몸통 높이 중앙값 (px)."""
    if not (plan.text or "").strip():
        return 0.0
    _, body_height = measure_character_body_size(
        BODY_REFERENCE_TEXT, plan.font_path, int(plan.font_size), plan.style
    )
    return float(body_height)


def is_comparable_dialogue(member: ConsistencyMember) -> bool:
    plan = member.plan
    if not plan.text or not plan.text.strip():
        return False
    if plan.vertical:
        return False
    if abs(plan.angle) > config.MIN_ROTATION_ANGLE:
        return False
    return member.style_name in DIALOGUE_STYLES


def dominant_cluster(items: Sequence[Tuple[int, float]]) -> Optional[List[Tuple[int, float]]]:
    """(key, 몸통 높이) 목록에서 합의로 볼 수 있는 다수 무리를 고른다. 없으면 None."""
    if not items:
        return None
    ordered = sorted(items, key=lambda kv: (kv[1], kv[0]))
    clusters = [[ordered[0]]]
    for item in ordered[1:]:
        if item[1] <= clusters[-1][-1][1] * CLUSTER_GAP_RATIO:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    largest = max(clusters, key=len)  # 동점이면 먼저 나온(작은) 무리 — 결정적
    if len(largest) < MIN_SAMPLES or len(largest) < len(items) * CLUSTER_MAJORITY:
        return None
    if largest[-1][1] > largest[0][1] * CLUSTER_MAX_SPREAD:
        return None
    return largest


def _size_window(member: ConsistencyMember, current: int) -> Tuple[int, int]:
    """원문 크기 허용 오차 창. 현재 크기가 이미 창 밖이면 더 밖으로는 안 보낸다."""
    predicted = max(1.0, float(member.predicted_size))
    floor_ratio = float(config.MODEL_FONT_SIZE_FLOOR_RATIO)
    ceiling_ratio = float(config.MODEL_FONT_SIZE_CEILING_RATIO)
    low = max(config.MIN_FONT_SIZE, math.ceil(predicted * floor_ratio))
    high = min(config.MAX_FONT_SIZE, math.floor(predicted * ceiling_ratio))
    return min(low, current), max(high, current)


def plan_dialogue_consistency(
    members: Sequence[ConsistencyMember],
    rewrap: RewrapFn,
) -> Dict[int, Tuple[TextRenderPlan, ConsistencyAdjustment]]:
    """비교 가능한 대사들의 크기 조정 계획. {key: (새 계획, 조정 기록)}.

    rewrap(member, size)는 그 크기에서 다시 줄바꿈한 계획을 돌려주거나,
    목표 영역에 들어가지 않으면 None을 돌려준다.
    """
    comparable = [m for m in members if is_comparable_dialogue(m)]
    if len(comparable) < MIN_SAMPLES:
        return {}
    bodies: Dict[int, float] = {}
    for member in comparable:
        body = apparent_body_height(member.plan)
        if body > 0.0:
            bodies[member.key] = body
    if len(bodies) < MIN_SAMPLES:
        return {}

    by_key = {m.key: m for m in comparable}
    adjustments: Dict[int, Tuple[TextRenderPlan, ConsistencyAdjustment]] = {}
    cluster = dominant_cluster(list(bodies.items()))
    if cluster is None:
        return _band_adjustments(by_key, bodies, rewrap, adjustments)
    center = float(np.median([body for _, body in cluster]))
    for key, body in cluster:
        member = by_key[key]
        ratio = center / body
        if abs(ratio - 1.0) <= DEADBAND_RATIO:
            continue
        ratio = max(1.0 - MAX_ADJUST_RATIO, min(1.0 + MAX_ADJUST_RATIO, ratio))
        current = int(member.plan.font_size)
        low, high = _size_window(member, current)
        # 정수 크기로 내릴 때 현재 크기 쪽으로 버려야 상한(MAX_ADJUST_RATIO)을 넘지 않는다.
        raw = current * ratio
        target = max(low, min(high, math.floor(raw) if raw > current else math.ceil(raw)))
        if target == current:
            continue
        # 목표 크기부터 현재 크기 쪽으로 한 단계씩 물러나며 들어가는 첫 크기를 택한다.
        step = 1 if target > current else -1
        chosen = None
        for size in range(target, current, -step):
            candidate = rewrap(member, size)
            if candidate is not None:
                chosen = candidate
                break
        if chosen is None:
            continue
        adjustments[key] = (
            chosen,
            ConsistencyAdjustment(
                key=key,
                text_preview=(member.plan.text or "").replace("\n", " ")[:12],
                size_before=current,
                size_after=int(chosen.font_size),
                body_before=body,
                body_after=apparent_body_height(chosen),
                cluster_center=center,
            ),
        )
    return _band_adjustments(by_key, bodies, rewrap, adjustments)


def _band_adjustments(by_key, bodies, rewrap, adjustments):
    """다수 무리 조정 뒤에도 쪽 중앙 몸통 높이의 ±BAND_RATIO 밖에 있는 대사를 띠 끝까지 끌어당긴다."""
    if len(bodies) < BAND_MIN_SAMPLES:
        return adjustments
    current_body = {key: (apparent_body_height(adjustments[key][0]) if key in adjustments else body)
                    for key, body in bodies.items()}
    center = float(np.median(list(current_body.values())))
    low_body, high_body = center * (1.0 - BAND_RATIO), center * (1.0 + BAND_RATIO)
    predicted_center = float(np.median([float(by_key[key].predicted_size or 0) for key in current_body]))
    for key, body in current_body.items():
        if low_body <= body <= high_body or body <= 0.0:
            continue
        member = by_key[key]
        if body > high_body and float(member.predicted_size or 0) >= BAND_KEEP_BIG * predicted_center > 0:
            continue
        plan = adjustments[key][0] if key in adjustments else member.plan
        current = int(plan.font_size)
        low, high = _size_window(member, current)
        if body < BAND_HARD_FLOOR * center:
            high = config.MAX_FONT_SIZE
        target_body = high_body if body > high_body else low_body
        raw = current * target_body / body
        target = max(low, min(high, math.floor(raw) if raw < current else math.ceil(raw)))
        if target == current:
            continue
        step = 1 if target > current else -1
        chosen = None
        # 줄이는 쪽은 목표 크기가 곧 답 (들어가는지는 rewrap이 본다), 키우는 쪽은 들어가는 가장 큰 크기
        for size in range(target, current, -step):
            candidate = rewrap(member, size)
            if candidate is not None:
                chosen = candidate
                break
        if chosen is None:
            continue
        adjustments[key] = (chosen, ConsistencyAdjustment(
            key=key, text_preview=(plan.text or "").replace("\n", " ")[:12], size_before=int(member.plan.font_size),
            size_after=int(chosen.font_size), body_before=bodies[key], body_after=apparent_body_height(chosen),
            cluster_center=center))
    return adjustments
