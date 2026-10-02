"""탐지·인식 결과를 페이지 단위 데이터(PageData)로 구조화.

- 말풍선 안·밖 분류로 두 번 잡힌 같은 글자는 말풍선 안 쪽만 남김
- 글자 블록 → 말풍선 배정 (중심이 든 가장 작은 말풍선 우선 + 거리 상한)
- 한 말풍선의 여러 블록은 읽는 순서로 합치고, 멀리 떨어진 블록과 짝 없는 블록은
  말풍선 밖 글자로 살린다 (대사를 버리지 않는다)
- 패널 테두리 붙음(attachment) 판정
"""
import logging
import os
import re
import unicodedata
from dataclasses import replace

import numpy as np

from src import config
from src.data_models import PageData, SpeechBubble
from src.line_detector import detect_bubble_attachment, detect_freeform_attachment
from src.utils import is_box_inside

logger = logging.getLogger(__name__)

# 한 말풍선 안 두 블록 사이가 글자 크기의 이 배수보다 멀면 합치지 않고 따로 살린다 —
# 탐지가 말풍선 두 개를 하나로 잡았을 때 서로 다른 대사를 한 덩어리로 식자하지 않도록.
MERGE_MAX_GAP_FONT_RATIO = 1.5
# 글자 크기(font_size ≈ 글자 높이×1.18)가 박스 짧은 변의 이 배수를 넘어야 측정 폭주로 본다 — 한 줄 제목은
# 짧은 변의 0.9배 안팎이고, 두 배로 잘못 잰 값은 1.8배쯤 된다
BLOWUP_BOX_RATIO = 1.25
# 말풍선 안·밖 두 분류로 같은 글자를 잡았다고 볼 겹침 (교집합 ÷ 작은 쪽 넓이)
DUPLICATE_OVERLAP = 0.6


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _assign_bubble(text_box, bubbles):
    """글자 블록이 속할 말풍선 번호 — 중심이 안에 든 말풍선 중 가장 작은 것, 없으면
    말풍선 대각선 절반 이내에서 가장 가까운 것 (없으면 None).

    말풍선마다 가장 가까운 글자를 고르면 OCR 필터로 비어버린 말풍선이 옆 말풍선의
    글자를 훔쳐가므로, 글자 쪽에서 자기 말풍선을 고른다.
    """
    cx, cy = _center(text_box)
    inside = [(_area(b), i) for i, b in enumerate(bubbles) if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
    if inside:
        return min(inside)[1]
    best = None
    for i, b in enumerate(bubbles):
        bx, by = _center(b)
        distance = float(np.hypot(bx - cx, by - cy))
        if distance <= float(np.hypot(b[2] - b[0], b[3] - b[1])) * 0.5 and (best is None or distance < best[0]):
            best = (distance, i)
    return None if best is None else best[1]


def _box_gap(a, b):
    """두 박스 사이의 빈 거리 (겹치면 0)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def _proximity_groups(elements):
    """서로 가까운 블록끼리 묶은 무리들 (넓이가 큰 무리부터)."""
    parent = list(range(len(elements)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            limit = MERGE_MAX_GAP_FONT_RATIO * max(elements[i].font_size, elements[j].font_size, 1)
            if _box_gap(elements[i].text_box, elements[j].text_box) <= limit:
                parent[find(i)] = find(j)
    groups = {}
    for i, element in enumerate(elements):
        groups.setdefault(find(i), []).append(element)
    return sorted(groups.values(), key=lambda group: -sum(_area(e.text_box) for e in group))


def _reading_order(elements):
    """한 말풍선 안 블록의 읽는 순서: 윗줄부터, 같은 줄(세로 범위가 반 이상 겹침)은 오른쪽부터."""
    rows = []
    for element in sorted(elements, key=lambda e: e.text_box[1]):
        top, bottom = element.text_box[1], element.text_box[3]
        for row in rows:
            overlap = min(bottom, row["bottom"]) - max(top, row["top"])
            if overlap > 0.5 * min(bottom - top, row["bottom"] - row["top"]):
                row["items"].append(element)
                row["top"], row["bottom"] = min(top, row["top"]), max(bottom, row["bottom"])
                break
        else:
            rows.append({"top": top, "bottom": bottom, "items": [element]})
    ordered = []
    for row in sorted(rows, key=lambda r: r["top"]):
        ordered.extend(sorted(row["items"], key=lambda e: -e.text_box[2]))
    return ordered


def _merge_elements(elements):
    """블록들을 한 대사로 합친다 — 글씨체·크기·기울기는 가장 큰 블록을 따른다."""
    if len(elements) == 1:
        return elements[0]
    ordered = _reading_order(elements)
    main = max(ordered, key=lambda e: _area(e.text_box))
    boxes = [e.text_box for e in ordered]
    ruby = list(dict.fromkeys(pair for e in ordered for pair in (e.furigana or [])))
    return replace(
        main,
        text_box=[min(b[0] for b in boxes), min(b[1] for b in boxes),
                  max(b[2] for b in boxes), max(b[3] for b in boxes)],
        original_text=" ".join(e.original_text for e in ordered),
        furigana=ruby or None,
    )


def _add(stats, key, amount=1):
    stats[key] = stats.get(key, 0) + amount


def _overlap_of_smaller(a, b):
    """교집합 ÷ 두 박스 중 작은 쪽 넓이."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min(_area(a), _area(b))
    return ix * iy / smaller if smaller > 0 else 0.0


def _same_reading(a, b):
    """두 판독이 같거나 한쪽이 다른 쪽을 품는지 (공백·전각 차이 무시)."""
    a = re.sub(r"\s+", "", unicodedata.normalize("NFKC", a or ""))
    b = re.sub(r"\s+", "", unicodedata.normalize("NFKC", b or ""))
    return bool(a) and bool(b) and (a in b or b in a)


def drop_duplicate_free_texts(page_texts, free_texts):
    """말풍선 안 글자와 같은 자리·같은 판독의 말풍선 밖 글자를 버린다 (말풍선 안 쪽을 남긴다).

    탐지기가 한 글자 덩어리를 두 분류로 다 잡으면 같은 대사를 두 번 식자한다. 분류가 달라
    박스 병합에서 합쳐지지 않으므로, 작은 쪽 넓이의 DUPLICATE_OVERLAP 이상 겹치고 판독이
    같거나 포함 관계일 때만 중복으로 본다 — 겹쳐도 다른 글자면 둘 다 살린다.
    """
    kept = []
    for free in free_texts:
        if any(_overlap_of_smaller(free.text_box, other.text_box) >= DUPLICATE_OVERLAP
               and _same_reading(free.original_text, other.original_text)
               for other in list(page_texts) + kept):
            continue
        kept.append(free)
    return kept


def structure_page_data(batch_paths, batch_images_rgb, all_bubbles_by_page, processed_text_elements,
                        stats=None):
    """Build final PageData objects for a batch.

    stats(dict)가 오면 한 말풍선으로 합친 블록 수(merged_blocks)와 말풍선 밖 글자로 살린
    블록 수(kept_as_freeform)를 더한다.
    """
    stats = stats if stats is not None else {}
    batch_page_data = []
    for page_idx, path in enumerate(batch_paths):
        page_data = PageData(source_page=os.path.basename(path), image_rgb=batch_images_rgb[page_idx])
        page_bubbles = all_bubbles_by_page[page_idx]
        page_texts = [
            item["element"] for item in processed_text_elements
            if item["page_idx"] == page_idx and item["class_name"] == "text"
        ]
        free_texts = [
            item["element"] for item in processed_text_elements
            if item["page_idx"] == page_idx and item["class_name"] == "free_text"
        ]
        before = len(free_texts)
        free_texts = drop_duplicate_free_texts(page_texts, free_texts)
        _add(stats, "duplicates", before - len(free_texts))

        by_bubble = {}
        for element in page_texts:
            index = _assign_bubble(element.text_box, page_bubbles)
            if index is None:
                free_texts.append(element)  # 말풍선을 못 찾은 대사도 버리지 않는다
                _add(stats, "kept_as_freeform")
            else:
                by_bubble.setdefault(index, []).append(element)

        for index, bubble_box in enumerate(page_bubbles):
            members = by_bubble.get(index)
            if not members:
                # 말풍선 안 글자가 '말풍선 밖 글자'로 탐지된 경우
                members = [ft for ft in free_texts if is_box_inside(ft.text_box, bubble_box)]
                if not members:
                    continue
                free_texts = [ft for ft in free_texts if not any(ft is m for m in members)]
            groups = _proximity_groups(members)
            for extra in groups[1:]:
                free_texts.extend(extra)  # 멀리 떨어진 블록은 합치지 않고 따로 살린다
                _add(stats, "kept_as_freeform", len(extra))
            _add(stats, "merged_blocks", len(groups[0]) - 1)

            b = bubble_box
            cropped_bubble_rgb = page_data.image_rgb[b[1]:b[3], b[0]:b[2]]
            attachment = detect_bubble_attachment(
                cropped_bubble_rgb,
                edge_ratio=config.BUBBLE_ATTACHMENT_EDGE_RATIO,
                min_length_ratio=config.BUBBLE_ATTACHMENT_MIN_LENGTH_RATIO,
            )
            page_data.speech_bubbles.append(SpeechBubble(
                bubble_box=b.tolist(),
                text_element=_merge_elements(groups[0]),
                attachment=attachment,
            ))

        for free_text in free_texts:
            free_text.attachment = detect_freeform_attachment(
                page_data.image_rgb,
                free_text.text_box,
                search_px=config.FREEFORM_ATTACHMENT_SEARCH_PX,
                min_length_ratio=config.FREEFORM_ATTACHMENT_MIN_LENGTH_RATIO,
            )
        new_sizes = harmonize_freeform_sizes(
            [(ft.font_style, ft.font_size) for ft in free_texts],
            boxes=[ft.text_box for ft in free_texts],
        )
        for free_text, new_size in zip(free_texts, new_sizes):
            if new_size != free_text.font_size:
                free_text.font_size = new_size
        page_data.freeform_texts = free_texts
        batch_page_data.append(page_data)

    return batch_page_data


def harmonize_freeform_sizes(style_size_pairs, boxes=None):
    """같은 페이지·같은 스타일의 프리텍스트 크기를 다수 클러스터에 맞춰 통일합니다.

    원본 만화의 모놀로그 컬럼들은 한 페이지에서 같은 크기로 식자되는데,
    크롭별 측정 잡음(±10%)과 드문 측정 폭주(흰 테두리 글자 등) 때문에
    번역본이 들쭉날쭉해진다.

    규칙 (스타일별, 4개 이상일 때만):
    - 크기를 정렬해 22% 간격 이내로 이어지는 클러스터를 만들고,
      최대 클러스터가 전체의 60% 이상이면 그 중앙값을 기준으로:
      · ±25% 안의 잔잡음 → 기준값으로 스냅
      · 기준의 1.5배 초과 → 측정 폭주로 보고 기준값으로 클램프. 단 boxes가 오면 자기 박스에 들어가는
        크기(짧은 변의 BLOWUP_BOX_RATIO배 이하)는 제목처럼 진짜 큰 글자로 보고 유지한다 — 작은 글자가
        많은 쪽에서 제목까지 작게 눌리지 않도록. 폭주한 측정값은 박스 안에 들어가지 않는다.
      · 기준의 0.75배 미만 → 의도된 작은 글씨(주석 등)로 보고 유지
    Returns: 입력 순서대로의 새 크기 리스트.
    """
    new_sizes = [size for _, size in style_size_pairs]

    def fits_box(idx, size):
        if boxes is None:
            return False
        x1, y1, x2, y2 = boxes[idx][:4]
        return size <= BLOWUP_BOX_RATIO * min(x2 - x1, y2 - y1)

    by_style = {}
    for idx, (style, size) in enumerate(style_size_pairs):
        by_style.setdefault(style, []).append((idx, size))

    for style, members in by_style.items():
        if len(members) < 4:
            continue
        ordered = sorted(members, key=lambda m: m[1])
        clusters = [[ordered[0]]]
        for item in ordered[1:]:
            if item[1] <= clusters[-1][-1][1] * 1.22:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        largest = max(clusters, key=len)
        if len(largest) * 10 < len(members) * 6:  # 60% 미만이면 합의 없음
            continue
        center = int(round(float(np.median([s for _, s in largest]))))
        for idx, size in members:
            if abs(size - center) <= center * 0.25 or (size > center * 1.5 and not fits_box(idx, size)):
                new_sizes[idx] = center

    return new_sizes
