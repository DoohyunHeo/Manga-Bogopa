"""탐지·인식 결과를 페이지 단위 데이터(PageData)로 구조화.

- 말풍선 ↔ 텍스트 매칭 (포함 우선 + 거리 상한)
- 패널 테두리 붙음(attachment) 판정
"""
import logging
import os

import numpy as np

from src import config
from src.data_models import PageData, SpeechBubble
from src.line_detector import detect_bubble_attachment, detect_freeform_attachment
from src.utils import is_box_inside

logger = logging.getLogger(__name__)


def structure_page_data(batch_paths, batch_images_rgb, all_bubbles_by_page, processed_text_elements):
    """Build final PageData objects for a batch."""
    batch_page_data = []
    for page_idx, path in enumerate(batch_paths):
        page_data = PageData(source_page=os.path.basename(path), image_rgb=batch_images_rgb[page_idx])
        page_bubbles = all_bubbles_by_page[page_idx]

        page_text_elements = [
            item for item in processed_text_elements
            if item["page_idx"] == page_idx and item["class_name"] == "text"
        ]
        page_free_texts = [
            item["element"] for item in processed_text_elements
            if item["page_idx"] == page_idx and item["class_name"] == "free_text"
        ]

        unmatched_text_indices = set(range(len(page_text_elements)))
        unmatched_free_text_indices = set(range(len(page_free_texts)))

        for bubble_box in page_bubbles:
            bubble_center = ((bubble_box[0] + bubble_box[2]) / 2, (bubble_box[1] + bubble_box[3]) / 2)
            bubble_diag = float(np.hypot(bubble_box[2] - bubble_box[0], bubble_box[3] - bubble_box[1]))

            # 매칭 우선순위: (1) 텍스트 중심이 말풍선 안에 있는 후보 중 최근접,
            # (2) 밖이라면 말풍선 대각선 절반 이내 최근접만 허용.
            # 거리만으로 잡으면 OCR 필터로 비어버린 말풍선이 옆 말풍선의
            # 텍스트를 훔쳐가는 오매칭이 생긴다.
            closest_text_idx = -1
            best_rank = None
            for j, text_info in enumerate(page_text_elements):
                if j not in unmatched_text_indices:
                    continue
                text_box = text_info["element"].text_box
                text_center = ((text_box[0] + text_box[2]) / 2, (text_box[1] + text_box[3]) / 2)
                distance = float(np.hypot(bubble_center[0] - text_center[0], bubble_center[1] - text_center[1]))
                inside = (
                    bubble_box[0] <= text_center[0] <= bubble_box[2]
                    and bubble_box[1] <= text_center[1] <= bubble_box[3]
                )
                if not inside and distance > bubble_diag * 0.5:
                    continue
                rank = (0 if inside else 1, distance)
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    closest_text_idx = j

            matched_element = None
            if closest_text_idx != -1:
                matched_element = page_text_elements[closest_text_idx]["element"]
                unmatched_text_indices.remove(closest_text_idx)
            else:
                found_free_text_idx = -1
                for j in unmatched_free_text_indices:
                    free_text_element = page_free_texts[j]
                    if is_box_inside(free_text_element.text_box, bubble_box):
                        found_free_text_idx = j
                        break
                if found_free_text_idx != -1:
                    matched_element = page_free_texts[found_free_text_idx]
                    unmatched_free_text_indices.remove(found_free_text_idx)

            if matched_element:
                b = bubble_box
                cropped_bubble_rgb = page_data.image_rgb[b[1]:b[3], b[0]:b[2]]
                attachment = detect_bubble_attachment(
                    cropped_bubble_rgb,
                    edge_ratio=config.BUBBLE_ATTACHMENT_EDGE_RATIO,
                    min_length_ratio=config.BUBBLE_ATTACHMENT_MIN_LENGTH_RATIO,
                )
                speech_bubble = SpeechBubble(
                    bubble_box=b.tolist(),
                    text_element=matched_element,
                    attachment=attachment,
                )
                page_data.speech_bubbles.append(speech_bubble)

        remaining_free_texts = [page_free_texts[i] for i in sorted(unmatched_free_text_indices)]
        for free_text in remaining_free_texts:
            free_text.attachment = detect_freeform_attachment(
                page_data.image_rgb,
                free_text.text_box,
                search_px=config.FREEFORM_ATTACHMENT_SEARCH_PX,
                min_length_ratio=config.FREEFORM_ATTACHMENT_MIN_LENGTH_RATIO,
            )
        new_sizes = harmonize_freeform_sizes(
            [(ft.font_style, ft.font_size) for ft in remaining_free_texts]
        )
        for free_text, new_size in zip(remaining_free_texts, new_sizes):
            if new_size != free_text.font_size:
                free_text.font_size = new_size
        page_data.freeform_texts = remaining_free_texts
        batch_page_data.append(page_data)

    return batch_page_data


def harmonize_freeform_sizes(style_size_pairs):
    """같은 페이지·같은 스타일의 프리텍스트 크기를 다수 클러스터에 맞춰 통일합니다.

    원본 만화의 모놀로그 컬럼들은 한 페이지에서 같은 크기로 식자되는데,
    크롭별 측정 잡음(±10%)과 드문 측정 폭주(흰 테두리 글자 등) 때문에
    번역본이 들쭉날쭉해진다.

    규칙 (스타일별, 4개 이상일 때만):
    - 크기를 정렬해 22% 간격 이내로 이어지는 클러스터를 만들고,
      최대 클러스터가 전체의 60% 이상이면 그 중앙값을 기준으로:
      · ±25% 안의 잔잡음 → 기준값으로 스냅
      · 기준의 1.5배 초과 → 측정 폭주로 보고 기준값으로 클램프
      · 기준의 0.75배 미만 → 의도된 작은 글씨(주석 등)로 보고 유지
    Returns: 입력 순서대로의 새 크기 리스트.
    """
    new_sizes = [size for _, size in style_size_pairs]
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
            if abs(size - center) <= center * 0.25 or size > center * 1.5:
                new_sizes[idx] = center

    return new_sizes
