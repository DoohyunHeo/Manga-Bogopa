import logging
import os
from collections import defaultdict, deque
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

from src import config
from src.data_models import Attachment, PageData, SpeechBubble, TextElement
from src.line_detector import detect_bubble_attachment, detect_freeform_attachment
from src.utils import Letterbox, calculate_iou, is_box_inside, merge_boxes

logger = logging.getLogger(__name__)

# Internal model/heuristic constants (not user-tunable).
_FALLBACK_FONT_SIZE = 20
_FONT_MODEL_INPUT_SIZE = (224, 224)
_CHAR_RATIO_FONT_SIZE_GAIN = 1.18
_STROKE_RATIO_FONT_SIZE_GAIN = 7.5
_STYLE_FALLBACK_MAX_ANGLE = 15.0
_STYLE_SPECIAL_MIN_CONFIDENCE = {
    "standard": 0.0,
    "pop": 0.46,
    "shouting": 0.42,
    "handwriting": 0.40,
    "angry": 0.38,
    "cute": 0.38,
    "scared": 0.38,
    "embarrassment": 0.36,
    "narration": 0.34,
}


def _cuda_autocast_context():
    if config.DEVICE != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def detect_objects(detection_model, batch_images_rgb):
    """Run object detection on a batch of pages."""
    logger.info(f"Detecting objects for {len(batch_images_rgb)} pages...")
    batch_results = detection_model(batch_images_rgb, conf=config.YOLO_CONF_THRESHOLD, verbose=False)

    all_text_items = []
    all_bubbles_by_page = [[] for _ in batch_images_rgb]
    for page_idx, results in enumerate(tqdm(batch_results, desc="Detection")):
        for box in results.boxes:
            class_name = results.names[int(box.cls[0])]
            coords = box.xyxy[0].cpu().numpy()
            if class_name == "bubble":
                all_bubbles_by_page[page_idx].append(coords.astype(int))
            elif class_name in ["text", "free_text"]:
                all_text_items.append({
                    "page_idx": page_idx,
                    "box": coords,
                    "class_name": class_name,
                })
    return all_text_items, all_bubbles_by_page


def merge_text_boxes(text_items):
    """Merge overlapping text boxes."""
    logger.info(f"Merging overlapping boxes from {len(text_items)} detected text objects...")
    grouped_items = defaultdict(list)
    for item in text_items:
        grouped_items[(item["page_idx"], item["class_name"])].append(item)

    final_items = []
    for (page_idx, class_name), items in grouped_items.items():
        if len(items) < 2:
            final_items.extend(items)
            continue

        num_items = len(items)
        adj_matrix = np.zeros((num_items, num_items))
        for i in range(num_items):
            for j in range(i + 1, num_items):
                iou = calculate_iou(items[i]["box"], items[j]["box"])
                if iou > config.TEXT_MERGE_OVERLAP_THRESHOLD:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1

        visited = [False] * num_items
        for i in range(num_items):
            if visited[i]:
                continue

            component = []
            q = deque([i])
            visited[i] = True
            while q:
                u = q.popleft()
                component.append(u)
                for v in range(num_items):
                    if adj_matrix[u, v] and not visited[v]:
                        visited[v] = True
                        q.append(v)

            if len(component) > 1:
                cluster_items = [items[k] for k in component]
                merged_box = merge_boxes([item["box"] for item in cluster_items])
                final_items.append({"page_idx": page_idx, "box": merged_box, "class_name": class_name})
            else:
                final_items.append(items[component[0]])

    logger.info(f"Box merge reduced the set to {len(final_items)} objects.")
    return final_items


def _prepare_crops(text_items, batch_images_rgb, batch_paths):
    """Prepare crop images for OCR and font analysis."""
    crops_for_ocr = []
    for item in text_items:
        image_rgb = batch_images_rgb[item["page_idx"]]
        coords = item["box"].astype(int)
        original_crop_pil = Image.fromarray(image_rgb[coords[1]:coords[3], coords[0]:coords[2]])
        item["crop"] = original_crop_pil
        crops_for_ocr.append(original_crop_pil)

    return crops_for_ocr


def _build_font_model_tta_views(crop: Image.Image):
    base = crop.convert("RGB")
    if not getattr(config, "FONT_MODEL_TTA_ENABLED", True):
        return [base]

    variants = max(1, int(getattr(config, "FONT_MODEL_TTA_VARIANTS", 3)))
    views = [base]

    if variants >= 2:
        contrast_view = ImageOps.autocontrast(base)
        contrast_view = ImageEnhance.Contrast(contrast_view).enhance(1.08)
        views.append(contrast_view)

    if variants >= 3:
        sharp_view = ImageEnhance.Sharpness(base).enhance(1.15)
        views.append(sharp_view)

    return views[:variants]


def _resolve_style_name(font_model, predicted_index):
    style_mapping = getattr(font_model, "style_mapping", {}) or {}
    if not style_mapping:
        return "standard"

    index_int = int(predicted_index)
    if index_int in style_mapping:
        return style_mapping[index_int]

    index_str = str(index_int)
    if index_str in style_mapping:
        return style_mapping[index_str]

    return "standard"


def _choose_style_name(font_model, style_probs_row, angle_deg, expressive_logits_row=None):
    style_probs_cpu = style_probs_row.detach().cpu()
    top_k = min(2, int(style_probs_cpu.shape[0]))
    top_values, top_indices = torch.topk(style_probs_cpu, k=top_k)
    top_confidence = float(top_values[0])
    second_confidence = float(top_values[1]) if top_k > 1 else 0.0
    confidence_margin = top_confidence - second_confidence

    predicted_style = _resolve_style_name(font_model, int(top_indices[0]))
    expressive_confidence = None
    if expressive_logits_row is not None and getattr(font_model, "has_expressive_head", False):
        expressive_confidence = float(torch.sigmoid(expressive_logits_row.detach().cpu()))

    if not getattr(config, "FONT_STYLE_FALLBACK_ENABLED", True):
        return predicted_style, top_confidence, expressive_confidence
    if predicted_style == "standard":
        return "standard", top_confidence, expressive_confidence

    low_confidence_threshold = float(getattr(config, "FONT_STYLE_LOW_CONFIDENCE_THRESHOLD", 0.24))
    low_margin_threshold = float(getattr(config, "FONT_STYLE_LOW_MARGIN_THRESHOLD", 0.04))
    fallback_max_angle = _STYLE_FALLBACK_MAX_ANGLE
    expressive_prob_threshold = float(getattr(config, "FONT_STYLE_EXPRESSIVE_PROB_THRESHOLD", 0.55))
    style_specific_thresholds = _STYLE_SPECIAL_MIN_CONFIDENCE
    style_specific_threshold = float(style_specific_thresholds.get(predicted_style, low_confidence_threshold))

    if expressive_confidence is not None and expressive_confidence < expressive_prob_threshold:
        return "standard", top_confidence, expressive_confidence
    if top_confidence < low_confidence_threshold:
        return "standard", top_confidence, expressive_confidence
    if top_confidence < style_specific_threshold:
        return "standard", top_confidence, expressive_confidence
    if confidence_margin < low_margin_threshold and abs(float(angle_deg)) <= fallback_max_angle:
        return "standard", top_confidence, expressive_confidence

    return predicted_style, top_confidence, expressive_confidence


def _resolve_font_size(font_model, raw_size_value, item, page_height):
    size_reference = getattr(font_model, "size_target_reference", "page_height")
    crop_height = (
        item["crop"].height
        if "crop" in item
        else int(item["box"][3] - item["box"][1])
    )
    reference_height = max(crop_height if size_reference == "crop_height" else page_height, 1)

    raw_size_value = float(raw_size_value)
    if size_reference == "absolute_px" or (size_reference == "page_height" and raw_size_value > 4.0):
        return max(1, int(round(raw_size_value)))
    return max(1, int(round(raw_size_value * reference_height)))


def _resolve_char_ratio(font_model, raw_size_value, item, page_height):
    size_reference = getattr(font_model, "size_target_reference", "crop_height")
    crop_height = (
        item["crop"].height
        if "crop" in item
        else int(item["box"][3] - item["box"][1])
    )
    reference_height = max(crop_height if size_reference == "crop_height" else page_height, 1)
    raw_size_value = float(raw_size_value)

    if size_reference == "absolute_px":
        return raw_size_value / reference_height
    return raw_size_value


def _predict_font_properties(font_appearance_model, font_size_model, legacy_font_model, text_items, page_heights=None):
    """Predict font style, angle, and size for each crop."""
    style_angle_model = font_appearance_model or legacy_font_model
    size_model = font_size_model or legacy_font_model
    if not style_angle_model and not size_model:
        return [{
            "font_size": _FALLBACK_FONT_SIZE,
            "angle": 0,
            "font_style": "standard",
            "font_char_ratio": None,
            "font_stroke_ratio": None,
            "font_style_confidence": None,
            "expressive_confidence": None,
        }] * len(text_items)

    logger.info(
        "Analyzing %s text crops with font models: style_angle=%s, size=%s, legacy=%s",
        len(text_items),
        getattr(style_angle_model, "checkpoint_path", None),
        getattr(size_model, "checkpoint_path", None),
        getattr(legacy_font_model, "checkpoint_path", None),
    )
    transform = transforms.Compose([
        Letterbox(_FONT_MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_props = []
    font_batch_size = config.FONT_MODEL_BATCH_SIZE
    for i in tqdm(range(0, len(text_items), font_batch_size), desc="Font Model"):
        batch_items = text_items[i:i + font_batch_size]
        batch_views = [_build_font_model_tta_views(item["crop"]) for item in batch_items]
        num_views = len(batch_views[0]) if batch_views else 1

        with torch.inference_mode():
            style_prob_sum = None
            angle_sin_sum = None
            angle_cos_sum = None
            expressive_logit_sum = None
            size_value_sum = None

            for view_idx in range(num_views):
                image_tensors = torch.stack([
                    transform(views[view_idx]) for views in batch_views
                ]).to(config.DEVICE)

                style_outputs = None
                if style_angle_model is not None:
                    with _cuda_autocast_context():
                        style_outputs = style_angle_model(image_tensors)

                    style_probs = torch.softmax(style_outputs["style"], dim=1)
                    angle_radians = torch.deg2rad(style_outputs["angle"].float())
                    angle_sin = torch.sin(angle_radians)
                    angle_cos = torch.cos(angle_radians)
                    expressive_logits = style_outputs.get("expressive")

                    style_prob_sum = style_probs if style_prob_sum is None else style_prob_sum + style_probs
                    angle_sin_sum = angle_sin if angle_sin_sum is None else angle_sin_sum + angle_sin
                    angle_cos_sum = angle_cos if angle_cos_sum is None else angle_cos_sum + angle_cos
                    if expressive_logits is not None:
                        expressive_logit_sum = expressive_logits if expressive_logit_sum is None else expressive_logit_sum + expressive_logits

                if size_model is not None:
                    if size_model is style_angle_model:
                        size_outputs = style_outputs
                    else:
                        with _cuda_autocast_context():
                            size_outputs = size_model(image_tensors)

                    size_values = size_outputs["size"].float() if "size" in size_outputs else None
                    if size_values is not None:
                        size_value_sum = size_values if size_value_sum is None else size_value_sum + size_values

            pred_style_indices = None
            pred_angles = None
            if style_prob_sum is not None and angle_sin_sum is not None and angle_cos_sum is not None:
                avg_style_probs = style_prob_sum / num_views
                pred_style_indices = torch.argmax(avg_style_probs, dim=1).cpu().numpy()
                pred_angles = torch.rad2deg(
                    torch.atan2(angle_sin_sum / num_views, angle_cos_sum / num_views)
                ).cpu().numpy()
            pred_size_values = (size_value_sum / num_views).cpu().numpy() if size_value_sum is not None else None

        for j in range(len(batch_items)):
            item_page_height = (
                page_heights[batch_items[j]["page_idx"]]
                if page_heights and batch_items[j].get("page_idx") is not None
                else 1600
            )
            style_name = "standard"
            style_confidence = None
            expressive_confidence = None
            if pred_style_indices is not None:
                averaged_expressive = (expressive_logit_sum / num_views)[j] if expressive_logit_sum is not None else None
                style_name, style_confidence, expressive_confidence = _choose_style_name(
                    style_angle_model,
                    avg_style_probs[j],
                    pred_angles[j],
                    averaged_expressive,
                )

            font_char_ratio = None
            font_stroke_ratio = None
            if pred_size_values is not None and size_model is not None:
                size_target_kind = getattr(size_model, "size_target_kind", "legacy_ratio")
                if size_target_kind == "stroke_width_ratio":
                    font_stroke_ratio = _resolve_char_ratio(size_model, pred_size_values[j], batch_items[j], item_page_height)
                    heuristic_gain = _STROKE_RATIO_FONT_SIZE_GAIN
                    reference_height = max(batch_items[j]["crop"].height, 1)
                    font_size = max(1, int(round(font_stroke_ratio * reference_height * heuristic_gain)))
                elif size_target_kind in {"char_height_ratio", "glyph_height_ratio"}:
                    font_char_ratio = _resolve_char_ratio(size_model, pred_size_values[j], batch_items[j], item_page_height)
                    heuristic_gain = _CHAR_RATIO_FONT_SIZE_GAIN
                    reference_height = max(batch_items[j]["crop"].height, 1)
                    font_size = max(1, int(round(font_char_ratio * reference_height * heuristic_gain)))
                else:
                    font_size = _resolve_font_size(size_model, pred_size_values[j], batch_items[j], item_page_height)
            else:
                font_size = _FALLBACK_FONT_SIZE

            all_props.append({
                "font_size": font_size,
                "angle": int(round(pred_angles[j])) if pred_angles is not None else 0,
                "font_style": style_name,
                "font_char_ratio": font_char_ratio,
                "font_stroke_ratio": font_stroke_ratio,
                "font_style_confidence": style_confidence,
                "expressive_confidence": expressive_confidence,
            })

    return all_props


def _coerce_freeform_style(props, class_name):
    """For free_text items, force narration when the style isn't a high-confidence non-standard pick.

    Freeform text in manga is typically narration/SFX, so "standard" is treated
    as the narration font; any weak prediction falls back to narration too.
    """
    if class_name != "free_text":
        return props

    min_confidence = float(getattr(config, "FREEFORM_STYLE_MIN_CONFIDENCE", 0.70))
    confidence = props.get("font_style_confidence")
    style = props.get("font_style") or "standard"
    low_confidence = confidence is None or confidence < min_confidence

    if style == "standard" or low_confidence:
        adjusted = dict(props)
        adjusted["font_style"] = "narration"
        return adjusted
    return props


def extract_text_properties(models, batch_images_rgb, text_items, batch_paths):
    """Run OCR and font-property prediction for text boxes."""
    if not text_items:
        return []

    crops_for_ocr = _prepare_crops(text_items, batch_images_rgb, batch_paths)

    logger.info(f"Running batch OCR for {len(crops_for_ocr)} text crops...")
    all_ocr_results = models["ocr"](crops_for_ocr)

    page_heights = [img.shape[0] for img in batch_images_rgb] if batch_images_rgb else [1600]
    all_props = _predict_font_properties(
        models.get("font_appearance_classifier"),
        models.get("font_size_regressor"),
        models.get("font_classifier"),
        text_items,
        page_heights=page_heights,
    )

    processed_text_elements = []
    filtered_count = 0
    for i, item in enumerate(text_items):
        ocr_text = all_ocr_results[i]
        if not ocr_text:
            continue
        if not _is_valid_text(ocr_text, item["box"]):
            filtered_count += 1
            continue
        props = _coerce_freeform_style(all_props[i], item["class_name"])
        element = TextElement(
            text_box=item["box"].tolist(),
            original_text=ocr_text,
            **props,
        )
        processed_text_elements.append({
            "element": element,
            "page_idx": item["page_idx"],
            "class_name": item["class_name"],
        })

    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} low-quality OCR text items.")

    return processed_text_elements


def _is_valid_text(text, box):
    """Validate OCR output."""
    import re

    text = text.strip()
    if len(text) == 0:
        return False
    if re.fullmatch(r"[\d\s\.\-\+\*/=@#%&:;,!?\(\)\[\]{}<>\"\'`~^|\\/_]+", text):
        return False
    if len(text) == 1 and not _is_cjk_or_kana(text[0]):
        return False
    x1, y1, x2, y2 = box[:4]
    if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
        return False
    if not any(_is_cjk_or_kana(c) or c.isalpha() for c in text):
        return False
    return True


def _is_cjk_or_kana(char):
    cp = ord(char)
    return (
        (0x3040 <= cp <= 0x309F)
        or (0x30A0 <= cp <= 0x30FF)
        or (0x4E00 <= cp <= 0x9FFF)
        or (0xAC00 <= cp <= 0xD7A3)
        or (0x3400 <= cp <= 0x4DBF)
        or (0xFF00 <= cp <= 0xFFEF)
    )


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

            closest_text_idx = -1
            min_distance = float("inf")
            for j, text_info in enumerate(page_text_elements):
                if j not in unmatched_text_indices:
                    continue
                text_box = text_info["element"].text_box
                text_center = ((text_box[0] + text_box[2]) / 2, (text_box[1] + text_box[3]) / 2)
                distance = np.sqrt((bubble_center[0] - text_center[0]) ** 2 + (bubble_center[1] - text_center[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
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

        remaining_free_texts = [page_free_texts[i] for i in sorted(list(unmatched_free_text_indices))]
        for free_text in remaining_free_texts:
            free_text.attachment = detect_freeform_attachment(
                page_data.image_rgb,
                free_text.text_box,
                search_px=config.FREEFORM_ATTACHMENT_SEARCH_PX,
                min_length_ratio=config.FREEFORM_ATTACHMENT_MIN_LENGTH_RATIO,
            )
        page_data.freeform_texts = remaining_free_texts
        batch_page_data.append(page_data)

    return batch_page_data


def process_image_batch(models, batch_images_rgb, batch_paths):
    """Process a batch of pages into PageData objects."""
    all_text_items, all_bubbles_by_page = detect_objects(models["detection"], batch_images_rgb)
    merged_text_items = merge_text_boxes(all_text_items)
    processed_text_elements = extract_text_properties(models, batch_images_rgb, merged_text_items, batch_paths)
    batch_page_data = structure_page_data(batch_paths, batch_images_rgb, all_bubbles_by_page, processed_text_elements)
    return batch_page_data
