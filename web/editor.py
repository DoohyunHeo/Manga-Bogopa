"""번역 고치기 — 쪽 하나의 대사를 그림 위 자리와 함께 내주고, 고친 번역을 대사집(translation_data.json)에 저장한다.

대사 번호는 쪽 안의 자리로 고정된다: 말풍선은 b<순서>, 말풍선 밖 글자는 f<순서> (대사집의 목록 순서).
- 번역을 새로 적으면 '번역됨' — 효과음처럼 건너뛴 글자도 이렇게 번역할 수 있다
- 번역을 비우면 '번역 제외' — 원문을 지우지 않고 그대로 둔다
저장한 뒤에는 그 쪽만 '식자만 다시'로 다시 쓴다 (web/jobs.py → pipeline.only_pages).
"""
import os
from typing import Dict, Optional

from PIL import Image

from src.data_models import TranslationStatus
from src.serialization import load_page_data_json, save_page_data_json
from src.text_filters import dialogue_size, exclusion_reason


class Conflict(Exception):
    """불러온 뒤 대사집이 바뀌었다 — 덮어쓰지 않는다."""


_SKIP_NOTES = {
    "효과음": "효과음이라 원문을 그대로 둔 글자예요",
    "부속물": "잡지 부속물(쪽 번호·잡지 이름 등)이라 원문을 그대로 둔 글자예요",
}


def _json_path(output_dir: str) -> str:
    return os.path.join(output_dir, "translation_data.json")


def version(output_dir: str) -> str:
    """대사집이 바뀌었는지 알아보는 값 (파일 수정 시각, 나노초). 화면의 숫자로는 자릿수가 모자라 글자로 주고받는다."""
    path = _json_path(output_dir)
    return str(os.stat(path).st_mtime_ns) if os.path.exists(path) else "0"


def _places(page):
    for index, bubble in enumerate(page.speech_bubbles):
        yield f"b{index}", "bubble", bubble.bubble_box, bubble.text_element
    for index, element in enumerate(page.freeform_texts):
        yield f"f{index}", "freeform", element.text_box, element


def _note(element, in_bubble: bool, reference: Optional[float] = None) -> Optional[str]:
    status = element.translation_status
    if status == TranslationStatus.FAILED or status == TranslationStatus.PENDING:
        return "번역을 받지 못한 대사예요. 직접 적거나 ‘이어서 번역’으로 다시 요청할 수 있어요"
    if status != TranslationStatus.EXCLUDED:
        return None
    if not (element.original_text or "").strip():
        return "글자를 읽지 못해 원문을 그대로 둔 자리예요"
    ratio = element.font_size / reference if reference and element.font_size else None
    return _SKIP_NOTES.get(exclusion_reason(element.original_text, in_bubble, ratio), "번역하지 않고 원문을 둔 자리예요")


def page_lines(book: dict, name: str) -> Optional[dict]:
    """쪽 하나의 대사 목록 (그림 크기·자리·원문·번역·상태). 대사집이 없으면 None."""
    path = _json_path(book["output_dir"])
    if not os.path.exists(path):
        return None
    page = next((p for p in load_page_data_json(path) if p.source_page == name), None)
    with Image.open(os.path.join(book["input_dir"], name)) as image:
        size = list(image.size)
    lines = []
    if page is not None:
        reference = dialogue_size(page.speech_bubbles)
        for key, kind, box, element in _places(page):
            translated = element.translation_status == TranslationStatus.TRANSLATED
            lines.append({
                "id": key,
                "kind": kind,
                "box": [int(v) for v in box],
                "original": element.original_text or "",
                "translation": (element.translated_text or "") if translated else "",
                "status": element.translation_status.value,
                "note": _note(element, kind == "bubble", reference),
            })
    return {"page": name, "size": size, "version": version(book["output_dir"]),
            "has_page": page is not None, "lines": lines}


def save_page_lines(book: dict, name: str, edits: Dict[str, str], expected_version: str) -> int:
    """고친 번역을 저장하고 바꾼 대사 수를 돌려준다. 불러온 뒤 대사집이 바뀌었으면 Conflict."""
    path = _json_path(book["output_dir"])
    if not os.path.exists(path):
        raise KeyError("대사집이 없어요")
    if str(expected_version or "") != version(book["output_dir"]):
        raise Conflict()
    pages = load_page_data_json(path)
    page = next((p for p in pages if p.source_page == name), None)
    if page is None:
        raise KeyError("대사집에 없는 쪽이에요")
    elements = {key: element for key, _, _, element in _places(page)}
    changed = 0
    for key, text in edits.items():
        element = elements.get(key)
        if element is None:
            raise KeyError(f"없는 대사예요: {key}")
        text = str(text or "").strip()
        current = (element.translated_text or "").strip() \
            if element.translation_status == TranslationStatus.TRANSLATED else ""
        if text == current:
            continue
        element.translated_text = text or None
        element.translation_status = TranslationStatus.TRANSLATED if text else TranslationStatus.EXCLUDED
        changed += 1
    if changed:
        save_page_data_json(pages, path)
    return changed
