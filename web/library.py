"""책장 — 번역한(또는 번역할) 책 목록과 책마다의 상태.

책 하나 = 원고 폴더 + 결과 폴더. 결과 폴더는 원고 폴더 옆 '<원고 폴더 이름>_번역'으로 정한다.
목록은 저장소 폴더의 library.json에 두고(설정이 아니라 기록), 상태는 매번 결과 폴더의
checkpoint_meta.json·translation_data.json과 원고 폴더의 그림 파일로 다시 계산한다.
"""
import hashlib
import json
import os
import threading
import time
from typing import Dict, List, Optional

from src.data_models import TranslationStatus
from src.serialization import load_page_data_json
from src.utils import list_page_images, replace_file

LIBRARY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "library.json")
OUTPUT_SUFFIX = "_번역"
_lock = threading.Lock()


def _normalize(path: str) -> str:
    return os.path.normpath(os.path.abspath(path.strip().strip('"')))


def book_id_for(output_dir: str) -> str:
    return hashlib.sha1(_normalize(output_dir).lower().encode("utf-8")).hexdigest()[:12]


def output_dir_for(input_dir: str) -> str:
    """원고 폴더 옆 '<이름>_번역' 폴더."""
    folder = _normalize(input_dir)
    return os.path.join(os.path.dirname(folder), os.path.basename(folder) + OUTPUT_SUFFIX)


def _read() -> List[dict]:
    try:
        with open(LIBRARY_PATH, encoding="utf-8") as f:
            books = json.load(f)
        return books if isinstance(books, list) else []
    except (OSError, ValueError):
        return []


def _write(books: List[dict]):
    tmp = LIBRARY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=1)
    replace_file(tmp, LIBRARY_PATH)


def list_books() -> List[dict]:
    """최근에 연 책부터."""
    with _lock:
        books = _read()
    return sorted(books, key=lambda b: b.get("opened_at", 0), reverse=True)


def get_book(book_id: str) -> Optional[dict]:
    return next((b for b in list_books() if b["id"] == book_id), None)


def add_book(input_dir: str) -> dict:
    """원고 폴더로 책을 만든다(이미 있으면 그 책). 결과 폴더는 자동으로 정한다."""
    input_dir = _normalize(input_dir)
    output_dir = output_dir_for(input_dir)
    book_id = book_id_for(output_dir)
    with _lock:
        books = _read()
        book = next((b for b in books if b["id"] == book_id), None)
        if book is None:
            book = {"id": book_id, "title": os.path.basename(input_dir), "input_dir": input_dir,
                    "output_dir": output_dir, "added_at": time.time()}
            books.append(book)
        book["input_dir"] = input_dir
        book["opened_at"] = time.time()
        _write(books)
    return book


def touch(book_id: str):
    with _lock:
        books = _read()
        for book in books:
            if book["id"] == book_id:
                book["opened_at"] = time.time()
        _write(books)


def remove_book(book_id: str):
    """책장에서만 뺀다 — 원고·결과 폴더의 파일은 그대로 둔다."""
    with _lock:
        _write([b for b in _read() if b["id"] != book_id])


def _meta(output_dir: str) -> dict:
    try:
        with open(os.path.join(output_dir, "checkpoint_meta.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def book_state(book: dict) -> dict:
    """책의 지금 상태 — 쪽 목록과 번역·식자 진행, 번역 못 한 대사 수."""
    input_dir, output_dir = book["input_dir"], book["output_dir"]
    page_paths = list_page_images(input_dir)
    names = [os.path.basename(p) for p in page_paths]
    meta = _meta(output_dir)
    typeset = set(meta.get("pass2_completed_pages") or [])
    json_path = os.path.join(output_dir, "translation_data.json")
    translated_pages, failed_lines, pending_lines, lines = set(), 0, 0, 0
    if os.path.exists(json_path):
        try:
            for page in load_page_data_json(json_path):
                elements = page.text_elements()
                lines += len(elements)
                failed_lines += sum(e.translation_status == TranslationStatus.FAILED for e in elements)
                pending_lines += sum(e.translation_status == TranslationStatus.PENDING for e in elements)
                if page.is_translated:
                    translated_pages.add(page.source_page)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    pages = []
    for name in names:
        done = name in typeset and os.path.exists(os.path.join(output_dir, name))
        pages.append({"name": name, "typeset": done, "translated": name in translated_pages,
                      "version": int(os.path.getmtime(os.path.join(output_dir, name))) if done else 0})
    typeset_count = sum(p["typeset"] for p in pages)
    if not os.path.isdir(input_dir):
        status = "missing"
    elif not meta and not os.path.exists(json_path):
        status = "new"
    elif names and typeset_count == len(names):
        status = "done"
    elif failed_lines or pending_lines:
        status = "needs_retry"
    else:
        status = "partial"
    return {
        **book,
        "status": status,
        "page_count": len(names),
        "typeset_count": typeset_count,
        "translated_count": len(translated_pages),
        "line_count": lines,
        "failed_lines": failed_lines + pending_lines,
        "pages": pages,
        "cover": next((p["name"] for p in pages if p["typeset"]), names[0] if names else None),
    }


def summary(book: dict) -> Dict:
    """책장 목록용 — 쪽 목록 없이."""
    state = book_state(book)
    state.pop("pages", None)
    return state
