"""웹 화면 서버 — FastAPI. 빌드한 화면(web/dist)과 화면이 쓰는 API(/api)를 내준다.

이 컴퓨터(127.0.0.1)에서만 열고, 다른 사이트가 이 서버를 부르지 못하게 Host·Origin 머리글을 확인한다.
그림은 책장에 등록된 책의 원고·결과 폴더 안 쪽 그림만 내준다.
"""
import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import cv2
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src import config, translator
from src.utils import list_page_images, read_image_bgr
from web import editor, library
from web.jobs import jobs
from web.state import app_state

DIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
THUMB_DIR = os.path.join(tempfile.gettempdir(), "manga_bogopa_thumbs")
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

# 화면의 설정 이름 → config 필드. 저장되는 설정(config.USER_SETTINGS)은 빠짐없이 여기 있어야 한다
SETTING_FIELDS = {
    "translation_backend": "TRANSLATION_BACKEND",
    "translation_models": "TRANSLATION_MODELS",
    "translation_efforts": "TRANSLATION_EFFORTS",
    "sessions": "AGY_SESSIONS",
    "agy_path": "AGY_PATH",
    "fonts": "FONT_MAP",
    "vertical_text": "ENABLE_VERTICAL_TEXT",
    "low_vram": "LOW_VRAM_MODE",
    "narration_as_standard": "NARRATION_AS_STANDARD",
    "fit_translation": "ENABLE_FIT_TRANSLATION",
    "slant_exclamations": "SLANT_EXCLAMATIONS",
}
if set(SETTING_FIELDS.values()) != config.USER_SETTINGS:
    raise RuntimeError(f"화면에 없는 저장 설정이 있습니다: {sorted(config.USER_SETTINGS - set(SETTING_FIELDS.values()))}")

# 글꼴 칸(FONT_MAP 키)과 화면에 보일 이름·미리보기 문구 — 칸은 원문 글씨의 생김새(모양 판정, src/font_relative.py)를
# 따른다. 굵은 대사(shouting)는 고르지 않는다 — 평범한 대사 글꼴의 같은 가족 굵은 글꼴로 정해지고
# (config.apply_font_modes) 설정 응답의 bold_font로 읽기만 한다
FONT_STYLES = [
    ("standard", "평범한 대사", "오늘도 잘 부탁해"),
    ("narration", "해설·내레이션", "그날, 모든 것이 바뀌었다."), ("handwriting", "손글씨", "몰래 먹어야지"),
    ("pop", "통통 튀는 강조", "짜잔!"),
    ("angry", "각진 글씨", "획이 모난 굵은 글씨"), ("scared", "호러 글씨", "거칠게 번진 으스스한 글씨"),
]

# 번역에 쓸 AI(번역 CLI)와 생각 깊이의 화면 이름
TRANSLATION_CHOICE_LABELS = {"auto": "자동 (설치된 것 중에서)", **translator.BACKEND_LABELS}
EFFORT_LABELS = {"low": "빠르게", "medium": "보통", "high": "꼼꼼하게", "xhigh": "더 꼼꼼하게",
                 "max": "아주 꼼꼼하게", "ultra": "가장 꼼꼼하게"}

def _warm_translation_options():
    for backend, path in translator.detect_backends().items():
        if path:
            translator.backend_options(backend, path)


@asynccontextmanager
async def _lifespan(_app):
    app_state.initialize_pipeline()  # 모델은 번역을 시작할 때 읽는다
    # 설정 화면이 바로 뜨도록 번역 CLI마다 고를 수 있는 모델 목록을 뒤에서 미리 읽어 둔다 (CLI마다 몇 초)
    threading.Thread(target=_warm_translation_options, daemon=True).start()
    yield


app = FastAPI(title="Manga-Bogopa", docs_url=None, redoc_url=None, openapi_url=None, lifespan=_lifespan)


@app.middleware("http")
async def only_this_computer(request: Request, call_next):
    host = urlparse("//" + (request.headers.get("host") or "")).hostname or ""
    origin = request.headers.get("origin")
    if host not in _LOCAL_HOSTS or (origin and urlparse(origin).hostname not in _LOCAL_HOSTS):
        return PlainTextResponse("이 컴퓨터에서만 열 수 있어요.", status_code=403)
    return await call_next(request)


def _error(status: int, message: str):
    raise HTTPException(status_code=status, detail=message)


def _book_or_404(book_id: str) -> dict:
    book = library.get_book(book_id)
    if not book:
        _error(404, "책장에 없는 책이에요.")
    return book


# ── 시스템 ──

@app.get("/api/system")
def system():
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 — 그래픽카드 이름은 곁들이는 정보
        pass
    path = translator.find_agy_executable()
    active = translator.resolve_backend()
    return {"agy_found": path is not None, "agy_path": path,
            # 설정에서 고른(자동이면 설치된 것 중 고른) 번역 CLI — 번역 연결 여부는 이것으로 본다
            "translation_found": active is not None,
            "translation_backend": active[0] if active else None,
            "translation_label": translator.BACKEND_LABELS[active[0]] if active else None,
            "translation_path": active[1] if active else None,
            "gpu": gpu, "busy": jobs.is_running()}


def _missing_backend_message() -> str:
    choice = config._config.TRANSLATION_BACKEND
    if choice in translator.BACKEND_LABELS:
        return f"설정에서 고른 {translator.BACKEND_LABELS[choice]}를 찾지 못했어요. 설정에서 연결을 확인해 주세요."
    return "번역을 맡을 AI(Antigravity CLI·Claude Code·Codex CLI)를 찾지 못했어요. 설정에서 연결을 확인해 주세요."


@app.post("/api/system/login-check")
def login_check(body: dict = Body(default={})):
    """번역 CLI 로그인 확인 — backend를 주면 그 CLI, 없으면 지금 쓰는 CLI (번역 요청은 보내지 않는다)."""
    backend = (body or {}).get("backend")
    if backend is not None and backend not in config.TRANSLATION_BACKENDS:
        _error(400, "번역에 쓸 AI 값이 올바르지 않아요.")
    if backend is None:
        active = translator.resolve_backend()
        if not active:
            return {"ok": False, "message": _missing_backend_message(), "backend": None}
        backend = active[0]
    ok, message = translator.check_login(backend)
    return {"ok": ok, "message": message if not ok else "로그인되어 있어요.", "backend": backend}


@app.post("/api/dialog/folder")
def pick_folder(body: dict = Body(default={})):
    """이 컴퓨터의 폴더 고르기 창 — 고른 경로(취소하면 null)."""
    initial = str(body.get("initial") or "")
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
    except Exception:  # noqa: BLE001 — 화면이 없는 환경
        _error(501, "폴더 고르기 창을 열 수 없어요. 경로를 직접 적어 주세요.")
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="원고 폴더 고르기",
                                         initialdir=initial if os.path.isdir(initial) else os.path.expanduser("~"))
    finally:
        root.destroy()
    return {"path": os.path.normpath(folder) if folder else None}


# ── 설정 ──

def _font_files():
    folder = config._config.FONT_DIR
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith((".ttf", ".otf", ".ttc")))


def _bold_emboldened():
    """굵은 대사에 합성 굵게를 더하는지 (그릴 때와 같은 판단). 글꼴을 못 읽으면 False."""
    try:
        from src.text_layout import needs_synthetic_bold
        return bool(needs_synthetic_bold())
    except Exception:  # noqa: BLE001 — 설정 화면은 글꼴 판단 실패로 막지 않는다
        return False


def _horizontal_scale(style):
    """말풍선 안 보통 크기 대사를 그 글씨체로 그릴 때의 장평 (미리보기가 따른다). 못 읽으면 1.0."""
    try:
        from src.text_renderer import DEFAULT_STYLES
        return float(DEFAULT_STYLES.get(style, DEFAULT_STYLES["standard"]).horizontal_scale)
    except Exception:  # noqa: BLE001 — 설정 화면은 미리보기 값 때문에 막지 않는다
        return 1.0


def _active_backend(found: dict) -> "str | None":
    """지금 쓰는 번역 CLI — 고른 CLI, 자동이면 설치된 것 중 첫째 (못 찾으면 None)."""
    choice = config._config.TRANSLATION_BACKEND
    return next((b for b in (config.TRANSLATION_BACKENDS if choice == "auto" else (choice,)) if found.get(b)), None)


def _allowed_efforts(backend: str, model: str) -> list:
    """그 모델에서 고를 수 있는 생각 깊이 (이름에 수준이 붙은 모델은 CLI 전체 값 — 저장만 하고 쓰지 않는다)."""
    options = translator.backend_options(backend)
    entry = next((m for m in options["models"] if m["id"] == model), None)
    return (entry or {}).get("efforts") or options["efforts"]


def _translation_backends(found: dict) -> list:
    """번역 CLI마다 설치 여부·위치와 고를 수 있는 모델·생각 깊이 (읽기 전용, 자동이 고르는 순서)."""
    with ThreadPoolExecutor(max_workers=len(found)) as pool:  # 모델 목록 명령이 CLI마다 몇 초씩 걸린다
        options = dict(zip(found, pool.map(lambda b: translator.backend_options(b, found[b]), found)))
    return [{"id": backend, "label": translator.BACKEND_LABELS[backend], "found": path is not None, "path": path,
             "models": options[backend]["models"],
             "efforts": [{"id": e, "label": EFFORT_LABELS.get(e, e)} for e in options[backend]["efforts"]]}
            for backend, path in found.items()]


def _per_backend(value, what: str):
    if not isinstance(value, dict) or not set(value) <= set(config.TRANSLATION_BACKENDS):
        _error(400, f"{what} 값의 모양이 올바르지 않아요.")
    return value.items()


def _settings():
    c = config._config
    found = translator.detect_backends()
    active = _active_backend(found)
    return {
        # 번역에 쓸 AI: 고른 값(auto·antigravity·claude·codex), 지금 쓰는 CLI, CLI마다 고른 모델·생각 깊이
        "translation_backend": c.TRANSLATION_BACKEND,
        "translation_choices": [{"id": k, "label": v} for k, v in TRANSLATION_CHOICE_LABELS.items()],
        "translation_active": active,
        "translation_models": dict(c.TRANSLATION_MODELS),
        "translation_efforts": dict(c.TRANSLATION_EFFORTS),
        "translation_effort_default": config.DEFAULT_TRANSLATION_EFFORT,  # 읽기 전용 — 화면의 권장 표시가 따른다
        "translation_backends": _translation_backends(found),
        "sessions": int(c.AGY_SESSIONS),
        "sessions_range": list(config.AGY_SESSIONS_RANGE),
        "sessions_default": config.PipelineConfig.AGY_SESSIONS,
        "agy_path": c.AGY_PATH,
        "vertical_text": bool(c.ENABLE_VERTICAL_TEXT),
        "low_vram": bool(c.LOW_VRAM_MODE),
        "narration_as_standard": bool(c.NARRATION_AS_STANDARD),
        "fit_translation": bool(c.ENABLE_FIT_TRANSLATION),
        "slant_exclamations": bool(c.SLANT_EXCLAMATIONS),
        # horizontal_scale(읽기 전용)은 그 글씨체의 장평 — 미리보기가 따른다
        "fonts": [{"style": style, "label": label, "sample": sample,
                   "file": os.path.basename(c.FONT_MAP.get(style, "")),
                   "horizontal_scale": _horizontal_scale(style)} for style, label, sample in FONT_STYLES],
        # 굵은 대사 글꼴 (읽기 전용): 글꼴 폴더의 파일 이름·굵기, 같은 가족 굵은 파일이 없어 합성 굵게로 그리는지,
        # 굵은 파일에 합성 굵게를 더하는지(파일이 보통 글꼴보다 획이 1.25배 굵지 않을 때 — text_layout.needs_synthetic_bold),
        # 굵은 대사의 장평
        "bold_font": {"file": os.path.basename(c.BOLD_FONT.source) if c.BOLD_FONT else "",
                      "weight": c.BOLD_FONT.weight if c.BOLD_FONT else 0,
                      "synthetic": bool(c.BOLD_FONT.synthetic) if c.BOLD_FONT else True,
                      "emboldened": _bold_emboldened(),
                      "horizontal_scale": _horizontal_scale("shouting")},
        "font_files": _font_files(),
        "locked": jobs.is_running(),
    }


@app.get("/api/settings")
def get_settings():
    return _settings()


_SWITCHES = {  # 화면의 켜고 끄는 설정 → config 필드
    "vertical_text": "ENABLE_VERTICAL_TEXT", "low_vram": "LOW_VRAM_MODE", "narration_as_standard": "NARRATION_AS_STANDARD",
    "fit_translation": "ENABLE_FIT_TRANSLATION", "slant_exclamations": "SLANT_EXCLAMATIONS",
}


@app.put("/api/settings")
def put_settings(body: dict = Body(...)):
    if jobs.is_running():
        _error(409, "번역하는 동안에는 설정을 바꿀 수 없어요. 끝나거나 멈춘 뒤에 바꿔 주세요.")
    c = config._config
    before = copy.deepcopy(c.to_dict())  # 하나라도 틀리면 전부 되돌린다 — 일부만 바뀐 채 거절되지 않게
    try:
        _apply_settings(c, body)
    except Exception:
        c.update_from_dict(before)
        c.apply_font_modes()
        raise
    c.apply_font_modes()
    config.save()
    return _settings()


def _apply_settings(c, body: dict):
    if "translation_backend" in body:
        if body["translation_backend"] not in config.TRANSLATION_BACKEND_CHOICES:
            _error(400, "번역에 쓸 AI 값이 올바르지 않아요.")
        c.TRANSLATION_BACKEND = body["translation_backend"]
    if "translation_models" in body:
        for backend, model in _per_backend(body["translation_models"], "모델"):
            model = str(model or "").strip()
            known = {m["id"] for m in translator.backend_options(backend)["models"]}
            if model != c.TRANSLATION_MODELS.get(backend) and model not in known:
                _error(400, f"{translator.BACKEND_LABELS[backend]}에서 고를 수 있는 모델이 아니에요.")
            c.TRANSLATION_MODELS[backend] = model
            allowed = _allowed_efforts(backend, model)
            if c.TRANSLATION_EFFORTS.get(backend) not in allowed:  # 새 모델이 받지 않는 생각 깊이는 기본으로
                default = config.DEFAULT_TRANSLATION_EFFORT
                c.TRANSLATION_EFFORTS[backend] = default if default in allowed else allowed[0]
    if "translation_efforts" in body:
        for backend, effort in _per_backend(body["translation_efforts"], "생각 깊이"):
            if effort not in _allowed_efforts(backend, c.TRANSLATION_MODELS.get(backend, "")):
                _error(400, "이 모델에서 고를 수 없는 생각 깊이예요.")
            c.TRANSLATION_EFFORTS[backend] = effort
    if "sessions" in body:
        low, high = config.AGY_SESSIONS_RANGE
        value = body["sessions"]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            _error(400, f"번역 동시 진행은 {low}~{high}묶음 가운데서 골라 주세요.")
        c.AGY_SESSIONS = value
    if "agy_path" in body:
        c.AGY_PATH = str(body["agy_path"] or "").strip().strip('"')
    for key, field_name in _SWITCHES.items():
        if key in body:
            if not isinstance(body[key], bool):
                _error(400, "켜고 끄는 설정 값이 올바르지 않아요.")
            setattr(c, field_name, body[key])
    if "fonts" in body:
        files = set(_font_files())
        styles = {style for style, _, _ in FONT_STYLES}
        for style, filename in (body["fonts"] or {}).items():
            if style == "shouting":
                continue  # 굵은 대사 글꼴은 평범한 대사 글꼴을 따라 정해진다
            if style not in styles or filename not in files:
                _error(400, f"글꼴 파일을 찾지 못했어요: {filename}")
            c.FONT_MAP[style] = os.path.join(c.FONT_DIR, filename)


@app.get("/api/fonts/{filename}")
def font_file(filename: str):
    if filename not in _font_files():
        _error(404, "글꼴 파일이 없어요.")
    return FileResponse(os.path.join(config._config.FONT_DIR, filename),
                        headers={"Cache-Control": "max-age=86400"})


# ── 책장 ──

@app.get("/api/books")
def books():
    running = jobs.snapshot()
    return {"books": [library.summary(book) for book in library.list_books()],
            "running_book": running["book_id"] if running and running["status"] == "running" else None}


@app.post("/api/books")
def add_book(body: dict = Body(...)):
    folder = str(body.get("input_dir") or "").strip().strip('"')
    if not folder or not os.path.isdir(folder):
        _error(400, "폴더를 찾지 못했어요. 경로를 다시 확인해 주세요.")
    if not list_page_images(folder):
        _error(400, "이 폴더에는 만화 그림 파일(jpg, png, webp 등)이 없어요.")
    return library.book_state(library.add_book(folder))


@app.get("/api/books/{book_id}")
def book(book_id: str):
    return library.book_state(_book_or_404(book_id))


@app.delete("/api/books/{book_id}")
def remove_book(book_id: str):
    _book_or_404(book_id)
    running = jobs.snapshot()
    if running and running["book_id"] == book_id and running["status"] == "running":
        _error(409, "번역 중인 책은 책장에서 뺄 수 없어요.")
    library.remove_book(book_id)
    return {"ok": True}


@app.post("/api/books/{book_id}/open")
def open_folder(book_id: str, body: dict = Body(default={})):
    book = _book_or_404(book_id)
    folder = book["input_dir"] if body.get("which") == "input" else book["output_dir"]
    if not os.path.isdir(folder):
        _error(404, "아직 결과 폴더가 없어요. 번역을 시작하면 만들어져요.")
    if sys.platform == "win32":
        os.startfile(folder)  # noqa: S606 — 이 컴퓨터의 탐색기로 연다
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder])
    return {"ok": True}


@app.get("/api/books/{book_id}/pages/{name}/{kind}")
def page_image(request: Request, book_id: str, name: str, kind: str, w: int = 0, v: str = ""):
    book = _book_or_404(book_id)
    if kind not in ("original", "translated"):
        _error(404, "없는 그림이에요.")
    if name not in {os.path.basename(p) for p in list_page_images(book["input_dir"])}:
        _error(404, "없는 쪽이에요.")
    path = os.path.join(book["input_dir"] if kind == "original" else book["output_dir"], name)
    if not os.path.exists(path):
        _error(404, "아직 만들지 않은 쪽이에요.")
    # 버전이 붙은 주소만 오래 둔다 — 같은 이름으로 원고를 바꿔도 화면이 옛 그림을 보여 주지 않게 나머지는 매번 묻는다
    cache = {"Cache-Control": "max-age=31536000, immutable" if v else "no-cache"}
    served = path if w <= 0 else _thumbnail(path, min(w, 1600))
    response = FileResponse(served, media_type=None if w <= 0 else "image/jpeg", headers=cache,
                            stat_result=os.stat(served))
    # 화면에 있는 그림과 같으면 그림 없이 '바뀌지 않음'(304)만 돌려준다
    if response.headers["etag"] in [tag.strip(" W/") for tag in request.headers.get("if-none-match", "").split(",")]:
        return Response(status_code=304, headers={"ETag": response.headers["etag"], **cache})
    return response


def _thumbnail(path: str, width: int) -> str:
    os.makedirs(THUMB_DIR, exist_ok=True)
    key = hashlib.sha1(f"{path}|{os.path.getmtime(path)}|{width}".encode("utf-8")).hexdigest()
    target = os.path.join(THUMB_DIR, key + ".jpg")
    if not os.path.exists(target):
        image = read_image_bgr(path)
        if image is None:
            _error(404, "그림을 읽지 못했어요.")
        h, w = image.shape[:2]
        if w > width:
            image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            _error(500, "그림을 줄이지 못했어요.")
        encoded.tofile(target)
    return target


# ── 번역 고치기 ──

def _page_or_404(book: dict, name: str):
    if name not in {os.path.basename(p) for p in list_page_images(book["input_dir"])}:
        _error(404, "없는 쪽이에요.")


def _busy_with(book_id: str) -> bool:
    running = jobs.snapshot()
    return bool(running and running["status"] == "running" and running["book_id"] == book_id)


@app.get("/api/books/{book_id}/lines/{name}")
def page_lines(book_id: str, name: str):
    book = _book_or_404(book_id)
    _page_or_404(book, name)
    data = editor.page_lines(book, name)
    if data is None:
        _error(404, "이 책은 아직 번역하지 않았어요.")
    return data


@app.put("/api/books/{book_id}/lines/{name}")
def save_page_lines(book_id: str, name: str, body: dict = Body(...)):
    book = _book_or_404(book_id)
    _page_or_404(book, name)
    if _busy_with(book_id):
        _error(409, "이 책을 번역하거나 다시 쓰는 동안에는 고칠 수 없어요. 끝난 뒤에 고쳐 주세요.")
    edits = body.get("edits") or {}
    if not isinstance(edits, dict):
        _error(400, "고친 내용의 모양이 올바르지 않아요.")
    try:
        changed = editor.save_page_lines(book, name, edits, body.get("version"))
    except editor.Conflict:
        _error(409, "그사이 대사집이 바뀌었어요. 쪽을 다시 열어 주세요.")
    except KeyError as e:
        _error(404, str(e.args[0]) if e.args else "없는 대사예요.")
    return {"changed": changed, **editor.page_lines(book, name)}


@app.post("/api/books/{book_id}/retypeset/{name}")
def retypeset_page(book_id: str, name: str):
    """이 쪽만 저장된 번역으로 다시 쓴다 (번역 요청 없음)."""
    book = _book_or_404(book_id)
    _page_or_404(book, name)
    try:
        jobs.start(book, "retypeset", pages=[name])
    except RuntimeError as e:
        _error(409, str(e))
    return jobs.snapshot()


# ── 번역 작업 ──

@app.post("/api/books/{book_id}/run")
def run_book(book_id: str, body: dict = Body(default={})):
    book = _book_or_404(book_id)
    mode = body.get("mode") or "resume"
    if mode not in ("resume", "fresh", "retypeset"):
        _error(400, "실행 방식이 올바르지 않아요.")
    if not os.path.isdir(book["input_dir"]):
        _error(400, "원고 폴더를 찾지 못했어요. 폴더를 옮기거나 지웠는지 확인해 주세요.")
    if mode != "retypeset" and not translator.resolve_backend():
        _error(400, _missing_backend_message())
    try:
        jobs.start(book, mode)
    except RuntimeError as e:
        _error(409, str(e))
    library.touch(book_id)
    return jobs.snapshot()


@app.post("/api/job/stop")
def stop_job():
    return {"ok": jobs.stop()}


@app.get("/api/job/events")
async def job_events(request: Request):
    """지금 작업 상태가 바뀔 때마다 한 번씩 보낸다 (SSE)."""
    async def stream():
        last, idle = None, 0
        while not await request.is_disconnected():
            version = jobs.version()
            if version != last:
                last, idle = version, 0
                yield f"data: {json.dumps(jobs.snapshot() or {}, ensure_ascii=False)}\n\n"
            else:
                idle += 1
                if idle % 40 == 0:
                    yield ": keep-alive\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    return JSONResponse({"message": exc.detail}, status_code=exc.status_code)


if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="ui")
