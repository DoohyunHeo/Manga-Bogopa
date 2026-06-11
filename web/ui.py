"""Gradio UI 레이아웃 및 이벤트 바인딩 모듈"""
import os
import zipfile
import tempfile
import tkinter as tk
from tkinter import filedialog

import cv2
import gradio as gr

from src import config
from src.progress import PipelinePhase
from web.state import app_state
from web.handlers import run_pipeline_with_events
from web.theme import FOOTER_HTML, HEADER_HTML


def _rgb_to_hex(rgb) -> str:
    """(R, G, B) 튜플을 '#RRGGBB' 문자열로."""
    try:
        r, g, b = (max(0, min(255, int(v))) for v in rgb)
    except (TypeError, ValueError):
        return "#000000"
    return f"#{r:02X}{g:02X}{b:02X}"


def _hex_to_rgb(hex_str: str, default=(0, 0, 0)):
    """'#RRGGBB' 또는 'rgba(...)' 문자열을 (R, G, B) 튜플로. 실패 시 default."""
    if not hex_str:
        return default
    s = hex_str.strip()
    if s.startswith("#") and len(s) == 7:
        try:
            return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
        except ValueError:
            return default
    if s.startswith("rgba") or s.startswith("rgb"):
        try:
            inner = s[s.index("(") + 1:s.index(")")]
            parts = [p.strip() for p in inner.split(",")[:3]]
            return tuple(max(0, min(255, int(float(p)))) for p in parts)
        except (ValueError, IndexError):
            return default
    return default


def _pick_folder(current_path: str = "") -> str:
    """네이티브 폴더 선택 다이얼로그를 열고 선택된 경로를 반환합니다.

    로컬 실행 전용. 서버/헤드리스 환경에서는 tkinter 초기화가 실패할 수 있으므로
    조용히 기존 경로를 반환.
    """
    try:
        root = tk.Tk()
    except tk.TclError:
        gr.Warning("디스플레이가 없어 폴더 선택 창을 열 수 없습니다. 경로를 직접 입력해 주세요.")
        return current_path

    try:
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(
            title="폴더 선택",
            initialdir=current_path if current_path and os.path.isdir(current_path) else ".",
        )
    finally:
        root.destroy()
    return folder if folder else current_path


# ---------------------------------------------------------------------------
# 초기 설정 (첫 실행)
# ---------------------------------------------------------------------------

def _panel_label(text: str):
    """컷(panel) 머리표 — 괘선 틱이 붙은 섹션 라벨."""
    return gr.HTML(f'<div class="panel-label">{text}</div>', elem_classes=["panel-label-wrap"])


def _get_font_list():
    """폰트 디렉토리에서 사용 가능한 폰트 파일 목록을 반환합니다."""
    font_dir = config._config.FONT_DIR
    if not os.path.isdir(font_dir):
        return []
    return sorted([f for f in os.listdir(font_dir) if f.lower().endswith(('.ttf', '.otf', '.ttc'))])


def _save_and_initialize(api_key, gemini_model, input_dir, output_dir):
    """초기 설정 저장 → 파이프라인 준비 → UI 전환.

    yield로 3단계 상태를 UI에 노출: 검증 실패 / 저장 완료 / 준비 완료 | 실패.
    """
    if not api_key.strip():
        yield ("API 키를 입력하세요.", gr.update(visible=True), gr.update(visible=False))
        return

    c = config._config
    c.GEMINI_API_KEY = api_key.strip()
    if gemini_model.strip():
        c.GEMINI_MODEL = gemini_model.strip()
    if input_dir.strip():
        c.INPUT_DIR = input_dir.strip()
    if output_dir.strip():
        c.OUTPUT_DIR = output_dir.strip()
    config.save()

    yield ("설정 저장 완료. 파이프라인을 초기화합니다...", gr.update(visible=True), gr.update(visible=False))

    try:
        app_state.initialize_pipeline()
    except Exception as e:
        yield (f"모델 로딩 실패: {e}", gr.update(visible=True), gr.update(visible=False))
        return

    yield ("준비 완료! 모델은 실행 시점에 로드됩니다.", gr.update(visible=False), gr.update(visible=True))


# ---------------------------------------------------------------------------
# 설정 탭 (전체 설정 편집)
# ---------------------------------------------------------------------------

class _SettingsRegistry:
    """Registers (widget, apply_fn) pairs so the save handler can be generic.

    - `setter(widget, field, cast=None)` for plain setattr bindings.
    - `bind(widget, apply_fn)` for custom logic (color conversion, dict updates).
    """

    def __init__(self):
        self.entries = []

    def bind(self, widget, apply_fn):
        self.entries.append((widget, apply_fn))
        return widget

    def setter(self, widget, field, cast=None):
        if cast is None:
            apply_fn = lambda c, v, f=field: setattr(c, f, v)
        else:
            apply_fn = lambda c, v, f=field, cc=cast: setattr(c, f, cc(v))
        return self.bind(widget, apply_fn)

    def widgets(self):
        return [w for w, _ in self.entries]


def _make_settings_save(registry):
    def save(*values):
        c = config._config
        for (_, apply_fn), value in zip(registry.entries, values):
            apply_fn(c, value)
        c.apply_font_modes()
        config.save()

        if app_state.is_ready:
            try:
                from src import model_loader
                new_session = model_loader._initialize_gemini()
                app_state.pipeline.models['translator'] = new_session
                return "설정이 저장되었습니다. Gemini 세션이 재초기화되었습니다."
            except Exception as e:
                return f"설정은 저장되었지만 Gemini 재초기화 실패: {e}"

        return "설정이 저장되었습니다."

    return save


def _apply_font_map_entry(style, filename, font_dir):
    if filename:
        config._config.FONT_MAP[style] = os.path.join(font_dir, filename)


# ---------------------------------------------------------------------------
# 번역 처리
# ---------------------------------------------------------------------------

_PHASE_LABELS = {
    PipelinePhase.LOADING_MODELS: "모델 로딩",
    PipelinePhase.DETECTION:      "탐지",
    PipelinePhase.OCR:            "글자 인식",
    PipelinePhase.FONT_ANALYSIS:  "글씨체 분석",
    PipelinePhase.TRANSLATION:    "번역",
    PipelinePhase.PASS1_BATCH:    "배치 완료",
    PipelinePhase.SAVING_JSON:    "데이터 저장",
    PipelinePhase.PASS2_PAGE:     "식자",
    PipelinePhase.COMPLETE:       "완료",
}

_LEVEL_PREFIX = {
    "warning": "⚠️ ",
    "error":   "❌ ",
    "info":    "",
}


def _validate_run_inputs(input_folder: str):
    """Return (cleaned_folder, error_message). error_message is None if valid."""
    if not input_folder or not input_folder.strip():
        return None, "입력 폴더 경로를 입력하세요."
    cleaned = input_folder.strip()
    if not os.path.isdir(cleaned):
        return None, f"폴더를 찾을 수 없습니다: {cleaned}"
    return cleaned, None


def _apply_run_settings(input_folder, output_folder, yolo_threshold, batch_size):
    """설정 저장 단일 진입점 — 번역 탭의 즉석 변경도 여기서만 반영."""
    c = config._config
    c.INPUT_DIR = input_folder
    c.YOLO_CONF_THRESHOLD = float(yolo_threshold)
    c.TRANSLATION_BATCH_SIZE = int(batch_size)
    if output_folder and output_folder.strip():
        c.OUTPUT_DIR = output_folder.strip()
    config.save()


def _format_event_line(event):
    """ProgressEvent를 로그 한 줄로 변환. level에 따라 ⚠️/❌ 접두사."""
    label = _PHASE_LABELS.get(event.phase, "")
    prefix = _LEVEL_PREFIX.get(event.level, "")
    if event.phase == PipelinePhase.PASS2_PAGE and event.total > 0:
        pct = int(event.current / max(event.total, 1) * 100)
        progress = f" {event.current}/{event.total} ({pct}%)"
    elif event.phase == PipelinePhase.PASS1_BATCH:
        progress = f" {event.current}/{event.total}"
    else:
        progress = ""
    return f"{prefix}[{label}]{progress} {event.message}".rstrip()


def _load_original_image(input_files: dict, page_name: str):
    """Lazy load one page's original RGB. Returns None on failure."""
    orig_path = input_files.get(page_name)
    if not orig_path:
        return None
    orig_bgr = cv2.imread(orig_path)
    if orig_bgr is None:
        return None
    return cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)


def _build_output_artifacts(output_dir: str):
    """결과 폴더에서 JSON + ZIP 경로 생성."""
    if not output_dir or not os.path.isdir(output_dir):
        return None, None
    json_path = None
    candidate = os.path.join(output_dir, "translation_data.json")
    if os.path.exists(candidate):
        json_path = candidate

    output_files = [
        os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir))
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ]
    zip_path = None
    if output_files:
        zip_path = os.path.join(tempfile.gettempdir(), "manga_bogopa_results.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fp in output_files:
                zf.write(fp, os.path.basename(fp))
    return json_path, zip_path


# ---------------------------------------------------------------------------
# 갤러리 상태 — 디버그 오버레이 토글을 위해 per-page 3종 이미지 유지
# ---------------------------------------------------------------------------

_GALLERY_STATE: dict = {"pages": []}  # each: {"orig","trans","debug","name"}


def _render_gallery(show_debug: bool):
    """현재 _GALLERY_STATE를 토글 상태에 맞춰 갤러리 리스트로 변환."""
    rendered = []
    for page in _GALLERY_STATE["pages"]:
        if page["orig"] is not None:
            rendered.append((page["orig"], f"원본 — {page['name']}"))
        chosen = page["debug"] if (show_debug and page["debug"] is not None) else page["trans"]
        label_prefix = "번역+탐지" if (show_debug and page["debug"] is not None) else "번역"
        rendered.append((chosen, f"{label_prefix} — {page['name']}"))
    return rendered


def _toggle_debug_view(show_debug: bool):
    """UI 토글 change 이벤트 — 갤러리만 다시 그림."""
    rendered = _render_gallery(bool(show_debug))
    return rendered if rendered else gr.update()


def _process_images(
    input_folder, output_folder,
    yolo_threshold, batch_size, show_debug, enable_checkpoint,
):
    cleaned_folder, error = _validate_run_inputs(input_folder)
    if error:
        gr.Warning(error)
        yield gr.update(), gr.update(), gr.update(), error
        return

    _apply_run_settings(cleaned_folder, output_folder, yolo_threshold, batch_size)
    resolved_output = output_folder.strip() if output_folder and output_folder.strip() else None

    input_files = {
        os.path.basename(os.path.join(cleaned_folder, f)): os.path.join(cleaned_folder, f)
        for f in os.listdir(cleaned_folder)
    }

    _GALLERY_STATE["pages"] = []
    output_dir = None
    log_lines = []

    for event in run_pipeline_with_events(resolved_output, enable_checkpoint):
        line = _format_event_line(event)
        if line:
            log_lines.append(line)

        gallery_update = gr.update()
        if event.phase == PipelinePhase.PASS2_PAGE and event.image_rgb is not None:
            page_name = event.page_name or f"Page {event.current}"
            orig_rgb = _load_original_image(input_files, page_name)
            debug_rgb = (event.extras or {}).get("debug_image_rgb")
            _GALLERY_STATE["pages"].append({
                "orig": orig_rgb,
                "trans": event.image_rgb.copy(),
                "debug": debug_rgb.copy() if debug_rgb is not None else None,
                "name": page_name,
            })
            gallery_update = _render_gallery(bool(show_debug))

        if event.phase == PipelinePhase.COMPLETE and event.page_name and os.path.isdir(event.page_name):
            output_dir = event.page_name

        yield (gallery_update, gr.update(), gr.update(), "\n".join(log_lines))

    json_path, zip_path = _build_output_artifacts(output_dir)
    final_gallery = _render_gallery(bool(show_debug))

    yield (
        final_gallery if final_gallery else gr.update(),
        gr.update(value=json_path, visible=json_path is not None),
        gr.update(value=zip_path, visible=zip_path is not None),
        "\n".join(log_lines),
    )


# ---------------------------------------------------------------------------
# UI 빌드
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    configured = config.is_configured() and app_state.is_ready
    c = config._config

    # Gradio 6부터 theme/css는 launch()에 전달한다 (main.py 참고).
    with gr.Blocks(title="Manga-Bogopa") as demo:
        gr.HTML(HEADER_HTML)

        # ── 초기 설정 (미설정 시) ──
        with gr.Group(visible=not configured, elem_classes=["panel"]) as setup_section:
            _panel_label("처음 설정")
            api_key_setup = gr.Textbox(
                label="Gemini API 키 (필수)", value=c.GEMINI_API_KEY,
                type="password", placeholder="AIza...",
            )
            with gr.Accordion("추가 설정", open=False):
                model_setup = gr.Textbox(label="Gemini 모델", value=c.GEMINI_MODEL)
                input_setup = gr.Textbox(label="기본 입력 폴더", value=c.INPUT_DIR)
                output_setup = gr.Textbox(label="기본 출력 폴더", value=c.OUTPUT_DIR)
            save_setup_btn = gr.Button("저장하고 시작", variant="primary", size="lg")
            setup_status = gr.Textbox(label="상태", interactive=False)

        # ── 메인 (번역 + 설정 탭) ──
        # Group은 자식 사이 경계선용 배경을 깔아 탭 뒤로 비치므로 Column을 쓴다.
        with gr.Column(visible=configured) as main_section:
            with gr.Tabs():
                # ── 번역 탭 ──
                with gr.Tab("번역"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2):
                            with gr.Group(elem_classes=["panel"]):
                                _panel_label("원고 폴더")
                                gr.Markdown("원본 페이지가 든 폴더")
                                with gr.Row():
                                    input_folder = gr.Textbox(value=c.INPUT_DIR, show_label=False,
                                                              container=False, scale=4)
                                    input_browse = gr.Button("찾아보기", scale=1)
                                gr.Markdown("완성본을 저장할 폴더")
                                with gr.Row():
                                    output_folder = gr.Textbox(value=c.OUTPUT_DIR, show_label=False,
                                                               container=False, scale=4)
                                    output_browse = gr.Button("찾아보기", scale=1)
                                with gr.Accordion("이번 작업 옵션", open=False):
                                    run_yolo = gr.Slider(0.1, 0.9, value=c.YOLO_CONF_THRESHOLD, step=0.05,
                                                         label="탐지 신뢰도",
                                                         info="말풍선·글자 탐지 최소 신뢰도. 낮추면 더 많이 잡히지만 오탐 증가")
                                    run_batch = gr.Number(value=c.TRANSLATION_BATCH_SIZE, label="번역 배치 크기", precision=0,
                                                          info="한 번에 처리할 페이지 수. 메모리 부족 시 줄이세요")
                                    run_ckpt = gr.Checkbox(value=True, label="이어서 작업",
                                                           info="중단했다가 다시 실행하면 끝난 페이지는 건너뜀")
                                run_btn = gr.Button("번역 시작", variant="primary", size="lg",
                                                    elem_id="run-button")

                        with gr.Column(scale=3):
                            with gr.Group(elem_classes=["panel"]):
                                _panel_label("진행 상황")
                                translate_status = gr.Textbox(show_label=False, interactive=False,
                                                              lines=14, elem_id="log-box",
                                                              placeholder="번역을 시작하면 여기에 공정별 진행이 표시됩니다.")
                                with gr.Row():
                                    json_dl = gr.File(label="번역 데이터 (JSON)", interactive=False,
                                                      height=84, visible=False)
                                    zip_dl = gr.File(label="결과 이미지 (ZIP)", interactive=False,
                                                     height=84, visible=False)

                    with gr.Group(elem_classes=["panel"]):
                        _panel_label("완성 원고")
                        show_debug_toggle = gr.Checkbox(
                            value=False,
                            label="탐지 영역 표시",
                            info="말풍선·글자 탐지 박스를 갤러리에 겹쳐 보여줍니다. 저장되는 이미지 파일엔 영향 없음",
                        )
                        gallery = gr.Gallery(label="왼쪽: 원본 / 오른쪽: 번역본",
                                             columns=2, height="auto", object_fit="contain",
                                             elem_id="result-gallery")

                    input_browse.click(fn=_pick_folder, inputs=[input_folder], outputs=[input_folder])
                    output_browse.click(fn=_pick_folder, inputs=[output_folder], outputs=[output_folder])
                    run_btn.click(
                        fn=_process_images,
                        inputs=[input_folder, output_folder, run_yolo, run_batch, show_debug_toggle, run_ckpt],
                        outputs=[gallery, json_dl, zip_dl, translate_status],
                    )
                    show_debug_toggle.change(
                        fn=_toggle_debug_view,
                        inputs=[show_debug_toggle],
                        outputs=[gallery],
                    )
                    translate_status.change(
                        fn=None, inputs=None, outputs=None,
                        js="() => { const el = document.querySelector('#log-box textarea'); if (el) el.scrollTop = el.scrollHeight; }"
                    )

                # ── 설정 탭 ──
                with gr.Tab("설정"):
                    _panel_label("전체 설정")
                    gr.Markdown(
                        "여기서 번역 도구의 모든 동작을 조절할 수 있습니다. "
                        "각 항목 아래 작은 글씨에 **무엇이 달라지는지**와 **언제 바꾸면 좋은지**를 적어 두었습니다. "
                        "잘 모르겠으면 그대로 두세요 — 기본값이 가장 무난합니다. "
                        "바꾼 뒤에는 맨 아래 **설정 저장**을 꼭 누르세요."
                    )
                    reg = _SettingsRegistry()
                    _strip = lambda v: v.strip() if isinstance(v, str) else v

                    with gr.Accordion("번역 서비스 연결 (Gemini)", open=True):
                        gr.Markdown("번역은 구글의 인공지능 'Gemini'가 담당합니다. 여기는 그 서비스에 접속하는 방법을 정하는 곳입니다.")
                        reg.setter(gr.Textbox(label="Gemini API 키", value=c.GEMINI_API_KEY, type="password",
                                              info="구글이 발급해 준 열쇠 문자열입니다. 이게 있어야 번역을 요청할 수 있어요. aistudio.google.com에서 무료로 만들 수 있습니다"),
                                   'GEMINI_API_KEY', _strip)
                        reg.setter(gr.Textbox(label="Gemini 모델 이름", value=c.GEMINI_MODEL,
                                              info="번역에 쓸 모델 이름입니다 (예: gemini-3-flash-preview). 구글이 새 모델을 내면 이름만 바꿔 끼우면 됩니다"),
                                   'GEMINI_MODEL', _strip)
                        reg.setter(gr.Radio(choices=["default", "low", "high"], value=c.GEMINI_THINKING_LEVEL,
                                            label="번역할 때 고민하는 정도",
                                            info="default = 알아서 / low = 빨리 번역 (품질 조금 손해) / high = 오래 고민하고 번역 (느리지만 더 자연스러움). 번역이 어색하게 느껴지면 high로 올려 보세요"),
                                   'GEMINI_THINKING_LEVEL', str)
                        reg.setter(gr.Number(value=c.TRANSLATION_MAX_HISTORY_EXCHANGES, label="앞 내용을 기억하는 범위", precision=0,
                                             info="앞에서 번역한 내용을 몇 묶음까지 기억한 채로 다음 묶음을 번역할지입니다. 기억이 길수록 등장인물 이름·말투가 끝까지 일관되지만, 번역 비용이 조금 늘어납니다. 보통 6이면 충분합니다"),
                                   'TRANSLATION_MAX_HISTORY_EXCHANGES', lambda v: max(1, int(v)))

                    with gr.Accordion("번역 지침 (시스템 프롬프트)", open=False):
                        gr.Markdown("번역 모델에게 건네는 '작업 지시서'입니다. 인명 표기 규칙, 말투, 피해야 할 번역투 같은 것이 적혀 있고, 여기 적힌 대로 번역 품질이 달라집니다.")
                        reg.setter(gr.Textbox(label="시스템 프롬프트", value=c.SYSTEM_PROMPT,
                                              lines=15, max_lines=40,
                                              info="예를 들어 '특정 캐릭터는 반말로', '이 작품의 인명은 이렇게 표기' 같은 규칙을 한국어로 적어 넣으면 그대로 반영됩니다"),
                                   'SYSTEM_PROMPT')

                    with gr.Accordion("기본 폴더 위치", open=False):
                        gr.Markdown("번역 탭을 열 때 미리 채워져 있을 폴더 위치입니다. 번역 탭에서 그때그때 바꿀 수도 있습니다.")
                        reg.setter(gr.Textbox(label="원본 만화가 든 폴더", value=c.INPUT_DIR,
                                              info="번역하고 싶은 만화 이미지(jpg, png 등)들을 이 폴더에 넣어 두면 됩니다"),
                                   'INPUT_DIR', _strip)
                        reg.setter(gr.Textbox(label="완성본이 저장될 폴더", value=c.OUTPUT_DIR,
                                              info="번역이 끝난 페이지가 같은 파일 이름으로 여기에 저장됩니다"),
                                   'OUTPUT_DIR', _strip)

                    with gr.Accordion("모델 파일 위치", open=False):
                        gr.Markdown("각 작업을 담당하는 모델 파일이 컴퓨터 어디에 있는지 적어 두는 곳입니다. **파일을 직접 옮기지 않았다면 건드릴 필요가 없습니다.**")
                        reg.setter(gr.Textbox(value=c.MODEL_PATH, label="말풍선·글자 탐지 모델",
                                              info="만화 페이지에서 말풍선과 글자 위치를 찾아내는 모델"),
                                   'MODEL_PATH', _strip)
                        reg.setter(gr.Textbox(value=c.FONT_APPEARANCE_MODEL_PATH, label="글씨 분위기·기울기 판정 모델",
                                              info="원본 글씨가 어떤 느낌(외침/귀여움/공포…)이고 얼마나 기울었는지 알아내는 모델"),
                                   'FONT_APPEARANCE_MODEL_PATH', _strip)
                        reg.setter(gr.Textbox(value=c.FONT_STYLE_MODEL_PATH, label="예비용 통합 모델 (옛 버전)",
                                              info="위 두 모델이 없을 때만 대신 쓰는 옛 버전. 보통은 쓰이지 않습니다"),
                                   'FONT_STYLE_MODEL_PATH', _strip)
                        reg.setter(gr.Textbox(value=c.INPAINT_MANGA_MODEL_PATH, label="글자 지우기 모델 (만화 특화)",
                                              info="원본 일본어 글자를 지우고 그 자리를 그림으로 메우는 모델. 없으면 첫 실행 때 자동으로 내려받습니다"),
                                   'INPAINT_MANGA_MODEL_PATH', _strip)

                    with gr.Accordion("말풍선·글자 찾기", open=False):
                        gr.Markdown("페이지에서 말풍선과 글자를 얼마나 꼼꼼하게 찾을지 정합니다. **글자를 자꾸 놓치면** 여기를 조정해 보세요.")
                        reg.setter(gr.Slider(0.1, 0.9, value=c.YOLO_CONF_THRESHOLD, step=0.05,
                                             label="얼마나 확실해야 글자로 인정할지",
                                             info="숫자를 낮추면 애매한 것까지 글자로 인정해서 더 많이 잡지만, 글자가 아닌 그림도 잘못 잡을 수 있습니다. 글자를 놓치면 낮추고, 엉뚱한 곳을 잡으면 올리세요. 보통 0.4 근처가 적당합니다"),
                                   'YOLO_CONF_THRESHOLD', float)
                        reg.setter(gr.Slider(640, 1664, value=c.DETECTION_IMGSZ, step=32,
                                             label="탐지 입력 해상도",
                                             info="페이지를 이 크기로 맞춰 놓고 글자를 찾습니다. 1344 권장 (탐지 모델이 그 해상도로 학습됨). 낮추면 빨라지지만 작은 글자를 놓칩니다"),
                                   'DETECTION_IMGSZ', int)
                        reg.setter(gr.Slider(0.05, 0.8, value=c.TEXT_MERGE_OVERLAP_THRESHOLD, step=0.05,
                                             label="겹친 글자 묶음을 하나로 합치는 기준",
                                             info="가까이 붙은 글자 영역끼리 이만큼 겹치면 한 덩어리로 봅니다. 한 대사가 두 동강 나면 낮추고, 다른 대사끼리 붙어 버리면 올리세요"),
                                   'TEXT_MERGE_OVERLAP_THRESHOLD', float)
                        reg.setter(gr.Number(value=c.OCR_NUM_BEAMS, label="글자 인식(OCR) 꼼꼼함 (1~5)", precision=0,
                                             info="일본어를 읽을 때 몇 가지 해석을 비교해 볼지입니다. 올리면 오타가 줄지만 느려집니다. 보통 2면 충분합니다"),
                                   'OCR_NUM_BEAMS', lambda v: max(1, int(v)))
                        reg.setter(gr.Checkbox(value=c.OCR_PREFER_LOCAL_FILES, label="OCR 모델을 인터넷에서 다시 받지 않기",
                                               info="켜 두면 컴퓨터에 이미 받아 둔 것을 그대로 씁니다. 없으면 알아서 인터넷에서 받아오니 그냥 켜 두세요"),
                                   'OCR_PREFER_LOCAL_FILES', bool)
                        reg.setter(gr.Checkbox(value=c.OCR_WARMUP_ON_LOAD, label="첫 페이지 처리 속도 올리기",
                                               info="시작할 때 미리 한 번 연습 가동을 해 둬서 첫 페이지부터 빠르게 처리합니다. 대신 시작 준비가 조금 길어집니다"),
                                   'OCR_WARMUP_ON_LOAD', bool)

                    with gr.Accordion("속도와 VRAM 부담 (고급)", open=False):
                        gr.Markdown(
                            "작업을 몇 개씩 묶어서 동시에 처리할지 정하는 곳입니다. 숫자가 크면 빨라지지만 VRAM 부담이 커집니다. "
                            "**메모리 부족(out of memory) 오류가 나면 여기 숫자들을 절반으로 줄여 보세요.** 문제없이 잘 돌아간다면 그대로 두면 됩니다."
                        )
                        reg.setter(gr.Number(value=c.TRANSLATION_BATCH_SIZE, label="한 번에 번역 보낼 페이지 수", precision=0,
                                             info="이만큼의 페이지를 모아 한 번에 번역을 요청합니다. 많을수록 앞뒤 맥락을 잘 살리지만, 너무 크면 한 번 실패했을 때 다시 할 양도 많아집니다. 보통 50"),
                                   'TRANSLATION_BATCH_SIZE', int)
                        reg.setter(gr.Number(value=c.DETECTION_BATCH_SIZE, label="글자 찾기: 동시에 처리할 페이지 수", precision=0),
                                   'DETECTION_BATCH_SIZE', lambda v: max(1, int(v)))
                        reg.setter(gr.Checkbox(value=c.DETECTION_HALF, label="글자 찾기를 가볍게 계산 (half precision)",
                                               info="GPU에서 더 가벼운 계산 방식을 씁니다. 결과 차이는 거의 없고 빨라지니 켜 두세요"),
                                   'DETECTION_HALF', bool)
                        reg.setter(gr.Number(value=c.OCR_BATCH_SIZE, label="글자 읽기: 동시에 처리할 조각 수", precision=0),
                                   'OCR_BATCH_SIZE', int)
                        reg.setter(gr.Number(value=c.FONT_MODEL_BATCH_SIZE, label="글씨 분석: 동시에 처리할 조각 수", precision=0),
                                   'FONT_MODEL_BATCH_SIZE', int)
                        reg.setter(gr.Number(value=c.INPAINT_BATCH_SIZE, label="글자 지우기: 동시에 처리할 조각 수", precision=0),
                                   'INPAINT_BATCH_SIZE', int)
                        reg.setter(gr.Number(value=c.PASS2_MICROBATCH_SIZE, label="글자 그리기: 동시에 처리할 페이지 수", precision=0),
                                   'PASS2_MICROBATCH_SIZE', lambda v: max(1, int(v)))
                        reg.setter(gr.Number(value=c.PASS1_IMAGE_LOAD_WORKERS, label="번역 준비: 동시에 여는 이미지 수", precision=0,
                                             info="하드디스크에서 페이지 이미지를 몇 장씩 동시에 읽어올지입니다"),
                                   'PASS1_IMAGE_LOAD_WORKERS', lambda v: max(1, int(v)))
                        reg.setter(gr.Number(value=c.PASS2_IMAGE_LOAD_WORKERS, label="글자 그리기 준비: 동시에 여는 이미지 수", precision=0),
                                   'PASS2_IMAGE_LOAD_WORKERS', lambda v: max(1, int(v)))
                        reg.setter(gr.Number(value=c.PASS1_EMPTY_CACHE_EVERY_N_BATCHES, label="번역 단계: VRAM 비우기 주기", precision=0,
                                             info="0 = 알아서. 메모리 부족 오류가 자주 나면 1~3으로 바꿔 보세요 (몇 묶음마다 한 번씩 VRAM 캐시를 비웁니다)"),
                                   'PASS1_EMPTY_CACHE_EVERY_N_BATCHES', lambda v: max(0, int(v)))
                        reg.setter(gr.Number(value=c.PASS2_EMPTY_CACHE_EVERY_N_BATCHES, label="식자 단계: VRAM 비우기 주기", precision=0,
                                             info="0 = 알아서. 메모리 부족 시 1~3 권장"),
                                   'PASS2_EMPTY_CACHE_EVERY_N_BATCHES', lambda v: max(0, int(v)))

                    with gr.Accordion("말풍선 안 글자 배치", open=False):
                        gr.Markdown("번역 글자를 말풍선 안 어디에, 얼마나 여유를 두고 앉힐지 정합니다. **글자가 말풍선에 꽉 차 보이거나 너무 작아 보이면** 여기를 조정하세요.")
                        reg.setter(gr.Slider(0.0, 0.5, value=c.BUBBLE_PADDING_RATIO, step=0.01,
                                             label="말풍선 안쪽 숨구멍 (여백)",
                                             info="말풍선 가장자리와 글자 사이를 비워 두는 비율입니다. 0.15면 양쪽을 15%씩 비웁니다. 글자가 답답해 보이면 올리고, 글씨를 더 크게 쓰고 싶으면 내리세요"),
                                   'BUBBLE_PADDING_RATIO', float)
                        reg.setter(gr.Number(value=c.BUBBLE_EDGE_SAFE_MARGIN, label="가장자리 최소 간격 (픽셀)", precision=0,
                                             info="글자가 말풍선 가장자리에 이 거리(화면의 점 단위)보다 가깝게 붙지 않도록 지켜 줍니다"),
                                   'BUBBLE_EDGE_SAFE_MARGIN', int)
                        reg.setter(gr.Number(value=c.ATTACHED_BUBBLE_TEXT_MARGIN, label="컷 테두리 쪽으로 당기는 거리 (픽셀)", precision=0,
                                             info="말풍선이 만화 컷 테두리에 딱 붙어 있을 때, 글자도 그쪽으로 살짝 당겨 자연스럽게 보이게 합니다"),
                                   'ATTACHED_BUBBLE_TEXT_MARGIN', int)
                        reg.setter(gr.Slider(0.02, 0.3, value=c.BUBBLE_ATTACHMENT_EDGE_RATIO, step=0.01,
                                             label="'테두리에 붙었나' 살펴보는 범위",
                                             info="말풍선 양 끝의 이만큼 영역을 살펴서 컷 테두리에 붙었는지 판단합니다. 보통 그대로 두면 됩니다"),
                                   'BUBBLE_ATTACHMENT_EDGE_RATIO', float)
                        reg.setter(gr.Slider(0.3, 1.0, value=c.BUBBLE_ATTACHMENT_MIN_LENGTH_RATIO, step=0.05,
                                             label="'테두리에 붙었다' 인정 기준",
                                             info="말풍선 높이에 비해 이만큼 길게 이어진 세로선이 있어야 '컷 테두리에 붙은 말풍선'으로 인정합니다. 보통 그대로 두면 됩니다"),
                                   'BUBBLE_ATTACHMENT_MIN_LENGTH_RATIO', float)

                    with gr.Accordion("원본 글자 지우기", open=False):
                        gr.Markdown("번역 글자를 쓰기 전에 원본 일본어 글자를 지우고 그 자리를 그림으로 메우는 단계입니다. **지운 자리에 글자 흔적이나 얼룩이 보이면** 여기를 조정하세요.")
                        reg.setter(gr.Radio(choices=["manga", "photo"], value=c.INPAINT_MODEL,
                                            label="지우기 모델",
                                            info="manga = 만화 그림 전문 (권장, 만화 특유의 점 무늬·패턴을 더 자연스럽게 메움) / photo = 일반 사진용"),
                                   'INPAINT_MODEL', str)
                        reg.setter(gr.Number(value=c.INPAINT_CONTEXT_PADDING, label="지울 때 함께 살펴볼 주변 범위 (픽셀, 최소값)", precision=0,
                                             info="글자 주변 그림을 최소 이만큼 같이 보면서 비슷하게 메웁니다. 큰 글자 영역은 자동으로 더 넓게 봅니다. 보통 50"),
                                   'INPAINT_CONTEXT_PADDING', int)
                        reg.setter(gr.Number(value=c.INPAINT_BUBBLE_MASK_PADDING, label="말풍선 안: 여유 있게 지우는 정도 (픽셀)", precision=0,
                                             info="찾아낸 글자 영역보다 이만큼 더 넓게 지웁니다. 지운 자리에 글자 부스러기나 검은 얼룩이 남으면 숫자를 올리세요. 말풍선 테두리 선은 알아서 피해 가니 안심해도 됩니다"),
                                   'INPAINT_BUBBLE_MASK_PADDING', lambda v: max(0, int(v)))
                        reg.setter(gr.Number(value=c.INPAINT_MASK_PADDING, label="말풍선 밖: 여유 있게 지우는 정도 (픽셀)", precision=0,
                                             info="효과음처럼 그림 위에 직접 쓰인 글자용입니다. 너무 올리면 멀쩡한 그림까지 지웠다 다시 그리게 되니 조금씩만 올리세요"),
                                   'INPAINT_MASK_PADDING', lambda v: max(0, int(v)))

                    with gr.Accordion("번역 글자 쓰기 (가로/세로·크기)", open=False):
                        gr.Markdown("번역 글자를 언제 세로로 쓸지, 글씨 크기를 어디까지 허용할지 정합니다. 세로쓰기는 항상 **한 줄**로만 씁니다 — 말풍선 글이 한 줄에 안 들어가면 가로로 씁니다.")
                        reg.setter(gr.Checkbox(value=c.ENABLE_VERTICAL_TEXT, label="세로쓰기 사용",
                                               info="끄면 모든 글자를 가로로만 씁니다. 좁고 긴 말풍선에는 세로쓰기가 자연스러우니 보통 켜 둡니다"),
                                   'ENABLE_VERTICAL_TEXT', bool)
                        reg.setter(gr.Number(value=c.VERTICAL_TEXT_THRESHOLD, label="세로쓰기를 고려하는 길쭉함 (배)", precision=0,
                                             info="글자 영역의 높이가 너비의 이 배수 이상으로 길쭉하면 세로쓰기를 고려합니다. 보통 4"),
                                   'VERTICAL_TEXT_THRESHOLD', int)
                        reg.setter(gr.Slider(2.0, 20.0, value=c.VERTICAL_FORCE_ASPECT_RATIO, step=0.5,
                                             label="무조건 세로쓰기로 가는 길쭉함 (배)",
                                             info="이 배수 이상으로 극단적으로 길쭉한 영역은 따질 것 없이 세로로 씁니다. 보통 6"),
                                   'VERTICAL_FORCE_ASPECT_RATIO', float)
                        reg.setter(gr.Number(value=c.MIN_ROTATION_ANGLE, label="이 각도 이하 기울기는 무시 (도)", precision=0,
                                             info="원본 글씨가 살짝만 기울어 있으면 굳이 따라 기울이지 않고 똑바로 씁니다"),
                                   'MIN_ROTATION_ANGLE', int)
                        reg.setter(gr.Slider(0.1, 1.0, value=c.FONT_SHRINK_THRESHOLD_RATIO, step=0.05,
                                             label="가로쓰기를 포기하는 기준",
                                             info="가로로 맞추려다 글씨가 원래 크기의 이 비율보다 작아지면, 가로를 포기하고 세로쓰기를 시도합니다. 0.75 = 원래 크기의 75% 밑으로 줄면 포기"),
                                   'FONT_SHRINK_THRESHOLD_RATIO', float)
                        reg.setter(gr.Number(value=c.MIN_READABLE_TEXT_SIZE, label="읽기 편한 최소 글씨 크기 (픽셀)", precision=0,
                                             info="원본 글씨가 이것보다 작은 자리에는 세로쓰기를 시도하지 않습니다 (작은 글씨 세로쓰기는 읽기 어려워서)"),
                                   'MIN_READABLE_TEXT_SIZE', int)
                        reg.setter(gr.Number(value=c.MIN_FONT_SIZE, label="글씨 크기 하한 (픽셀)", precision=0,
                                             info="공간이 아무리 좁아도 이보다 작은 글씨는 쓰지 않습니다"),
                                   'MIN_FONT_SIZE', int)
                        reg.setter(gr.Number(value=c.MAX_FONT_SIZE, label="글씨 크기 상한 (픽셀)", precision=0,
                                             info="공간이 아무리 넓어도 이보다 큰 글씨는 쓰지 않습니다"),
                                   'MAX_FONT_SIZE', int)
                        reg.setter(gr.Slider(0.1, 1.0, value=c.FONT_AREA_FILL_RATIO, step=0.05,
                                             label="허전함 방지 기준",
                                             info="글자가 자리의 이 비율도 못 채우고 허전해 보이면 글씨를 자동으로 키웁니다"),
                                   'FONT_AREA_FILL_RATIO', float)

                    with gr.Accordion("글씨체·크기 자동 판정", open=False):
                        gr.Markdown("원본 글씨의 분위기와 크기를 알아내서 번역 글자에 그대로 입히는 단계입니다. 크기는 원본 글자 모양을 직접 재서 정합니다. **글씨체가 자꾸 엉뚱하게 골라지거나 크기가 이상하면** 여기를 조정하세요.")
                        reg.setter(gr.Radio(choices=["off", "light", "strong"], value=c.FONT_SIZE_CORRECTION_MODE,
                                            label="원본 글씨 크기 따라가기",
                                            info="off = 보정 없이 그대로 / light = 원본 글자 크기를 살짝 참고 / strong = 원본 글자 크기를 적극 반영 (권장)"),
                                   'FONT_SIZE_CORRECTION_MODE', str)
                        reg.setter(gr.Slider(0.05, 0.5, value=c.MODEL_FONT_SIZE_TOLERANCE, step=0.05,
                                             label="크기 조절 허용 폭 (±)",
                                             info="공간에 맞추느라 글씨를 키우거나 줄일 때, 원본에서 잰 글자 크기의 ±이만큼까지만 허용합니다. 0.2 = 원본의 80~120% 사이"),
                                   'MODEL_FONT_SIZE_TOLERANCE', float)
                        reg.setter(gr.Radio(choices=["off", "loose", "strict"], value=c.FONT_STYLE_FALLBACK_MODE,
                                            label="애매하면 기본 글씨체 쓰기",
                                            info="모델이 글씨체를 확신하지 못할 때 무난한 기본체로 쓸지 정합니다. off = 항상 판정대로 / loose = 아주 애매할 때만 기본체 / strict = 조금만 애매해도 기본체 (권장, 엉뚱한 글씨체 사고 방지)"),
                                   'FONT_STYLE_FALLBACK_MODE', str)
                        reg.setter(gr.Radio(choices=["off", "fast", "accurate"], value=c.FONT_MODEL_TTA_MODE,
                                            label="글씨체 판정 신중함",
                                            info="off = 한 번 보고 결정 (빠름) / fast = 두 번 보고 결정 / accurate = 세 번 보고 결정 (권장, 조금 느리지만 정확)"),
                                   'FONT_MODEL_TTA_MODE', str)
                        reg.setter(gr.Checkbox(value=c.VERTICAL_FURIGANA_STRIP_ENABLED,
                                               label="후리가나(한자 옆 작은 글자) 무시하기",
                                               info="일본 만화의 한자 옆에는 읽는 법을 알려주는 깨알 글자가 붙어 있는데, 이게 글씨 크기 판정을 흐릴 수 있어 판정할 때만 잘라내고 봅니다. 켜 두는 것을 권장"),
                                   'VERTICAL_FURIGANA_STRIP_ENABLED', bool)
                        reg.setter(gr.Slider(0.02, 0.3, value=c.VERTICAL_FURIGANA_MIN_GAP_RATIO, step=0.01,
                                             label="후리가나로 인정하는 틈새 크기",
                                             info="본문 글자와 깨알 글자 사이에 이만큼 틈이 있어야만 후리가나로 보고 잘라냅니다. 낮출수록 과감하게 잘라냅니다. 보통 그대로 두면 됩니다"),
                                   'VERTICAL_FURIGANA_MIN_GAP_RATIO', float)

                    with gr.Accordion("분위기별 글꼴 고르기", open=False):
                        gr.Markdown(
                            "모델이 판정한 글씨 분위기마다 어떤 글꼴 파일로 쓸지 정합니다. "
                            "글꼴 파일(.ttf, .otf)을 `data/fonts` 폴더에 넣으면 목록에 나타납니다.\n\n"
                            "분위기 이름 안내 — **standard**: 평범한 대사 / **shouting**: 외침 / **cute**: 귀여운 말투 / "
                            "**narration**: 해설·내레이션 / **handwriting**: 손글씨 느낌 / **pop**: 통통 튀는 강조 / "
                            "**angry**: 화난 대사 / **scared**: 겁먹은 대사 / **embarrassment**: 당황한 대사"
                        )
                        _font_files = _get_font_list()
                        for style in ["standard", "shouting", "cute", "narration", "handwriting", "pop", "angry", "scared", "embarrassment"]:
                            current = os.path.basename(c.FONT_MAP.get(style, ""))
                            reg.bind(
                                gr.Dropdown(choices=_font_files, value=current, label=style, info=f"현재: {current}"),
                                lambda cfg, v, s=style: _apply_font_map_entry(s, v, cfg.FONT_DIR),
                            )

                    with gr.Accordion("말풍선 밖 글자 (효과음·해설)", open=False):
                        gr.Markdown("말풍선 없이 그림 위에 바로 쓰인 글자(효과음, 해설, 중얼거림 등)를 어떻게 그릴지 정합니다. 이런 글자는 배경 그림과 섞이지 않도록 흰 테두리를 둘러서 씁니다.")
                        reg.setter(gr.Slider(0.0, 0.3, value=c.FREEFORM_PADDING_RATIO, step=0.01,
                                             label="글자 양옆 숨구멍 (여백)",
                                             info="찾아낸 글자 자리의 양옆을 이만큼 비우고 씁니다. 0.05 = 양쪽 5%씩"),
                                   'FREEFORM_PADDING_RATIO', float)
                        reg.setter(gr.Slider(0.0, 0.6, value=c.FREEFORM_BOX_OVERFLOW_RATIO, step=0.05,
                                             label="자리보다 크게 써도 되는 정도",
                                             info="한국어 번역이 원본 일본어보다 길 때가 많아, 찾아낸 자리보다 이 비율만큼 더 넓게 퍼지는 것을 허용합니다. 글씨가 답답하게 작아지면 올려 보세요. 0.2 = 20% 더 넓게 허용"),
                                   'FREEFORM_BOX_OVERFLOW_RATIO', float)
                        reg.setter(gr.Number(value=c.FREEFORM_STROKE_WIDTH, label="흰 테두리 두께 (픽셀)", precision=0,
                                             info="글자에 두르는 테두리의 두께입니다. 복잡한 그림 위에서 글자가 잘 안 보이면 두껍게 하세요"),
                                   'FREEFORM_STROKE_WIDTH', int)
                        reg.bind(
                            gr.ColorPicker(value=_rgb_to_hex(c.FREEFORM_FONT_COLOR),
                                           label="글자 색", info="글자 본체의 색입니다. 보통 검정"),
                            lambda cfg, v: setattr(cfg, 'FREEFORM_FONT_COLOR', _hex_to_rgb(v, cfg.FREEFORM_FONT_COLOR)),
                        )
                        reg.bind(
                            gr.ColorPicker(value=_rgb_to_hex(c.FREEFORM_STROKE_COLOR),
                                           label="테두리 색", info="글자를 두르는 테두리 색입니다. 보통 흰색"),
                            lambda cfg, v: setattr(cfg, 'FREEFORM_STROKE_COLOR', _hex_to_rgb(v, cfg.FREEFORM_STROKE_COLOR)),
                        )
                        reg.setter(gr.Number(value=c.FREEFORM_ATTACHMENT_SEARCH_PX, label="컷 테두리를 찾아보는 거리 (픽셀)", precision=0,
                                             info="글자 좌우로 이 거리 안에 만화 컷 테두리가 있으면 글자를 그쪽에 붙여 정렬합니다 (원본 만화의 배치 습관을 따라가는 기능)"),
                                   'FREEFORM_ATTACHMENT_SEARCH_PX', int)
                        reg.setter(gr.Slider(0.3, 1.0, value=c.FREEFORM_ATTACHMENT_MIN_LENGTH_RATIO, step=0.05,
                                             label="'테두리에 붙었다' 인정 기준",
                                             info="글자 높이에 비해 이만큼 길게 이어진 세로선이 있어야 컷 테두리로 인정합니다. 보통 그대로 두면 됩니다"),
                                   'FREEFORM_ATTACHMENT_MIN_LENGTH_RATIO', float)
                        reg.setter(gr.Number(value=c.FREEFORM_ATTACHMENT_TEXT_MARGIN, label="테두리 쪽으로 당기는 거리 (픽셀)", precision=0,
                                             info="컷 테두리에 붙은 글자로 판정됐을 때, 글자를 테두리 방향으로 이만큼 더 붙입니다"),
                                   'FREEFORM_ATTACHMENT_TEXT_MARGIN', int)
                        reg.setter(gr.Slider(0.3, 1.0, value=c.FREEFORM_STYLE_MIN_CONFIDENCE, step=0.05,
                                             label="해설 글씨체로 통일하는 기준",
                                             info="말풍선 밖 글자는 대부분 해설이라, 모델이 이 확신 아래로 글씨체를 고르면 그냥 해설용 글씨체로 통일합니다. 엉뚱한 글씨체가 자주 보이면 올리세요"),
                                   'FREEFORM_STYLE_MIN_CONFIDENCE', float)

                    save_settings_btn = gr.Button("설정 저장", variant="primary")
                    settings_result = gr.Textbox(label="저장 결과", interactive=False,
                                                 placeholder="설정을 저장하면 여기에 결과가 표시됩니다.")

                    save_settings_btn.click(
                        fn=_make_settings_save(reg),
                        inputs=reg.widgets(),
                        outputs=[settings_result],
                    )

        # 초기 설정 → 메인 전환
        save_setup_btn.click(
            fn=_save_and_initialize,
            inputs=[api_key_setup, model_setup, input_setup, output_setup],
            outputs=[setup_status, setup_section, main_section],
        )

        gr.HTML(FOOTER_HTML)

    demo.queue(max_size=1)
    return demo
