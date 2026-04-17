"""Gradio UI 레이아웃 및 이벤트 바인딩 모듈"""
import os
import zipfile
import tempfile
import tkinter as tk
from tkinter import filedialog

import cv2
import gradio as gr
import numpy as np

from src import config
from src.progress import PipelinePhase
from web.state import app_state
from web.handlers import run_pipeline_with_events


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
    """네이티브 폴더 선택 다이얼로그를 열고 선택된 경로를 반환합니다."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(
        title="폴더 선택",
        initialdir=current_path if current_path and os.path.isdir(current_path) else ".",
    )
    root.destroy()
    return folder if folder else current_path


# ---------------------------------------------------------------------------
# 초기 설정 (첫 실행)
# ---------------------------------------------------------------------------

def _get_font_list():
    """폰트 디렉토리에서 사용 가능한 폰트 파일 목록을 반환합니다."""
    font_dir = config._config.FONT_DIR
    if not os.path.isdir(font_dir):
        return []
    return sorted([f for f in os.listdir(font_dir) if f.lower().endswith(('.ttf', '.otf', '.ttc'))])


def _save_and_initialize(api_key, gemini_model, input_dir, output_dir):
    """설정 저장 → 모델 로딩 → UI 전환."""
    if not api_key.strip():
        yield ("API 키를 입력하세요.", gr.update(visible=True), gr.update(visible=False))
        return

    config._config.GEMINI_API_KEY = api_key.strip()
    if gemini_model.strip():
        config._config.GEMINI_MODEL = gemini_model.strip()
    if input_dir.strip():
        config._config.INPUT_DIR = input_dir.strip()
    if output_dir.strip():
        config._config.OUTPUT_DIR = output_dir.strip()
    config.save()

    yield ("설정 저장 완료. 파이프라인을 초기화합니다...", gr.update(visible=True), gr.update(visible=False))

    try:
        app_state.initialize_pipeline()
        yield ("준비 완료! 모델은 실행 시점에 로드됩니다.", gr.update(visible=False), gr.update(visible=True))
    except Exception as e:
        yield (f"모델 로딩 실패: {e}", gr.update(visible=True), gr.update(visible=False))


# ---------------------------------------------------------------------------
# 설정 탭 (전체 설정 편집)
# ---------------------------------------------------------------------------

def _save_all_settings(
    api_key, gemini_model, system_prompt,
    input_dir, output_dir,
    model_path, font_appearance_path, font_size_path, font_style_path,
    yolo_thresh, text_merge_iou,
    translation_batch, ocr_batch, inpaint_batch, font_model_batch,
    ocr_num_beams, ocr_prefer_local, ocr_warmup,
    pass1_workers, pass1_cache_every, pass2_microbatch, pass2_workers, pass2_cache_every,
    bubble_padding_ratio, bubble_edge_margin, attached_bubble_margin,
    bubble_attach_edge_ratio, bubble_attach_min_len_ratio,
    inpaint_ctx_padding, inpaint_mask_padding,
    enable_vertical, vertical_threshold, vertical_force_aspect, min_rotation,
    font_shrink_ratio, min_font, max_font, min_readable, font_area_fill,
    size_mode, size_tol, fallback_mode, tta_mode,
    font_standard, font_shouting, font_cute, font_narration, font_handwriting,
    font_pop, font_angry, font_scared, font_embarrassment,
    freeform_padding_ratio, freeform_stroke_width,
    freeform_font_color, freeform_stroke_color,
    freeform_attach_search_px, freeform_attach_min_len_ratio, freeform_attach_text_margin,
    freeform_style_min_conf,
    draw_debug_boxes,
):
    """전체 설정을 config.json에 저장합니다."""
    c = config._config
    c.GEMINI_API_KEY = api_key.strip()
    c.GEMINI_MODEL = gemini_model.strip()
    c.SYSTEM_PROMPT = system_prompt
    c.INPUT_DIR = input_dir.strip()
    c.OUTPUT_DIR = output_dir.strip()
    c.MODEL_PATH = model_path.strip()
    c.FONT_APPEARANCE_MODEL_PATH = font_appearance_path.strip()
    c.FONT_SIZE_MODEL_PATH = font_size_path.strip()
    c.FONT_STYLE_MODEL_PATH = font_style_path.strip()
    c.YOLO_CONF_THRESHOLD = yolo_thresh
    c.TEXT_MERGE_OVERLAP_THRESHOLD = float(text_merge_iou)
    c.TRANSLATION_BATCH_SIZE = int(translation_batch)
    c.OCR_BATCH_SIZE = int(ocr_batch)
    c.INPAINT_BATCH_SIZE = int(inpaint_batch)
    c.FONT_MODEL_BATCH_SIZE = int(font_model_batch)
    c.OCR_NUM_BEAMS = max(1, int(ocr_num_beams))
    c.OCR_PREFER_LOCAL_FILES = bool(ocr_prefer_local)
    c.OCR_WARMUP_ON_LOAD = bool(ocr_warmup)
    c.PASS1_IMAGE_LOAD_WORKERS = max(1, int(pass1_workers))
    c.PASS1_EMPTY_CACHE_EVERY_N_BATCHES = max(0, int(pass1_cache_every))
    c.PASS2_MICROBATCH_SIZE = max(1, int(pass2_microbatch))
    c.PASS2_IMAGE_LOAD_WORKERS = max(1, int(pass2_workers))
    c.PASS2_EMPTY_CACHE_EVERY_N_BATCHES = max(0, int(pass2_cache_every))
    c.BUBBLE_PADDING_RATIO = bubble_padding_ratio
    c.BUBBLE_EDGE_SAFE_MARGIN = int(bubble_edge_margin)
    c.ATTACHED_BUBBLE_TEXT_MARGIN = int(attached_bubble_margin)
    c.BUBBLE_ATTACHMENT_EDGE_RATIO = float(bubble_attach_edge_ratio)
    c.BUBBLE_ATTACHMENT_MIN_LENGTH_RATIO = float(bubble_attach_min_len_ratio)
    c.INPAINT_CONTEXT_PADDING = int(inpaint_ctx_padding)
    c.INPAINT_MASK_PADDING = int(inpaint_mask_padding)
    c.ENABLE_VERTICAL_TEXT = enable_vertical
    c.VERTICAL_TEXT_THRESHOLD = int(vertical_threshold)
    c.VERTICAL_FORCE_ASPECT_RATIO = float(vertical_force_aspect)
    c.MIN_ROTATION_ANGLE = int(min_rotation)
    c.FONT_SHRINK_THRESHOLD_RATIO = font_shrink_ratio
    c.MIN_FONT_SIZE = int(min_font)
    c.MAX_FONT_SIZE = int(max_font)
    c.MIN_READABLE_TEXT_SIZE = int(min_readable)
    c.FONT_AREA_FILL_RATIO = font_area_fill
    # 폰트 모델 동작 (고수준 모드 → 저수준 값 자동 파생)
    c.FONT_SIZE_CORRECTION_MODE = str(size_mode)
    c.MODEL_FONT_SIZE_TOLERANCE = float(size_tol)
    c.FONT_STYLE_FALLBACK_MODE = str(fallback_mode)
    c.FONT_MODEL_TTA_MODE = str(tta_mode)
    c.apply_font_modes()
    # 폰트 맵 업데이트
    font_dir = c.FONT_DIR
    font_selections = {
        "standard": font_standard, "shouting": font_shouting, "cute": font_cute,
        "narration": font_narration, "handwriting": font_handwriting, "pop": font_pop,
        "angry": font_angry, "scared": font_scared, "embarrassment": font_embarrassment,
    }
    for style, filename in font_selections.items():
        if filename:
            c.FONT_MAP[style] = os.path.join(font_dir, filename)

    c.FREEFORM_PADDING_RATIO = float(freeform_padding_ratio)
    c.FREEFORM_STROKE_WIDTH = int(freeform_stroke_width)
    c.FREEFORM_FONT_COLOR = _hex_to_rgb(freeform_font_color, c.FREEFORM_FONT_COLOR)
    c.FREEFORM_STROKE_COLOR = _hex_to_rgb(freeform_stroke_color, c.FREEFORM_STROKE_COLOR)
    c.FREEFORM_ATTACHMENT_SEARCH_PX = int(freeform_attach_search_px)
    c.FREEFORM_ATTACHMENT_MIN_LENGTH_RATIO = float(freeform_attach_min_len_ratio)
    c.FREEFORM_ATTACHMENT_TEXT_MARGIN = int(freeform_attach_text_margin)
    c.FREEFORM_STYLE_MIN_CONFIDENCE = float(freeform_style_min_conf)
    c.DRAW_DEBUG_BOXES = draw_debug_boxes
    config.save()

    # Gemini 모델이 변경되었으면 세션 재초기화
    if app_state.is_ready:
        try:
            from src import model_loader
            new_session = model_loader._initialize_gemini()
            app_state.pipeline.models['translator'] = new_session
            return "설정이 저장되었습니다. Gemini 세션이 재초기화되었습니다."
        except Exception as e:
            return f"설정은 저장되었지만 Gemini 재초기화 실패: {e}"

    return "설정이 저장되었습니다."


# ---------------------------------------------------------------------------
# 번역 처리
# ---------------------------------------------------------------------------

def _process_images(
    input_folder, output_folder,
    yolo_threshold, batch_size, draw_debug, enable_checkpoint,
):
    if not input_folder or not input_folder.strip():
        gr.Warning("입력 폴더 경로를 입력하세요.")
        yield gr.update(), gr.update(), gr.update(), "입력 폴더 경로를 입력하세요."
        return

    input_folder = input_folder.strip()
    if not os.path.isdir(input_folder):
        gr.Warning(f"폴더를 찾을 수 없습니다: {input_folder}")
        yield gr.update(), gr.update(), gr.update(), f"폴더를 찾을 수 없습니다: {input_folder}"
        return

    # 번역 옵션을 config에 저장
    config._config.INPUT_DIR = input_folder
    config._config.YOLO_CONF_THRESHOLD = yolo_threshold
    config._config.TRANSLATION_BATCH_SIZE = int(batch_size)
    config._config.DRAW_DEBUG_BOXES = draw_debug
    if output_folder and output_folder.strip():
        config._config.OUTPUT_DIR = output_folder.strip()
    config.save()

    output_folder = output_folder.strip() if output_folder and output_folder.strip() else None
    pairs = []         # (원본rgb, 번역rgb, page_name) 쌍 누적
    output_dir = None
    log_lines = []

    # 원본 이미지 로드용 경로 맵
    input_files = {
        os.path.basename(os.path.join(input_folder, f)): os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
    }

    for event in run_pipeline_with_events(
        input_folder, output_folder,
        yolo_threshold, int(batch_size), draw_debug, enable_checkpoint,
    ):
        phase_labels = {
            PipelinePhase.LOADING_MODELS: "모델 로딩",
            PipelinePhase.DETECTION:      "탐지",
            PipelinePhase.OCR:            "OCR + 폰트분석",
            PipelinePhase.FONT_ANALYSIS:  "폰트분석",
            PipelinePhase.TRANSLATION:    "번역",
            PipelinePhase.PASS1_BATCH:    "배치 완료",
            PipelinePhase.SAVING_JSON:    "데이터 저장",
            PipelinePhase.PASS2_PAGE:     "인페인팅 + 식자",
            PipelinePhase.COMPLETE:       "완료",
        }
        label = phase_labels.get(event.phase, "")
        pct = event.current / max(event.total, 1)

        line = ""
        if event.phase == PipelinePhase.LOADING_MODELS:
            line = f"[{label}] {event.message}"
        elif event.phase in (PipelinePhase.DETECTION, PipelinePhase.OCR,
                              PipelinePhase.FONT_ANALYSIS, PipelinePhase.TRANSLATION):
            line = f"[{label}] {event.message}"
        elif event.phase == PipelinePhase.PASS1_BATCH:
            line = f"[{label}] 배치 {event.current}/{event.total} — {event.message}"
        elif event.phase == PipelinePhase.SAVING_JSON:
            line = f"[{label}] {event.message}"
        elif event.phase == PipelinePhase.PASS2_PAGE:
            pct = int(event.current / max(event.total, 1) * 100)
            line = f"[{label}] {event.current}/{event.total} ({pct}%) — {event.message}"
            if event.image_rgb is not None:
                page_name = event.page_name or f"Page {event.current}"
                trans_rgb = event.image_rgb.copy()
                orig_rgb = None
                orig_path = input_files.get(page_name)
                if orig_path:
                    orig_bgr = cv2.imread(orig_path)
                    if orig_bgr is not None:
                        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                pairs.append((orig_rgb, trans_rgb, page_name))
        elif event.phase == PipelinePhase.COMPLETE:
            if event.page_name and os.path.isdir(event.page_name):
                output_dir = event.page_name
            line = f"[{label}] {event.message}"

        if line:
            log_lines.append(line)

        # 진행 중에는 로그만 업데이트, 이미지는 완료 후 한번에
        yield (gr.update(), gr.update(), gr.update(), "\n".join(log_lines))

    json_path = None
    zip_path = None
    if output_dir:
        candidate = os.path.join(output_dir, "translation_data.json")
        if os.path.exists(candidate):
            json_path = candidate
        output_files = [
            os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        if output_files:
            zip_path = os.path.join(tempfile.gettempdir(), "manga_bogopa_results.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fp in output_files:
                    zf.write(fp, os.path.basename(fp))

    # 완료 후 이미지 한번에 표시 — 원본/번역 교차 배치 (columns=2로 좌우 정렬)
    gallery_images = []
    for orig_rgb, trans_rgb, name in pairs:
        if orig_rgb is not None:
            gallery_images.append((orig_rgb, f"원본 — {name}"))
        gallery_images.append((trans_rgb, f"번역 — {name}"))

    yield (
        gallery_images if gallery_images else gr.update(),
        json_path, zip_path,
        "\n".join(log_lines),
    )


# ---------------------------------------------------------------------------
# UI 빌드
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    configured = config.is_configured() and app_state.is_ready
    c = config._config

    with gr.Blocks(title="Manga-Bogopa") as demo:
        gr.Markdown("# Manga-Bogopa\n일본 만화 자동 번역 및 식자 도구")

        # ── 초기 설정 (미설정 시) ──
        with gr.Group(visible=not configured) as setup_section:
            gr.Markdown("## 초기 설정")
            api_key_setup = gr.Textbox(
                label="Gemini API 키 (필수)", value=c.GEMINI_API_KEY,
                type="password", placeholder="AIza...",
            )
            with gr.Accordion("추가 설정", open=False):
                model_setup = gr.Textbox(label="Gemini 모델", value=c.GEMINI_MODEL)
                input_setup = gr.Textbox(label="기본 입력 폴더", value=c.INPUT_DIR)
                output_setup = gr.Textbox(label="기본 출력 폴더", value=c.OUTPUT_DIR)
            save_setup_btn = gr.Button("설정 저장 및 시작", variant="primary", size="lg")
            setup_status = gr.Textbox(label="상태", interactive=False)

        # ── 메인 (번역 + 설정 탭) ──
        with gr.Group(visible=configured) as main_section:
            with gr.Tabs():
                # ── 번역 탭 ──
                with gr.Tab("번역"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 입력 폴더")
                            with gr.Row():
                                input_folder = gr.Textbox(value=c.INPUT_DIR, show_label=False, scale=4)
                                input_browse = gr.Button("찾아보기", scale=1)
                            gr.Markdown("#### 출력 폴더")
                            with gr.Row():
                                output_folder = gr.Textbox(value=c.OUTPUT_DIR, show_label=False, scale=4)
                                output_browse = gr.Button("찾아보기", scale=1)
                            with gr.Accordion("번역 옵션", open=False):
                                run_yolo = gr.Slider(0.1, 0.9, value=c.YOLO_CONF_THRESHOLD, step=0.05,
                                                     label="YOLO 탐지 신뢰도",
                                                     info="말풍선·텍스트 탐지 최소 신뢰도. 낮추면 더 많이 잡히지만 오탐 증가")
                                run_batch = gr.Number(value=c.TRANSLATION_BATCH_SIZE, label="번역 배치 크기", precision=0,
                                                      info="한 번에 처리할 페이지 수. VRAM 부족 시 줄이세요")
                                run_debug = gr.Checkbox(value=c.DRAW_DEBUG_BOXES, label="탐지 영역 시각화",
                                                        info="결과에 말풍선·텍스트 탐지 박스를 색상별로 표시")
                                run_ckpt = gr.Checkbox(value=True, label="체크포인트 활성화",
                                                       info="중단 후 재실행 시 완료된 페이지를 건너뜀")
                            run_btn = gr.Button("번역 시작", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            translate_status = gr.Textbox(label="로그", interactive=False, lines=12, elem_id="log-box")
                            with gr.Row():
                                json_dl = gr.File(label="번역 데이터 (JSON)", interactive=False)
                                zip_dl = gr.File(label="결과 이미지 (ZIP)", interactive=False)

                    gallery = gr.Gallery(label="번역 결과 (왼쪽: 원본 / 오른쪽: 번역본)",
                                         columns=2, height="auto", object_fit="contain")

                    input_browse.click(fn=_pick_folder, inputs=[input_folder], outputs=[input_folder])
                    output_browse.click(fn=_pick_folder, inputs=[output_folder], outputs=[output_folder])
                    run_btn.click(
                        fn=_process_images,
                        inputs=[input_folder, output_folder, run_yolo, run_batch, run_debug, run_ckpt],
                        outputs=[gallery, json_dl, zip_dl, translate_status],
                    )
                    translate_status.change(
                        fn=None, inputs=None, outputs=None,
                        js="() => { const el = document.querySelector('#log-box textarea'); if (el) el.scrollTop = el.scrollHeight; }"
                    )

                # ── 설정 탭 ──
                with gr.Tab("설정"):
                    gr.Markdown("### 전체 설정 편집\n변경 후 **설정 저장**을 누르세요. API 키 변경은 서버 재시작이 필요합니다.")

                    with gr.Accordion("API", open=True):
                        s_api_key = gr.Textbox(label="Gemini API 키", value=c.GEMINI_API_KEY, type="password")
                        s_gemini_model = gr.Textbox(label="Gemini 모델명", value=c.GEMINI_MODEL,
                                                    info="번역에 사용할 Gemini 모델 (예: gemini-2.5-flash)")

                    with gr.Accordion("번역 프롬프트", open=False):
                        s_prompt = gr.Textbox(label="시스템 프롬프트", value=c.SYSTEM_PROMPT,
                                              lines=15, max_lines=40,
                                              info="Gemini에 보내는 번역 지침. 번역 품질에 직접적으로 영향을 줌")

                    with gr.Accordion("경로", open=False):
                        s_input_dir = gr.Textbox(label="기본 입력 폴더", value=c.INPUT_DIR)
                        s_output_dir = gr.Textbox(label="기본 출력 폴더", value=c.OUTPUT_DIR)

                    with gr.Accordion("모델 파일", open=False):
                        s_model_path = gr.Textbox(value=c.MODEL_PATH, label="말풍선·글자 탐지 모델",
                                                  info="만화 페이지에서 말풍선과 글자 영역을 찾아주는 모델 파일 위치")
                        s_font_appearance = gr.Textbox(value=c.FONT_APPEARANCE_MODEL_PATH, label="글씨체·기울기 모델",
                                                       info="원본 글씨가 어떤 분위기(화남/귀여움/외침…)이고 얼마나 기울어졌는지 판정하는 모델")
                        s_font_size = gr.Textbox(value=c.FONT_SIZE_MODEL_PATH, label="글씨 크기 모델",
                                                 info="원본 글씨 크기를 예측하는 모델")
                        s_font_style = gr.Textbox(value=c.FONT_STYLE_MODEL_PATH, label="옛 통합 모델 (대체용)",
                                                  info="위 두 모델이 없을 때 대신 쓰는 옛 버전 통합 모델")

                    with gr.Accordion("탐지 / 배치", open=False):
                        s_yolo = gr.Slider(0.1, 0.9, value=c.YOLO_CONF_THRESHOLD, step=0.05,
                                           label="탐지 신뢰도",
                                           info="말풍선·글자를 찾을 때 어느 정도 확신이 있어야 인정할지. 낮추면 더 많이 잡히지만 엉뚱한 것도 잡힘")
                        s_text_merge_iou = gr.Slider(0.05, 0.8, value=c.TEXT_MERGE_OVERLAP_THRESHOLD, step=0.05,
                                                     label="글자 영역 병합 겹침 비율",
                                                     info="가까이 있는 글자 박스들이 이 비율 이상 겹치면 하나로 합침. 높이면 더 잘게 쪼개지고, 낮추면 더 과감하게 합침")
                        s_trans_batch = gr.Number(value=c.TRANSLATION_BATCH_SIZE, label="번역 한 번에 보낼 페이지 수", precision=0,
                                                  info="한 번에 번역 서버에 보낼 페이지 수")
                        s_ocr_batch = gr.Number(value=c.OCR_BATCH_SIZE, label="글자 인식 묶음 크기", precision=0,
                                                info="글자를 읽는 단계에서 한 번에 처리할 조각 수. 메모리 부족하면 줄이세요")
                        s_font_model_batch = gr.Number(value=c.FONT_MODEL_BATCH_SIZE, label="글씨체 분석 묶음 크기", precision=0,
                                                       info="글씨체·크기·기울기 분석에서 한 번에 처리할 조각 수. 메모리 부족하면 줄이세요")
                        s_inpaint_batch = gr.Number(value=c.INPAINT_BATCH_SIZE, label="원본 글자 지우기 묶음 크기", precision=0,
                                                    info="원본 글자를 지우는 단계에서 한 번에 처리할 조각 수. 메모리 부족하면 줄이세요")
                        s_ocr_beams = gr.Number(value=c.OCR_NUM_BEAMS, label="글자 인식 정확도 레벨", precision=0,
                                                info="글자를 읽을 때 얼마나 많은 후보를 비교해볼지 (1~5). 올리면 정확해지지만 느려짐")
                        s_ocr_local = gr.Checkbox(value=c.OCR_PREFER_LOCAL_FILES, label="글자 인식 모델 내려받기 생략",
                                                  info="이미 저장된 글자 인식 모델이 있으면 그대로 사용. 없으면 자동으로 온라인에서 받음")
                        s_ocr_warmup = gr.Checkbox(value=c.OCR_WARMUP_ON_LOAD, label="첫 번역 속도 향상",
                                                   info="모델을 준비할 때 빈 이미지로 한 번 미리 돌려 놓아 첫 번역을 빠르게 시작 (준비 시간이 조금 길어짐)")

                    with gr.Accordion("성능 튜닝", open=False):
                        s_pass1_workers = gr.Number(value=c.PASS1_IMAGE_LOAD_WORKERS, label="번역 단계 이미지 로딩 병렬 수", precision=0,
                                                    info="번역 준비 단계에서 이미지를 동시에 몇 개 읽을지. 디스크가 빠르면 올려도 좋음")
                        s_pass1_cache = gr.Number(value=c.PASS1_EMPTY_CACHE_EVERY_N_BATCHES, label="번역 단계 메모리 비우기 주기", precision=0,
                                                  info="몇 묶음마다 GPU 메모리를 비울지. 0 = 자동. 메모리 부족이 생기면 1~3 정도로")
                        s_pass2_microbatch = gr.Number(value=c.PASS2_MICROBATCH_SIZE, label="식자 단계 동시 처리 페이지", precision=0,
                                                       info="식자(그리기) 단계에서 한 번에 몇 페이지를 같이 처리할지")
                        s_pass2_workers = gr.Number(value=c.PASS2_IMAGE_LOAD_WORKERS, label="식자 단계 이미지 로딩 병렬 수", precision=0,
                                                    info="식자 단계에서 이미지를 동시에 몇 개 읽을지")
                        s_pass2_cache = gr.Number(value=c.PASS2_EMPTY_CACHE_EVERY_N_BATCHES, label="식자 단계 메모리 비우기 주기", precision=0,
                                                  info="몇 묶음마다 GPU 메모리를 비울지. 0 = 자동")

                    with gr.Accordion("말풍선 식자", open=False):
                        s_bubble_pad = gr.Slider(0.0, 0.5, value=c.BUBBLE_PADDING_RATIO, step=0.01,
                                                 label="말풍선 안쪽 여백 비율",
                                                 info="말풍선 폭 대비 좌우 여백. 0.15 = 양쪽 15%씩 비워두고 글자 배치")
                        s_bubble_edge = gr.Number(value=c.BUBBLE_EDGE_SAFE_MARGIN, label="말풍선 가장자리 안전 거리 (px)", precision=0,
                                                  info="말꼬리가 있는 말풍선에서 글자가 가장자리에 너무 붙지 않도록 하는 최소 거리")
                        s_attached_margin = gr.Number(value=c.ATTACHED_BUBBLE_TEXT_MARGIN, label="테두리 붙음 정렬 여백 (px)", precision=0,
                                                      info="말풍선이 컷 테두리에 붙어 있을 때, 번역 글자를 테두리 방향으로 몇 픽셀 더 당길지")
                        s_bubble_attach_edge = gr.Slider(0.02, 0.3, value=c.BUBBLE_ATTACHMENT_EDGE_RATIO, step=0.01,
                                                         label="말풍선 가장자리 확인 범위",
                                                         info="말풍선 좌/우 몇 %를 컷 테두리 붙음 판정에 쓸지. 0.10 = 좌우 10%씩 확인")
                        s_bubble_attach_len = gr.Slider(0.3, 1.0, value=c.BUBBLE_ATTACHMENT_MIN_LENGTH_RATIO, step=0.05,
                                                        label="말풍선 붙음 판정 최소 세로선 길이",
                                                        info="말풍선 높이 대비 이 비율 이상 이어진 세로선이 있어야 '테두리에 붙었다'고 판정")

                    with gr.Accordion("원본 글자 지우기", open=False):
                        s_inpaint_ctx = gr.Number(value=c.INPAINT_CONTEXT_PADDING, label="주변 참조 영역 (px)", precision=0,
                                                  info="글자 주변을 얼마나 넓게 같이 보고 지울지. 넓을수록 자연스럽지만 느려짐")
                        s_inpaint_mask = gr.Number(value=c.INPAINT_MASK_PADDING, label="지울 영역 확장 (px)", precision=0,
                                                   info="글자 박스보다 얼마나 더 넓게 지울지. 지우고 나서 글자 잔상이 보이면 늘리세요")

                    with gr.Accordion("글자 식자", open=False):
                        s_vert = gr.Checkbox(value=c.ENABLE_VERTICAL_TEXT, label="세로쓰기 자동 감지",
                                             info="세로로 긴 글자 영역을 감지하면 세로쓰기로 식자")
                        s_vert_thresh = gr.Number(value=c.VERTICAL_TEXT_THRESHOLD, label="세로쓰기 판정 비율", precision=0,
                                                  info="글자 영역의 높이가 너비의 몇 배 이상이면 세로쓰기로 판정")
                        s_vert_force = gr.Slider(2.0, 20.0, value=c.VERTICAL_FORCE_ASPECT_RATIO, step=0.5,
                                                 label="세로쓰기 강제 비율",
                                                 info="글자 영역이 이 비율 이상 세로로 길면 다른 조건 모두 무시하고 무조건 세로쓰기")
                        s_min_rot = gr.Number(value=c.MIN_ROTATION_ANGLE, label="회전 무시 각도 (도)", precision=0,
                                              info="감지된 글자 기울기가 이 각도 이하면 회전하지 않고 똑바로 씀")
                        s_shrink = gr.Slider(0.1, 1.0, value=c.FONT_SHRINK_THRESHOLD_RATIO, step=0.05,
                                             label="가로쓰기 포기 기준",
                                             info="가로쓰기로 맞추다가 글씨가 원본 예측 크기의 이 비율 미만으로 작아지면, 가로쓰기를 포기하고 세로쓰기로 다시 시도")
                        s_min_readable = gr.Number(value=c.MIN_READABLE_TEXT_SIZE, label="읽기 편한 최소 크기 (px)", precision=0,
                                                   info="자동 세로쓰기 판정 시, 원본 글씨 예측 크기가 이보다 작으면 세로쓰기 대신 가로 유지 (강제 비율은 영향 X)")
                        s_min_font = gr.Number(value=c.MIN_FONT_SIZE, label="최소 글씨 크기 (px)", precision=0,
                                               info="아무리 좁아도 이 크기보다는 작게 쓰지 않음")
                        s_max_font = gr.Number(value=c.MAX_FONT_SIZE, label="최대 글씨 크기 (px)", precision=0,
                                               info="아무리 넓어도 이 크기보다는 크게 쓰지 않음")
                        s_fill = gr.Slider(0.1, 1.0, value=c.FONT_AREA_FILL_RATIO, step=0.05,
                                           label="글자 최소 채움 비율",
                                           info="글자가 영역 면적의 이 비율 이하만 차지하면 글씨를 자동으로 키움")

                    with gr.Accordion("글씨체 모델 동작", open=False):
                        s_size_mode = gr.Radio(
                            choices=["off", "light", "strong"],
                            value=c.FONT_SIZE_CORRECTION_MODE,
                            label="글씨 크기 자동 보정",
                            info="off = 모델 예측 크기 그대로 사용 / light = 실제 글자 비율 살짝 참고 / strong = 비율+획 두께 모두 반영 (권장)",
                        )
                        s_size_tol = gr.Slider(
                            0.05, 0.5, value=c.MODEL_FONT_SIZE_TOLERANCE, step=0.05,
                            label="모델 크기 허용 오차 (±비율)",
                            info="모델 예측 대비 ±몇 %까지 조절을 허용할지. 0.2 = 예측값의 80~120%",
                        )
                        s_fallback_mode = gr.Radio(
                            choices=["off", "loose", "strict"],
                            value=c.FONT_STYLE_FALLBACK_MODE,
                            label="확신 없을 때 기본 글씨체로 되돌리기",
                            info="off = 항상 모델 판정 사용 / loose = 확신이 아주 낮을 때만 기본으로 / strict = 엄격하게 되돌림 (권장)",
                        )
                        s_tta_mode = gr.Radio(
                            choices=["off", "fast", "accurate"],
                            value=c.FONT_MODEL_TTA_MODE,
                            label="글씨체 판정 정확도 모드",
                            info="off = 한 번만 판정 (빠름) / fast = 2번 평균 / accurate = 3번 평균 (권장, 약간 느림)",
                        )

                    with gr.Accordion("폰트 매핑", open=False):
                        gr.Markdown("각 스타일에 사용할 폰트를 선택하세요.")
                        _font_files = _get_font_list()
                        s_font_selects = {}
                        for style in ["standard", "shouting", "cute", "narration", "handwriting", "pop", "angry", "scared", "embarrassment"]:
                            current = os.path.basename(c.FONT_MAP.get(style, ""))
                            s_font_selects[style] = gr.Dropdown(
                                choices=_font_files, value=current, label=style,
                                info=f"현재: {current}"
                            )

                    with gr.Accordion("말풍선 밖 글자 (효과음·나레이션)", open=False):
                        s_ff_padding = gr.Slider(0.0, 0.3, value=c.FREEFORM_PADDING_RATIO, step=0.01,
                                                 label="글자 박스 여백 비율",
                                                 info="말풍선 밖 글자 박스 폭 대비 좌우 여백. 0.05 = 양쪽 5%씩 비워둠")
                        s_stroke_w = gr.Number(value=c.FREEFORM_STROKE_WIDTH, label="외곽선 두께 (px)", precision=0,
                                               info="말풍선 밖 글자의 외곽선 두께. 배경에 잘 보이도록 외곽선을 더해줌")
                        s_ff_font_color = gr.ColorPicker(value=_rgb_to_hex(c.FREEFORM_FONT_COLOR),
                                                         label="글자색",
                                                         info="말풍선 밖 글자 본체 색상")
                        s_ff_stroke_color = gr.ColorPicker(value=_rgb_to_hex(c.FREEFORM_STROKE_COLOR),
                                                           label="외곽선 색",
                                                           info="말풍선 밖 글자 테두리 색상")
                        s_ff_attach_px = gr.Number(value=c.FREEFORM_ATTACHMENT_SEARCH_PX, label="테두리 붙음 확인 범위 (px)", precision=0,
                                                   info="글자 박스 좌/우로 이 거리까지 컷 테두리가 있는지 확인해서 그쪽으로 정렬. 페이지 가장자리면 볼 수 있는 만큼만 봄")
                        s_ff_attach_len = gr.Slider(0.3, 1.0, value=c.FREEFORM_ATTACHMENT_MIN_LENGTH_RATIO, step=0.05,
                                                    label="붙음 판정 최소 세로선 길이",
                                                    info="글자 박스 높이 대비 이 비율 이상 이어진 세로선이 있어야 '테두리에 붙었다'고 판정")
                        s_ff_attach_margin = gr.Number(value=c.FREEFORM_ATTACHMENT_TEXT_MARGIN, label="붙음 정렬 여백 (px)", precision=0,
                                                       info="테두리에 붙은 것으로 판정됐을 때, 번역 글자를 테두리 방향으로 몇 픽셀 더 당길지")
                        s_ff_style_conf = gr.Slider(0.3, 1.0, value=c.FREEFORM_STYLE_MIN_CONFIDENCE, step=0.05,
                                                    label="글씨체 판정 확신도 임계값",
                                                    info="말풍선 밖 글자는 대부분 나레이션/효과음이라, 모델이 이 확신도 미만으로 글씨체를 집어내면 자동으로 나레이션 글씨체로 처리")

                    with gr.Accordion("디버그", open=False):
                        s_draw_debug = gr.Checkbox(value=c.DRAW_DEBUG_BOXES, label="탐지 영역 시각화",
                                                   info="결과 이미지에 말풍선(빨강), 텍스트(초록), 말풍선 밖 텍스트(파랑) 박스를 표시")

                    save_settings_btn = gr.Button("설정 저장", variant="primary")
                    settings_result = gr.Textbox(label="결과", interactive=False)

                    save_settings_btn.click(
                        fn=_save_all_settings,
                        inputs=[
                            s_api_key, s_gemini_model, s_prompt,
                            s_input_dir, s_output_dir,
                            s_model_path, s_font_appearance, s_font_size, s_font_style,
                            s_yolo, s_text_merge_iou,
                            s_trans_batch, s_ocr_batch, s_inpaint_batch, s_font_model_batch,
                            s_ocr_beams, s_ocr_local, s_ocr_warmup,
                            s_pass1_workers, s_pass1_cache, s_pass2_microbatch, s_pass2_workers, s_pass2_cache,
                            s_bubble_pad, s_bubble_edge, s_attached_margin,
                            s_bubble_attach_edge, s_bubble_attach_len,
                            s_inpaint_ctx, s_inpaint_mask,
                            s_vert, s_vert_thresh, s_vert_force, s_min_rot,
                            s_shrink, s_min_font, s_max_font, s_min_readable, s_fill,
                            s_size_mode, s_size_tol, s_fallback_mode, s_tta_mode,
                            s_font_selects["standard"], s_font_selects["shouting"], s_font_selects["cute"],
                            s_font_selects["narration"], s_font_selects["handwriting"], s_font_selects["pop"],
                            s_font_selects["angry"], s_font_selects["scared"], s_font_selects["embarrassment"],
                            s_ff_padding, s_stroke_w,
                            s_ff_font_color, s_ff_stroke_color,
                            s_ff_attach_px, s_ff_attach_len, s_ff_attach_margin,
                            s_ff_style_conf,
                            s_draw_debug,
                        ],
                        outputs=[settings_result],
                    )

        # 초기 설정 → 메인 전환
        save_setup_btn.click(
            fn=_save_and_initialize,
            inputs=[api_key_setup, model_setup, input_setup, output_setup],
            outputs=[setup_status, setup_section, main_section],
        )

    demo.queue(max_size=1)
    return demo
