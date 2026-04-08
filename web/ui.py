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

    yield ("설정 저장 완료. 모델을 로딩합니다... (10~30초 소요)", gr.update(visible=True), gr.update(visible=False))

    try:
        app_state.initialize_pipeline()
        yield ("준비 완료!", gr.update(visible=False), gr.update(visible=True))
    except Exception as e:
        yield (f"모델 로딩 실패: {e}", gr.update(visible=True), gr.update(visible=False))


# ---------------------------------------------------------------------------
# 설정 탭 (전체 설정 편집)
# ---------------------------------------------------------------------------

def _save_all_settings(
    api_key, gemini_model, system_prompt,
    input_dir, output_dir,
    yolo_thresh, translation_batch, ocr_batch, inpaint_batch,
    bubble_padding_ratio, bubble_edge_margin,
    inpaint_ctx_padding, inpaint_mask_padding,
    enable_vertical, vertical_threshold, min_rotation,
    font_shrink_ratio, min_font, max_font, font_area_fill,
    freeform_stroke_width,
    save_debug_crops, draw_debug_boxes,
):
    """전체 설정을 config.json에 저장합니다."""
    c = config._config
    c.GEMINI_API_KEY = api_key.strip()
    c.GEMINI_MODEL = gemini_model.strip()
    c.SYSTEM_PROMPT = system_prompt
    c.INPUT_DIR = input_dir.strip()
    c.OUTPUT_DIR = output_dir.strip()
    c.YOLO_CONF_THRESHOLD = yolo_thresh
    c.TRANSLATION_BATCH_SIZE = int(translation_batch)
    c.OCR_BATCH_SIZE = int(ocr_batch)
    c.INPAINT_BATCH_SIZE = int(inpaint_batch)
    c.BUBBLE_PADDING_RATIO = bubble_padding_ratio
    c.BUBBLE_EDGE_SAFE_MARGIN = int(bubble_edge_margin)
    c.INPAINT_CONTEXT_PADDING = int(inpaint_ctx_padding)
    c.INPAINT_MASK_PADDING = int(inpaint_mask_padding)
    c.ENABLE_VERTICAL_TEXT = enable_vertical
    c.VERTICAL_TEXT_THRESHOLD = int(vertical_threshold)
    c.MIN_ROTATION_ANGLE = int(min_rotation)
    c.FONT_SHRINK_THRESHOLD_RATIO = font_shrink_ratio
    c.MIN_FONT_SIZE = int(min_font)
    c.MAX_FONT_SIZE = int(max_font)
    c.FONT_AREA_FILL_RATIO = font_area_fill
    c.FREEFORM_STROKE_WIDTH = int(freeform_stroke_width)
    c.SAVE_DEBUG_CROPS = save_debug_crops
    c.DRAW_DEBUG_BOXES = draw_debug_boxes
    config.save()
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

                    with gr.Accordion("탐지 / 배치", open=False):
                        s_yolo = gr.Slider(0.1, 0.9, value=c.YOLO_CONF_THRESHOLD, step=0.05,
                                           label="YOLO 탐지 신뢰도",
                                           info="말풍선·텍스트 탐지 최소 신뢰도. 낮추면 더 많이 잡히지만 오탐도 늘어남")
                        s_trans_batch = gr.Number(value=c.TRANSLATION_BATCH_SIZE, label="번역 배치 크기", precision=0,
                                                  info="한 번에 Gemini에 보낼 페이지 수. VRAM이 부족하면 줄이세요")
                        s_ocr_batch = gr.Number(value=c.OCR_BATCH_SIZE, label="OCR 배치 크기", precision=0,
                                                info="manga-ocr에 한 번에 넣을 텍스트 크롭 수")
                        s_inpaint_batch = gr.Number(value=c.INPAINT_BATCH_SIZE, label="인페인팅 배치 크기", precision=0,
                                                    info="LaMa에 한 번에 넣을 패치 수. VRAM이 부족하면 줄이세요")

                    with gr.Accordion("말풍선 식자", open=False):
                        s_bubble_pad = gr.Slider(0.0, 0.5, value=c.BUBBLE_PADDING_RATIO, step=0.01,
                                                 label="말풍선 안쪽 여백 비율",
                                                 info="말풍선 폭 대비 좌우 여백. 0.15면 양쪽 15%씩 비워두고 텍스트 배치")
                        s_bubble_edge = gr.Number(value=c.BUBBLE_EDGE_SAFE_MARGIN, label="말풍선 가장자리 안전 거리 (px)", precision=0,
                                                  info="말꼬리가 있는 말풍선에서 텍스트가 가장자리에 너무 붙지 않도록 하는 최소 거리")

                    with gr.Accordion("인페인팅 (텍스트 지우기)", open=False):
                        s_inpaint_ctx = gr.Number(value=c.INPAINT_CONTEXT_PADDING, label="주변 참조 영역 (px)", precision=0,
                                                  info="텍스트 주변을 얼마나 넓게 잘라서 LaMa에 넘길지. 넓을수록 자연스럽지만 느려짐")
                        s_inpaint_mask = gr.Number(value=c.INPAINT_MASK_PADDING, label="마스크 확장 (px)", precision=0,
                                                   info="지울 영역을 텍스트 박스보다 얼마나 더 넓게 잡을지. 글자 잔상이 남으면 늘리세요")

                    with gr.Accordion("텍스트 렌더링", open=False):
                        s_vert = gr.Checkbox(value=c.ENABLE_VERTICAL_TEXT, label="세로쓰기 자동 감지",
                                             info="세로로 긴 텍스트 영역을 감지하면 세로쓰기로 식자")
                        s_vert_thresh = gr.Number(value=c.VERTICAL_TEXT_THRESHOLD, label="세로쓰기 판정 비율", precision=0,
                                                  info="텍스트 영역의 높이/너비가 이 값 이상이면 세로쓰기로 판정")
                        s_min_rot = gr.Number(value=c.MIN_ROTATION_ANGLE, label="회전 무시 각도 (도)", precision=0,
                                              info="예측된 기울기가 이 각도 이하면 회전 없이 수평으로 식자")
                        s_shrink = gr.Slider(0.1, 1.0, value=c.FONT_SHRINK_THRESHOLD_RATIO, step=0.05,
                                             label="강제 줄바꿈 임계 비율",
                                             info="폰트를 줄여도 원래 크기의 이 비율 이하가 되면, 가장 긴 줄을 반으로 쪼개서 재시도")
                        s_min_font = gr.Number(value=c.MIN_FONT_SIZE, label="최소 폰트 크기 (px)", precision=0,
                                               info="이보다 작아지면 폰트 축소를 멈춤")
                        s_max_font = gr.Number(value=c.MAX_FONT_SIZE, label="최대 폰트 크기 (px)", precision=0,
                                               info="이보다 크게는 키우지 않음")
                        s_fill = gr.Slider(0.1, 1.0, value=c.FONT_AREA_FILL_RATIO, step=0.05,
                                           label="텍스트 최소 채움 비율",
                                           info="텍스트가 영역 면적의 이 비율 이하만 차지하면 폰트를 자동으로 키움")

                    with gr.Accordion("말풍선 밖 텍스트", open=False):
                        s_stroke_w = gr.Number(value=c.FREEFORM_STROKE_WIDTH, label="외곽선 두께 (px)", precision=0,
                                               info="말풍선 밖 텍스트(효과음, 나레이션 등)의 글자 외곽선 두께")

                    with gr.Accordion("디버그", open=False):
                        s_save_crops = gr.Checkbox(value=c.SAVE_DEBUG_CROPS, label="텍스트 크롭 이미지 저장",
                                                   info="OCR에 넣은 텍스트 잘라낸 이미지를 debug_crops 폴더에 저장")
                        s_draw_debug = gr.Checkbox(value=c.DRAW_DEBUG_BOXES, label="탐지 영역 시각화",
                                                   info="결과 이미지에 말풍선(빨강), 텍스트(초록), 말풍선 밖 텍스트(파랑) 박스를 표시")

                    save_settings_btn = gr.Button("설정 저장", variant="primary")
                    settings_result = gr.Textbox(label="결과", interactive=False)

                    save_settings_btn.click(
                        fn=_save_all_settings,
                        inputs=[
                            s_api_key, s_gemini_model, s_prompt,
                            s_input_dir, s_output_dir,
                            s_yolo, s_trans_batch, s_ocr_batch, s_inpaint_batch,
                            s_bubble_pad, s_bubble_edge,
                            s_inpaint_ctx, s_inpaint_mask,
                            s_vert, s_vert_thresh, s_min_rot,
                            s_shrink, s_min_font, s_max_font, s_fill,
                            s_stroke_w,
                            s_save_crops, s_draw_debug,
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
