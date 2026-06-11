import glob
import logging
import os

import cv2

from src import config, model_loader
from src.checkpoint import CheckpointManager
from src.pass1_stage import Pass1Stage
from src.pass2_stage import Pass2Stage
from src.progress import EventLevel, ProgressCallback, ProgressEvent, PipelinePhase, noop_callback

logger = logging.getLogger(__name__)


class MangaTranslationPipeline:
    def __init__(self, progress_callback: ProgressCallback = None, enable_checkpoint: bool = True):
        """파이프라인을 초기화합니다. 모델은 실행 시점에 지연 로드합니다."""
        self.callback = progress_callback or noop_callback
        self.enable_checkpoint = enable_checkpoint
        self.output_dir = config.OUTPUT_DIR
        self.models = {}
        self.pass1_stage = Pass1Stage(
            models=self.models,
            output_dir=self.output_dir,
            progress_callback=self.callback,
        )

    def run(self, rerender_only: bool = False):
        """전체 만화 번역 및 식자 프로세스를 실행합니다.

        rerender_only=True: 저장된 번역(대사집)을 그대로 재사용하고 식자만
        다시 한다 — 식자 변경 검증용. Gemini API를 호출하지 않는다.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
        logger.info(f"Using device: {config.DEVICE}")
        os.makedirs(self.output_dir, exist_ok=True)
        # UI에서 실행 중 출력 폴더를 바꿔도 Pass 1 산출물(JSON)이 같은 폴더로 가도록 동기화
        self.pass1_stage.output_dir = self.output_dir

        image_paths = sorted(glob.glob(os.path.join(config.INPUT_DIR, "*")))
        if not image_paths:
            logger.info(f"'{config.INPUT_DIR}' 폴더에 이미지가 없습니다.")
            return

        ckpt = None
        force_fresh_pass1 = (not rerender_only) and self.pass1_stage.has_full_output_image_set(image_paths)
        if self.enable_checkpoint or rerender_only:
            ckpt = CheckpointManager(self.output_dir, config.INPUT_DIR)
            ckpt.load_or_create(len(image_paths))
            if rerender_only:
                logger.info("재식자 모드: 저장된 번역을 재사용하고 식자만 다시 합니다 (API 호출 없음).")
                ckpt.reset_pass2()
            elif force_fresh_pass1:
                logger.info(
                    "출력 폴더에 입력 전체와 동일한 결과 이미지가 이미 있어 "
                    "체크포인트를 재사용하지 않고 새 번역을 시작합니다."
                )
                ckpt.reset_for_new_run(clear_json=True)

        all_page_data = self.pass1_stage.run(
            image_paths,
            ckpt=ckpt,
            force_fresh=force_fresh_pass1,
        )
        if not all_page_data:
            logger.info("처리할 데이터가 없어 파이프라인을 종료합니다.")
            return

        # 번역이 안 된 페이지(API 실패로 "번역 불가" 마커만 있는 페이지)는 식자하지
        # 않는다 — 원본을 지우고 마커를 그려 넣는 사고 방지. 체크포인트도 미완료로
        # 남아 재실행 시 번역부터 다시 시도된다.
        untranslated_pages = [
            page for page in all_page_data
            if not self.pass1_stage._is_page_translated(page)
        ]
        if untranslated_pages:
            names = [page.source_page for page in untranslated_pages]
            logger.warning(
                "번역이 완료되지 않은 %d페이지는 식자를 건너뜁니다 (재실행 시 번역 재시도): %s",
                len(names), names[:5],
            )
            self.callback(ProgressEvent(
                PipelinePhase.TRANSLATION, 0, len(all_page_data),
                f"번역 실패 {len(names)}페이지는 식자하지 않고 남겨둠 — 다시 실행하면 번역부터 재시도합니다",
                level=EventLevel.WARNING,
                extras={"untranslated_pages": names},
            ))
            untranslated_names = set(names)
            all_page_data = [p for p in all_page_data if p.source_page not in untranslated_names]
            if not all_page_data:
                logger.warning("번역된 페이지가 없어 식자 단계를 건너뜁니다.")
                return

        pages_to_process = ckpt.get_pass2_remaining_pages(all_page_data) if ckpt else all_page_data
        if ckpt:
            remaining_names = {page.source_page for page in pages_to_process}
            skipped_pages = [page for page in all_page_data if page.source_page not in remaining_names]
        else:
            skipped_pages = []

        if skipped_pages:
            self._emit_skipped_pages(skipped_pages, all_page_data)

        if pages_to_process:
            self._inpaint_and_draw_streaming(pages_to_process, image_paths, ckpt)
        else:
            logger.info("Pass 2 체크포인트가 이미 완료되어 렌더링을 건너뜁니다.")

        if ckpt:
            ckpt.mark_complete()

        self.callback(ProgressEvent(PipelinePhase.COMPLETE, 1, 1, "모든 프로세스 완료"))
        logger.info("모든 프로세스 완료.")

    def _emit_skipped_pages(self, skipped_pages, all_page_data):
        """체크포인트로 Pass 2가 스킵된 페이지도 갤러리에 보이도록 디스크에서 로드해 이벤트 발행.

        디스크의 이전 결과 이미지를 그대로 clean으로 취급한다 (사용자가 outputs를
        수동으로 지우지 않은 이상 파이프라인은 그 상태를 유지).
        """
        total = len(all_page_data)
        for page_data in skipped_pages:
            output_path = os.path.join(self.output_dir, page_data.source_page)
            if not os.path.exists(output_path):
                continue
            image_bgr = cv2.imread(output_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            debug_overlay_rgb = image_rgb.copy()
            self._draw_debug_boxes(debug_overlay_rgb, page_data)

            index = all_page_data.index(page_data) + 1
            self.callback(ProgressEvent(
                PipelinePhase.PASS2_PAGE,
                index,
                total,
                f"{page_data.source_page} (이전 결과 재사용)",
                page_name=page_data.source_page,
                image_rgb=image_rgb,
                extras={
                    "bubbles": len(page_data.speech_bubbles),
                    "freeform": len(page_data.freeform_texts),
                    "debug_image_rgb": debug_overlay_rgb,
                    "restored_from_disk": True,
                },
            ))

    def _ensure_inpainting_model(self):
        """Pass 2 렌더링에 필요한 Inpainting 모델만 로드합니다."""
        if "inpainting" in self.models:
            return
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 0, 1, "Inpainting 모델 로딩 중..."))
        self.models.update(model_loader.load_inpainting_model())
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 1, 1, "Inpainting 모델 로딩 완료"))

    def _build_pass2_service(self) -> Pass2Stage:
        """Pass 2 실행 책임을 전달하는 서비스를 구성합니다.

        debug_box_drawer는 항상 제공하고, 실제 디스크 저장은 클린 이미지만 한다.
        UI 쪽에서 오버레이 표시 여부를 토글한다.
        """
        return Pass2Stage(
            models=self.models,
            output_dir=self.output_dir,
            progress_callback=self.callback,
            debug_box_drawer=self._draw_debug_boxes,
            ensure_inpainting_model=self._ensure_inpainting_model,
        )

    def _inpaint_and_draw_streaming(self, pages_to_process, image_paths, ckpt=None):
        """Pass 2 실행을 전용 서비스에 위임합니다."""
        self._build_pass2_service().run(pages_to_process, image_paths, ckpt)

    def _draw_debug_boxes(self, image_rgb, page_data):
        """디버깅 목적으로 탐지된 모든 박스를 이미지에 그립니다."""
        COLOR_BUBBLE = (255, 0, 0)
        COLOR_TEXT = (0, 255, 0)
        COLOR_FREE_TEXT = (0, 0, 255)

        for bubble in page_data.speech_bubbles:
            x1, y1, x2, y2 = map(int, bubble.bubble_box)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), COLOR_BUBBLE, 2)
            tx1, ty1, tx2, ty2 = map(int, bubble.text_element.text_box)
            cv2.rectangle(image_rgb, (tx1, ty1), (tx2, ty2), COLOR_TEXT, 2)

        for ff_text in page_data.freeform_texts:
            tx1, ty1, tx2, ty2 = map(int, ff_text.text_box)
            cv2.rectangle(image_rgb, (tx1, ty1), (tx2, ty2), COLOR_FREE_TEXT, 2)
