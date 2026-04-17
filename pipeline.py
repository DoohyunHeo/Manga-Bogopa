import glob
import logging
import os

import cv2

from src import config, model_loader
from src.checkpoint import CheckpointManager
from src.pass1_stage import Pass1Stage
from src.pass2_stage import Pass2Stage
from src.progress import ProgressCallback, ProgressEvent, PipelinePhase, noop_callback

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

    def run(self):
        """전체 만화 번역 및 식자 프로세스를 실행합니다."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
        logger.info(f"Using device: {config.DEVICE}")
        os.makedirs(self.output_dir, exist_ok=True)

        image_paths = sorted(glob.glob(os.path.join(config.INPUT_DIR, "*")))
        if not image_paths:
            logger.info(f"'{config.INPUT_DIR}' 폴더에 이미지가 없습니다.")
            return

        ckpt = None
        force_fresh_pass1 = self.pass1_stage.has_full_output_image_set(image_paths)
        if self.enable_checkpoint:
            ckpt = CheckpointManager(self.output_dir, config.INPUT_DIR)
            ckpt.load_or_create(len(image_paths))
            if force_fresh_pass1:
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

        pages_to_process = ckpt.get_pass2_remaining_pages(all_page_data) if ckpt else all_page_data
        if pages_to_process:
            self._inpaint_and_draw_streaming(pages_to_process, image_paths, ckpt)
        else:
            logger.info("Pass 2 체크포인트가 이미 완료되어 렌더링을 건너뜁니다.")

        if ckpt:
            ckpt.mark_complete()

        self.callback(ProgressEvent(PipelinePhase.COMPLETE, 1, 1, "모든 프로세스 완료"))
        logger.info("모든 프로세스 완료.")

    def _ensure_inpainting_model(self):
        """Pass 2 렌더링에 필요한 Inpainting 모델만 로드합니다."""
        if "inpainting" in self.models:
            return
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 0, 1, "Inpainting 모델 로딩 중..."))
        self.models.update(model_loader.load_inpainting_model())
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 1, 1, "Inpainting 모델 로딩 완료"))

    def _build_pass2_service(self) -> Pass2Stage:
        """Pass 2 실행 책임을 전달하는 서비스를 구성합니다."""
        return Pass2Stage(
            models=self.models,
            output_dir=self.output_dir,
            progress_callback=self.callback,
            debug_draw_boxes=config.DRAW_DEBUG_BOXES,
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
