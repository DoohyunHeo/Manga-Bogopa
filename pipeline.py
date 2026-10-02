import logging
import os
from enum import StrEnum

import torch

from src import config, fit_translation, model_files, model_loader
from src.checkpoint import CheckpointManager
from src.font_relative import apply_book_styles, load_or_measure
from src.glossary import GLOSSARY_FILE, Glossary
from src.pass1_stage import Pass1Stage
from src.pass2_stage import Pass2Stage
from src.progress import EventLevel, ProgressCallback, ProgressEvent, PipelinePhase, StageClock, noop_callback
from src.text_filters import dialogue_size, split_logo_parts
from src.utils import list_page_images

logger = logging.getLogger(__name__)


class RunMode(StrEnum):
    """한 번의 실행 방식."""
    RESUME = "resume"        # 이어하기: 끝난 페이지는 건너뛰고 남은 것만 (끝난 책이면 바로 끝남)
    FRESH = "fresh"          # 새로 번역: 기존 대사집은 백업해 두고 처음부터
    RETYPESET = "retypeset"  # 식자만 다시: 저장된 번역으로 지우기·식자만 (번역 요청·1단계 모델 없음)


# 책 단위 글씨체 판정(apply_book_styles)이 대사에 쓰는 값 — 의심 대사 확인이 번역할지를 바꾼 쪽은 판정 전 값으로 돌려
# 다시 정한다
_STYLE_FIELDS = ("font_size", "font_char_ratio", "font_style", "font_style_raw", "font_style_scores", "font_style_reason")


def _typeset_view_keeping_logos(page):
    """번역된 대사만 남긴 사본 — 일부만 번역된 도안 로고 조각은 빼서 로고 전체를 원문으로 둔다."""
    view = page.typeset_view()
    logo_parts = split_logo_parts(page, dialogue_size(page.speech_bubbles))
    if logo_parts:
        view.freeform_texts = [t for t in view.freeform_texts if id(t) not in logo_parts]
        logger.info("%s: 도안 로고 조각 %d개를 원문으로 둡니다 (쪼개져 일부만 번역됐거나 라틴 대문자 제목)", page.source_page, len(logo_parts))
    # 원문으로 둘 글자 자리 — 지우기가 번역된 글자 곁 조각을 더 지울 때 이 자리는 건드리지 않는다 (대사집에는 안 씀)
    kept = {id(t) for t in view.freeform_texts} | {id(b.text_element) for b in view.speech_bubbles}
    view.untouched_boxes = [e.text_box for e in page.text_elements() if id(e) not in kept]
    return view


class MangaTranslationPipeline:
    def __init__(self, progress_callback: ProgressCallback = None):
        """파이프라인을 초기화합니다. 모델은 실행 시점에 지연 로드합니다."""
        self.callback = progress_callback or noop_callback
        # 웹은 실행마다 이 값을 바꿔 끼운 뒤 run()을 부른다
        self.run_mode = RunMode.RESUME
        self.output_dir = config.OUTPUT_DIR
        self.models = {}
        self.clock = StageClock()  # 실행마다 새로 만든다
        # UI가 실행 중 중단 신호를 보낼 수 있도록 교체 가능한 콜러블로 둔다.
        # 각 단계는 배치/페이지 경계에서만 확인하므로 체크포인트가 항상 유효하다.
        self.should_stop = None
        # 식자만 다시에서 이 쪽들만 다시 쓴다 (웹의 번역 고치기 — 쪽 이름 집합, 비우면 책 전체)
        self.only_pages = None
        self.pass1_stage = Pass1Stage(
            models=self.models,
            output_dir=self.output_dir,
            # 웹이 실행마다 self.callback을 바꿔 끼우므로 호출 시점의 것으로 넘긴다
            progress_callback=lambda event: self.callback(event),
            should_stop=self._stop_requested,
        )
        self.pass1_stage.while_translating = self._prepare_typesetting

    def _stop_requested(self) -> bool:
        return bool(self.should_stop and self.should_stop())

    def run(self, mode=None):
        """전체 만화 번역 및 식자 프로세스를 실행합니다.

        mode: RunMode (비우면 self.run_mode).
        """
        mode = RunMode(mode or self.run_mode)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
        logger.info(f"Using device: {config.DEVICE} / 실행 방식: {mode.value}")
        self.clock = StageClock()
        self.pass1_stage.clock = self.clock
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        os.makedirs(self.output_dir, exist_ok=True)
        # UI에서 실행 중 출력 폴더를 바꿔도 Pass 1 산출물(JSON)이 같은 폴더로 가도록 동기화
        self.pass1_stage.output_dir = self.output_dir

        image_paths = list_page_images(config.INPUT_DIR)
        if not image_paths:
            logger.info(f"'{config.INPUT_DIR}' 폴더에 이미지가 없습니다.")
            return

        # 릴리즈로 배포하는 모델 파일은 없으면 여기서 받는다 — 처음 실행하는 사람이 손으로 받지 않게
        if not model_files.ensure_model_files(self.callback, self._stop_requested):
            return  # 받는 중에 멈춰 달라고 했다

        ckpt = CheckpointManager(self.output_dir, config.INPUT_DIR)
        ckpt.load_or_create(len(image_paths))
        changed = ckpt.sync_input_fingerprints(image_paths)
        if changed and mode != RunMode.FRESH:
            self.callback(ProgressEvent(
                PipelinePhase.LOADING_MODELS, 0, 0,
                f"내용이 바뀐 입력 {len(changed)}페이지는 이전 결과를 쓰지 않고 새로 처리합니다 "
                f"(이전 대사집은 백업해 두었습니다)",
                level=EventLevel.WARNING, extras={"changed_pages": changed},
            ))

        only_pages = set(self.only_pages or ()) if mode == RunMode.RETYPESET else set()
        if mode == RunMode.RETYPESET:
            logger.info("식자만 다시: 저장된 번역으로 지우기·식자만 합니다 (번역 요청 없음)%s.",
                        f" — {len(only_pages)}쪽만" if only_pages else "")
            ckpt.reset_pass2(only_pages or None)
            all_page_data = self.pass1_stage._sort_page_data_by_input_order(ckpt.load_pass1_data(), image_paths)
            if not all_page_data:
                message = ("저장된 번역이 없어 식자만 다시 할 수 없습니다 — "
                           "'이어하기'나 '새로 번역'으로 먼저 번역해 주세요")
                logger.warning(message)
                self.callback(ProgressEvent(PipelinePhase.COMPLETE, 0, 0, message, level=EventLevel.ERROR))
                return
            book_pages = all_page_data  # 글씨체 판정(굵기 기준)은 고른 쪽만이 아니라 늘 책 전체로 한다
            if only_pages:
                all_page_data = [page for page in all_page_data if page.source_page in only_pages]
            missing = 0 if only_pages else len(image_paths) - len(all_page_data)
            if missing:
                self.callback(ProgressEvent(
                    PipelinePhase.TRANSLATION, 0, len(image_paths),
                    f"번역이 저장되지 않은 {missing}페이지는 건너뜁니다 — '이어하기'로 번역할 수 있습니다",
                    level=EventLevel.WARNING,
                ))
        else:
            if mode == RunMode.FRESH:
                logger.info("새로 번역: 기존 대사집은 백업하고 처음부터 시작합니다.")
                ckpt.reset_for_new_run(clear_json=True)
            all_page_data = self.pass1_stage.run(image_paths, ckpt=ckpt)
            book_pages = all_page_data
        translating = mode != RunMode.RETYPESET  # 번역한 실행이면 뒤로 보낸 의심 대사 확인을 끝내고 나가야 한다
        if not all_page_data:
            if translating:
                self._finish_check(book_pages or [], ckpt)
            logger.info("처리할 데이터가 없어 파이프라인을 종료합니다.")
            return

        if self._stop_requested():
            if translating:
                self._finish_check(book_pages, ckpt)
            self._emit_stopped()
            return

        # 번역이 덜 된 페이지(실패·대기 대사가 남은 페이지)는 식자하지 않는다 —
        # 반쯤 번역된 페이지를 내보내지 않도록. 체크포인트도 미완료로 남아 재실행 시
        # 남은 대사만 다시 요청한다.
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
                f"번역 실패 {len(names)}페이지는 식자하지 않고 남겨둠 — 다시 실행하면 남은 대사만 다시 요청합니다",
                level=EventLevel.WARNING,
                extras={"untranslated_pages": names},
            ))
            untranslated_names = set(names)
            all_page_data = [p for p in all_page_data if p.source_page not in untranslated_names]
            if not all_page_data:
                if translating:
                    self._finish_check(book_pages, ckpt)
                logger.warning("번역된 페이지가 없어 식자 단계를 건너뜁니다.")
                return

        # 지우기·식자는 번역된 대사만 다룬다 — 번역 제외로 답한 대사는 원문을 그대로 둔다
        all_page_data = [_typeset_view_keeping_logos(page) for page in all_page_data]
        styles = {id(e): tuple(getattr(e, name) for name in _STYLE_FIELDS)
                  for page in book_pages for e in page.text_elements()}
        # 글씨체는 책 단위로 정한다 — 모양 6분류 + 책 기준 굵기 (출력 폴더 font_style.json, 대사집에는 안 씀)
        apply_book_styles(book_pages, all_page_data, self.output_dir, config.INPUT_DIR, self.callback)
        early_names = set()
        if translating:
            # 의심 대사 확인은 Pass 1 끝에서 뒤로 보내 두었다 — 짧은 번역(켰을 때)과 함께 돌고, 답을 기다리는 동안 확인할
            # 대사가 없는 쪽을 먼저 지우고 쓴 뒤 여기서 답을 채운다
            if config.ENABLE_FIT_TRANSLATION:
                self._fit_translations(book_pages, all_page_data, ckpt)
            early_names = self._typeset_while_checking(book_pages, all_page_data, image_paths, ckpt)
            all_page_data, changed_pages = self._finish_check(book_pages, ckpt, all_page_data, styles)
            # 확인·인명 표기 맞추기로 대사가 바뀐 쪽은 다시 쓴다 — 이번에 먼저 쓴 쪽이든 지난 실행에서 쓴 쪽이든
            redo = {name for name in changed_pages if ckpt.is_pass2_done([name])}
            if redo:
                logger.info("의심 대사 확인으로 대사가 바뀐 %d쪽을 다시 씁니다: %s", len(redo), sorted(redo)[:5])
                ckpt.reset_pass2(redo)
                early_names -= redo
        pages_to_process = ckpt.get_pass2_remaining_pages(all_page_data)
        remaining_names = {page.source_page for page in pages_to_process}
        skipped_pages = [page for page in all_page_data
                         if page.source_page not in remaining_names and page.source_page not in early_names]

        if skipped_pages:
            self._emit_skipped_pages(skipped_pages, all_page_data)

        failed_pages = []
        if pages_to_process:
            failed_pages = self._inpaint_and_draw_streaming(pages_to_process, image_paths, ckpt,
                                                            done_before=len(early_names))
        if pages_to_process or early_names:
            self._emit_stage_times(len(pages_to_process) + len(early_names))
        else:
            logger.info("Pass 2 체크포인트가 이미 완료되어 렌더링을 건너뜁니다.")

        if self._stop_requested():
            self._emit_stopped()
            return

        if only_pages:
            # 고른 쪽만 다시 쓴 경우 — 책이 다 끝났는지는 식자 기록으로 본다
            names = [os.path.basename(path) for path in image_paths]
            if not failed_pages and ckpt.is_pass2_done(names):
                ckpt.mark_complete()
            if failed_pages:
                self.callback(ProgressEvent(
                    PipelinePhase.COMPLETE, 1, 1,
                    f"고른 {len(only_pages)}쪽 가운데 {len(failed_pages)}쪽을 만들지 못했습니다 — 다시 실행하면 그 쪽만 처리합니다",
                    level=EventLevel.WARNING,
                ))
            else:
                self.callback(ProgressEvent(PipelinePhase.COMPLETE, 1, 1, f"고른 {len(only_pages)}쪽을 다시 썼습니다"))
            return

        # 모든 입력이 번역·식자·저장까지 끝났을 때만 완료로 기록한다
        complete = not untranslated_pages and not failed_pages and len(all_page_data) == len(image_paths)
        if complete:
            ckpt.mark_complete()
            self.callback(ProgressEvent(PipelinePhase.COMPLETE, 1, 1, "모든 프로세스 완료"))
            logger.info("모든 프로세스 완료.")
        else:
            self.callback(ProgressEvent(
                PipelinePhase.COMPLETE, 1, 1,
                "끝나지 않은 페이지가 있습니다 — '이어하기'로 다시 실행하면 남은 것만 처리합니다",
                level=EventLevel.WARNING,
            ))
            logger.info("일부 페이지 미완료로 종료 (다음 실행에서 이어서 처리).")

    def _fit_translations(self, book_pages, typeset_pages, ckpt):
        """배치가 어려운 말풍선만 짧은 번역을 한 번 더 받는다 (설정 ENABLE_FIT_TRANSLATION, 책마다 한 번). 실패하면 받은 번역 그대로."""
        if self._stop_requested():
            return
        self.callback(ProgressEvent(PipelinePhase.TRANSLATION, 0, 0, "좁은 말풍선에 맞춰 짧은 번역을 받는 중...",
                                    extras={"fit_started": True}))
        try:
            with self.clock.measure("짧은 번역"):
                record = fit_translation.shorten_hard_bubbles(
                    book_pages, typeset_pages, self.output_dir, config.INPUT_DIR, ckpt.json_path,
                    model_loader.load_translator_session, Glossary(os.path.join(self.output_dir, GLOSSARY_FILE)),
                )
        except Exception as exc:  # noqa: BLE001 — 다듬기일 뿐, 실패해도 받은 번역으로 식자한다
            logger.warning(f"짧은 번역 단계를 건너뜁니다: {exc}")
            return
        if record and record["adopted"]:
            self.callback(ProgressEvent(
                PipelinePhase.TRANSLATION, 0, 0,
                f"좁은 말풍선 {record['requested']}곳 중 {len(record['adopted'])}곳을 짧은 번역으로 바꿨습니다",
                extras={"fit_adopted": len(record["adopted"])},
            ))

    def _typeset_while_checking(self, book_pages, typeset_pages, image_paths, ckpt):
        """의심 대사 확인 답을 기다리는 동안, 다시 묻는 대사가 없는 쪽을 먼저 지우고 쓴다 — 끝내고 저장한 쪽 이름.

        확인이 없거나 모든 쪽에 걸렸으면 아무것도 하지 않는다. 멈춤·저장 실패로 못 끝낸 쪽은 체크포인트에 남아 나중
        식자가 다시 한다. 답을 채운 뒤 대사가 바뀐 쪽(확인이 고친 인명 표기를 책 전체에 맞춘 쪽 포함)은 부르는 쪽이 다시
        쓴다. 확인 답을 기다리는 시간과 지우기·식자 시간이 겹친다."""
        held = self.pass1_stage.check_pages(book_pages)
        if held is None or self._stop_requested():
            return set()
        remaining = ckpt.get_pass2_remaining_pages(typeset_pages)
        early = [page for page in remaining if page.source_page not in held]
        if not early:
            return set()
        logger.info("의심 대사 확인 답을 기다리는 동안 확인할 대사가 없는 %d쪽을 먼저 씁니다 (나중에 쓸 쪽 %d)",
                    len(early), len(remaining) - len(early))
        self._inpaint_and_draw_streaming(early, image_paths, ckpt, total=len(remaining))
        return {page.source_page for page in early if ckpt.is_pass2_done([page.source_page])}

    def _finish_check(self, book_pages, ckpt, typeset_pages=None, styles=None):
        """뒤에서 돈 의심 대사 확인의 답을 채우고 바뀐 번역만 대사집에 쓴다 — (식자용 쪽, 대사가 바뀐 쪽 이름).

        식자용 쪽(typeset_pages)을 주면, 확인이 번역할지를 바꾼(빼거나 되살린) 쪽만 다시 만들어 글씨체 판정 전 값(styles)에서
        다시 판정한 목록을 돌려준다. 대사가 바뀐 쪽에는 번역할지가 바뀐 쪽도 든다."""
        changed, flipped = self.pass1_stage.finish_check(book_pages)
        changed_ids = {id(element) for element in changed}
        changed_pages = set(flipped) | {page.source_page for page in book_pages
                                        if any(id(element) in changed_ids for element in page.text_elements())}
        if changed:
            try:
                fit_translation.save_translations(book_pages, changed, ckpt.json_path)
            except Exception as exc:  # noqa: BLE001 — 저장을 못 해도 이번 식자는 고친 번역으로 한다
                logger.warning(f"의심 대사 확인이 고친 번역을 대사집에 쓰지 못했습니다: {exc}")
        if not typeset_pages or not flipped:
            return typeset_pages, changed_pages
        pages = {page.source_page: page for page in book_pages}
        rebuilt = []
        for i, view in enumerate(typeset_pages):
            page = pages.get(view.source_page)
            if view.source_page not in flipped or page is None:
                continue
            for element in page.text_elements():
                for name, value in zip(_STYLE_FIELDS, styles.get(id(element), ())):
                    setattr(element, name, value)
            typeset_pages[i] = _typeset_view_keeping_logos(page)
            rebuilt.append(typeset_pages[i])
        if rebuilt:
            logger.info("의심 대사 확인이 번역할지를 바꾼 %d쪽의 식자할 대사를 다시 골랐습니다", len(rebuilt))
            apply_book_styles(book_pages, rebuilt, self.output_dir, config.INPUT_DIR, self.callback)
        return typeset_pages, changed_pages

    def _emit_stage_times(self, pages):
        """이번 실행에서 단계마다 걸린 시간을 한 줄로 알린다."""
        message = self.clock.summary(pages)
        logger.info(message)
        self.callback(ProgressEvent(PipelinePhase.COMPLETE, 0, 0, message, extras=dict(self.clock.seconds)))

    def _emit_stopped(self):
        """중단 종료 — 체크포인트는 미완료로 남아 다음 실행에서 이어진다."""
        logger.info("사용자 요청으로 중단되었습니다 (체크포인트 유지).")
        self.callback(ProgressEvent(
            PipelinePhase.COMPLETE, 0, 1,
            "중단됨 — 진행된 페이지까지 저장되었고, 다시 실행하면 이어서 진행합니다",
            level=EventLevel.WARNING,
        ))

    def _emit_skipped_pages(self, skipped_pages, all_page_data):
        """체크포인트로 Pass 2를 건너뛴 페이지도 이전 결과를 다시 쓴 쪽으로 알린다 (결과 그림이 디스크에 있는 쪽만).

        디스크의 이전 결과 이미지를 그대로 쓴다 (사용자가 outputs를
        수동으로 지우지 않은 이상 파이프라인은 그 상태를 유지).
        """
        total = len(all_page_data)
        for page_data in skipped_pages:
            output_path = os.path.join(self.output_dir, page_data.source_page)
            if not os.path.exists(output_path):
                continue
            self.callback(ProgressEvent(
                PipelinePhase.PASS2_PAGE, all_page_data.index(page_data) + 1, total,
                f"{page_data.source_page} (이전 결과 재사용)", page_name=page_data.source_page,
                extras={"restored_from_disk": True},
            ))

    def _prepare_typesetting(self, book_pages):
        """쪽을 다 읽고 번역 답만 기다리는 동안 식자 준비 — 책의 글씨체를 판정하고 지우기 모델을 읽는다.

        글씨체 판정의 시작·끝은 알린다 (번역이 남아 있는 동안 화면은 단계를 바꾸지 않고 문구만 바꾼다).
        지우기 모델은 알림 없이 읽는다. 쓰기 단계는 이 결과(font_style.json, 모델)를 그대로 쓴다.
        """
        load_or_measure(book_pages, self.output_dir, config.INPUT_DIR, self.callback)
        if "inpainting" not in self.models:
            self.models.update(model_loader.load_inpainting_model())

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
            ensure_inpainting_model=self._ensure_inpainting_model,
            should_stop=self._stop_requested,
            clock=self.clock,
        )

    def _inpaint_and_draw_streaming(self, pages_to_process, image_paths, ckpt=None, done_before=0, total=None):
        """Pass 2 실행을 전용 서비스에 위임합니다. 만들지 못한 페이지 이름(원본 없음·읽기 실패·저장 실패)을 돌려준다.

        두 번에 나눠 쓸 때는 앞서 끝낸 쪽 수(done_before)와 전체 쪽 수(total)로 진행 번호를 이어 센다."""
        total = max(int(total or 0), done_before + len(pages_to_process))
        # 첫 쪽이 끝나기까지(모델 준비 + 첫 묶음 지우기) 시간이 걸리므로 시작을 먼저 알린다
        self.callback(ProgressEvent(PipelinePhase.PASS2_PAGE, done_before, total,
                                    f"{len(pages_to_process)}쪽 지우고 쓰기를 시작합니다"))
        return self._build_pass2_service().run(pages_to_process, image_paths, ckpt, done_before=done_before, total=total)
