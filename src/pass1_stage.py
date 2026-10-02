import logging
import math
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import cv2

from src import config, extractor, model_loader, translator
from src.data_models import PageData, TranslationStatus
from src.glossary import GLOSSARY_FILE, Glossary
from src.progress import EventLevel, PipelinePhase, ProgressCallback, ProgressEvent, StageClock, noop_callback
from src.utils import read_image_bgr

logger = logging.getLogger(__name__)

_WAIT_TICK_SEC = 2.0


def _even_batches(total: int, limit: int) -> List[int]:
    """total쪽을 limit쪽 이하 묶음으로 고르게 나눈 크기들 — 16쪽은 12+4가 아니라 8+8, 25쪽은 9+8+8.

    묶음은 동시에 번역되므로 가장 큰 묶음의 답을 기다린다 — 12+4로 나누면 12쪽 묶음 답이 늦어 전체가 늦어진다.
    """
    if total <= 0:
        return []
    count = math.ceil(total / max(1, int(limit)))
    base, extra = divmod(total, count)
    return [base + (1 if i < extra else 0) for i in range(count)]


@dataclass
class _TranslationJob:
    """번역 창구에 넘긴 쪽 묶음 하나."""
    future: Future
    pages: List[PageData]
    new: bool  # 이번 실행에서 새로 읽은 쪽 — 번역이 끝나면 대사집에 덧붙인다 (다시 요청한 쪽은 끝에 한꺼번에 저장)


class Pass1Stage:
    """Pass 1 orchestration for load, detect, OCR, translate, and checkpoint reuse."""

    def __init__(self, models, output_dir: str, progress_callback: Optional[ProgressCallback] = None,
                 should_stop=None):
        self.models = models
        self.output_dir = output_dir
        self.callback = progress_callback or noop_callback
        self.should_stop = should_stop or (lambda: False)
        self._pool = None
        self._glossary = None
        self._jobs: List[_TranslationJob] = []
        self._page_numbers = {}
        self._check = None  # 뒤에서 도는 의심 대사 확인 (start_check가 돌려준 것) — finish_check가 채운다
        self._check_started = 0.0
        self.clock = StageClock()  # 파이프라인이 실행마다 바꿔 끼운다
        # 번역 답만 남았을 때 그동안 할 식자 준비 (파이프라인이 넣는다) — 책의 쪽 목록을 받는다
        self.while_translating: Optional[Callable[[List[PageData]], None]] = None

    def run(self, image_paths: Sequence[str], ckpt) -> List[PageData]:
        # 번역 창구와 인명·용어 사전은 실행(책)마다 새로 연다 — 앞 책의 이름이 섞이지 않게
        self._pool = None
        self._glossary = None
        self._jobs = []
        self._check = None
        self._sent = translator.KeyedItems()  # 이번 실행에서 번역을 보낸 대사 — 다 받은 뒤 의심 대사를 다시 묻는다
        self._stats = Counter()
        self._page_numbers = {os.path.basename(path): number for number, path in enumerate(image_paths, start=1)}
        try:
            all_page_data = self._prepare_pass1_data(image_paths, ckpt=ckpt)
            if self._jobs:
                self._emit_waiting()  # 다 읽었다 — 화면이 식자 준비 전에 바로 '번역 기다리는 중'으로 넘어가게
            if self._jobs and all_page_data and self.while_translating and not self.should_stop():
                try:
                    self.while_translating(all_page_data)
                except Exception as exc:  # noqa: BLE001 — 준비는 식자 단계에서 다시 한다
                    logger.warning(f"번역을 기다리는 동안 식자 준비를 하지 못했습니다: {exc}")
            self._collect_translations(ckpt, wait=True)
            if self._glossary is not None:
                self._refresh_by_glossary(all_page_data)
            self._start_check()
        finally:
            if self._check is None:
                self._close_pool()
        if self._check is None:
            self._emit_summary()  # 확인이 돌고 있으면 finish_check가 확인까지 센 뒤에 알린다
        if not all_page_data:
            return []

        json_path = os.path.join(self.output_dir, "translation_data.json")
        # 중단으로 일부만 처리했으면 미완료로 남겨 다음 실행에서 이어간다.
        ckpt.replace_pass1_data(all_page_data, complete=not self.should_stop())
        self.callback(ProgressEvent(PipelinePhase.SAVING_JSON, 1, 1, "JSON 저장 완료"))
        logger.info(f"번역 데이터를 '{json_path}' 파일로 저장했습니다.")
        return all_page_data

    def _ensure_pass1_models(self):
        """Load only the models required for Pass 1."""
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 0, 1, "Pass 1 모델 로딩 중..."))
        if "detection" not in self.models or "ocr" not in self.models:
            self.models.update(model_loader.load_detection_ocr_models())
        self._translation_pool()
        self.callback(ProgressEvent(PipelinePhase.LOADING_MODELS, 1, 1, "Pass 1 모델 로딩 완료"))

    def _translation_pool(self):
        """이번 실행의 번역 창구 (처음 필요할 때 연다). CLI를 못 찾으면 여기서 알린다."""
        if self._pool is None:
            if self._glossary is None:
                self._glossary = Glossary(os.path.join(self.output_dir, GLOSSARY_FILE))
            pool = translator.TranslationPool(
                model_loader.load_translator_session, config.AGY_SESSIONS, glossary=self._glossary,
                callback=self.callback, should_stop=self.should_stop,
            )
            try:
                pool.check()
            except Exception:
                pool.close()
                raise
            self._pool = pool
        return self._pool

    def _prepare_pass1_data(self, image_paths, ckpt):
        """Reuse checkpoints when possible and run Pass 1 only for missing pages."""
        glossary_path = os.path.join(self.output_dir, GLOSSARY_FILE)
        existing_page_data = ckpt.load_pass1_data()
        if existing_page_data:
            existing_page_data = self._sort_page_data_by_input_order(existing_page_data, image_paths)
        expected_pages = {os.path.basename(path) for path in image_paths}
        extracted_names = {page.source_page for page in existing_page_data}
        incomplete_pages = [page for page in existing_page_data if not page.is_translated]

        if extracted_names == expected_pages and not incomplete_pages:
            logger.info(f"완전한 체크포인트 JSON을 재사용합니다: {len(existing_page_data)}페이지")
            ckpt.replace_pass1_data(existing_page_data, complete=True)
            return existing_page_data

        # 이어하기면 지난 실행의 인명·용어 사전을 이어 쓴다 (새로 번역하면 대사집과 함께 새로 만든다)
        self._glossary = Glossary.load(glossary_path) if existing_page_data else Glossary(glossary_path)
        if existing_page_data:
            logger.info(
                f"Pass 1 체크포인트 재개: 추출 끝난 {len(extracted_names)}페이지(번역 남음 "
                f"{len(incomplete_pages)}), 새로 처리 {len(expected_pages - extracted_names)}페이지"
            )
            ckpt.replace_pass1_data(existing_page_data, complete=False)
        if incomplete_pages:
            self._retranslate_pages(incomplete_pages)

        remaining_paths = [
            path for path in image_paths
            if os.path.basename(path) not in extracted_names
        ]
        if not remaining_paths or self.should_stop():
            return existing_page_data

        with self.clock.measure("모델 준비"):
            self._ensure_pass1_models()
        new_page_data = self._extract_and_translate_data(remaining_paths, ckpt)
        return self._sort_page_data_by_input_order(existing_page_data + new_page_data, image_paths)

    def _retranslate_pages(self, pages):
        """추출은 끝났지만 번역이 덜 된 페이지 — OCR을 다시 하지 않고 남은 대사만 다시 요청한다.

        요청은 번역 창구에서 뒤에서 돌고, 그동안 아직 안 읽은 쪽을 읽는다.
        """
        pending = sum(1 for page in pages for element in page.text_elements() if element.needs_translation)
        self.callback(ProgressEvent(
            PipelinePhase.TRANSLATION, 0, len(pages),
            f"지난 실행에서 번역이 안 된 대사 {pending}개를 다시 요청합니다 ({len(pages)}페이지)",
        ))
        start = 0
        for size in _even_batches(len(pages), config.TRANSLATION_BATCH_SIZE):
            if self.should_stop():
                break
            self._send_translation(pages[start:start + size], ckpt=None, new=False)
            start += size

    def _send_translation(self, pages, ckpt, new):
        """쪽 묶음의 대사를 번역 창구에 넘긴다 — 답은 뒤에서 받고, 그동안 다음 쪽을 읽는다."""
        numbers = [self._page_numbers.get(page.source_page, index) for index, page in enumerate(pages, start=1)]
        keyed_items = translator.prepare_items(pages, numbers)
        for page in pages:
            page.image_rgb = None  # 읽는 순서를 정했으니 그림은 더 쓰지 않는다
        if not keyed_items:
            self._finish_pages(pages, ckpt, new)
            return
        pool = self._translation_pool()
        label = translator.pages_label(numbers)
        self._jobs.append(_TranslationJob(pool.submit(keyed_items, label), pages, new))
        self._sent.update(keyed_items)
        self._sent.free |= keyed_items.free
        progress = pool.progress()
        self.callback(ProgressEvent(
            PipelinePhase.TRANSLATION, progress["requests_done"], progress["requests_total"],
            f"{label} 대사 {len(keyed_items)}개를 번역기에 보냈습니다",
            extras={**progress, "texts": len(keyed_items), "pages": len(pages)},
        ))

    def _finish_pages(self, pages, ckpt, new):
        """번역이 끝난(또는 멈춘) 쪽 묶음을 세고 저장한다."""
        self._count_outcomes(pages)
        if ckpt and new:
            ckpt.mark_pass1_batch_complete(pages)
        if self._glossary is not None:
            self._glossary.save()

    def _collect_translations(self, ckpt, wait=False):
        """끝난 번역 요청을 거둬 저장한다. wait=True면 남은 요청이 다 끝날 때까지 기다린다.

        요청 하나가 실행을 멈출 오류(로그인 필요 등)로 끝나면 아직 시작하지 않은 요청은 거두고,
        돌던 요청은 끝까지 받아 저장한 뒤 그 오류를 올린다 — 읽은 쪽은 대기로 남아 다음 실행에서 번역한다.
        """
        error = None
        started = time.perf_counter()
        while self._jobs:
            for job in [job for job in self._jobs if job.future.done()]:
                self._jobs.remove(job)
                failure = None if job.future.cancelled() else job.future.exception()
                if failure is not None and error is None:
                    error = failure
                    for other in self._jobs:
                        other.future.cancel()
                self._finish_pages(job.pages, ckpt, job.new)
            if not self._jobs or not (wait or error):
                break
            futures_wait([job.future for job in self._jobs], timeout=_WAIT_TICK_SEC, return_when=FIRST_COMPLETED)
            if error is None and any(not job.future.done() for job in self._jobs):
                self._emit_waiting(time.perf_counter() - started)
        if wait:
            self.clock.add("번역", time.perf_counter() - started)  # 쪽을 다 읽은 뒤 번역만 기다린 시간
        if error is not None:
            raise error

    def _emit_waiting(self, elapsed=None):
        """쪽을 다 읽고 번역 답만 기다린다는 알림 — 남은 요청 수와 기다린 시간."""
        progress = self._pool.progress()
        remaining = progress["requests_total"] - progress["requests_done"]
        waited = f", {elapsed:.0f}초" if elapsed is not None else ""
        self.callback(ProgressEvent(
            PipelinePhase.TRANSLATION, progress["requests_done"], progress["requests_total"],
            f"번역 답을 기다리는 중... (남은 요청 {remaining}개{waited})",
            extras={**progress, "waiting": True},
        ))

    def _refresh_by_glossary(self, pages):
        """세션마다 다르게 옮긴 이름을 사전 표기(더 많은 세션이 고른 표기)로 맞춘다."""
        changes = self._glossary.refresh(pages)
        if not changes:
            return
        pairs = Counter(f"{old}→{new}" for _, old, new in changes)
        message = (f"인명·용어 사전에 맞춰 대사 {len(changes)}곳의 표기를 맞췄습니다 ("
                   + ", ".join(pair for pair, _ in pairs.most_common(3)) + ")")
        logger.info(message)
        self.callback(ProgressEvent(PipelinePhase.TRANSLATION, 0, 0, message, extras={"glossary_fixed": len(changes)}))

    def _start_check(self):
        """이번 실행에서 번역한 대사 가운데 틀림 의심(translator.check_notes)이 있는 것만 뒤에서 다시 묻는다 (늘 켜짐).

        걸린 대사가 없으면 묻지 않는다. 받은 답은 finish_check가 채우고, 답을 기다리는 동안 파이프라인은 글씨체 판정과
        (켰으면) 짧은 번역을 돌린다.
        """
        if self._pool is None or not self._sent or self.should_stop():
            return
        self._check_started = time.perf_counter()
        try:
            pending = self._pool.start_check(self._sent)
        except Exception as exc:  # noqa: BLE001 — 확인은 다듬기일 뿐, 실패해도 번역은 그대로 쓴다
            logger.warning(f"의심 대사 확인을 하지 못했습니다 — 받은 번역을 그대로 씁니다: {exc}")
            pending = []
        if not pending:
            logger.info("번역에 틀림 의심이 없어 다시 묻지 않습니다")
            return
        self._check = pending
        count = sum(len(subset) for subset, _, _ in pending)
        self.callback(ProgressEvent(PipelinePhase.TRANSLATION, 0, 0, f"의심스러운 대사 {count}곳을 다시 묻는 중...",
                                    extras={"review_started": True, "check_items": count}))

    def _close_pool(self):
        if self._pool is not None:
            self._pool.close()
            if self._pool.submitted:
                logger.info("번역 요청 %d번 (동시에 %d개까지) · 답을 기다린 시간 합 %.0f초 (쪽을 읽는 동안 함께 돌았다)",
                            self._pool.submitted, self._pool.size, self._pool.busy_seconds)

    def check_pages(self, pages):
        """뒤에서 도는 의심 대사 확인이 다시 묻는 대사가 있는 쪽 이름들 — 확인이 없으면 None.

        이 쪽들은 답을 받은 뒤에 식자한다 (파이프라인이 나머지 쪽을 먼저 지우고 쓴다)."""
        if not self._check:
            return None
        asked = {id(element) for subset, _, _ in self._check for element in subset.values()}
        return {page.source_page for page in pages if any(id(element) in asked for element in page.text_elements())}

    def finish_check(self, pages):
        """뒤에서 돈 의심 대사 확인의 답을 받아 채우고, 번역문에 남은 일본 글자를 한글로 거른다 (확인이 없었어도).

        파이프라인이 글씨체 판정·짧은 번역(켰을 때) 뒤에 부른다. pages는 책의 쪽 (사전 표기를 맞추고 거를 범위). 번역이 바뀐 대사와,
        번역할지가 바뀐 쪽 이름을 돌려준다 — 대사집에 저장하는 것은 부르는 쪽이 한다. 확인이 고친 인명 표기는
        사전에 굳히고, 확인이 손대지 않은 대사도 그 표기로 맞춘다.
        """
        pending, self._check = self._check, None
        elements = [element for page in pages for element in page.text_elements()]
        before = {id(element): (element.translated_text, element.translation_status) for element in elements}
        if pending is not None:
            waited = time.perf_counter()
            if self._pool.check_waiting(pending):
                self.callback(ProgressEvent(PipelinePhase.TRANSLATION, 0, 0, "의심 대사 확인 답을 기다리는 중...",
                                            extras={"review_waiting": True}))
            try:
                changes = self._pool.finish_check(pending)
            except Exception as exc:  # noqa: BLE001 — 확인은 다듬기일 뿐, 실패해도 번역은 그대로 쓴다
                logger.warning(f"의심 대사 확인을 하지 못했습니다 — 받은 번역을 그대로 씁니다: {exc}")
                changes = []
            finally:
                self._close_pool()
            if self._glossary is not None:
                self._refresh_by_glossary(pages)
                self._glossary.save()
            now = time.perf_counter()
            self.clock.add("의심 대사 확인", now - waited)  # 글씨체 판정·짧은 번역과 함께 돌고 나서 더 기다린 시간만
            for key, old, new, reason in changes:
                logger.info(f"의심 대사 확인 {key}: {old if old is not None else '(번역 안 함)'} → "
                            f"{new if new is not None else '(번역 안 함)'} ({reason})")
            skipped = sum(1 for change in changes if change[2] is None)
            revived = sum(1 for change in changes if change[1] is None)
            # 대사 집계는 번역을 받을 때 셌다 — 확인이 빼거나 되살린 글자를 옮겨 센다
            self._stats["translated"] += revived - skipped
            self._stats["excluded"] += skipped - revived
            asked = sum(len(subset) for subset, _, _ in pending)
            message = f"의심 대사 {asked}곳 중 {len(changes)}곳을 고쳤습니다"
            logger.info(f"{message} — 확인 {now - self._check_started:.0f}초, 식자 준비 뒤 더 기다린 시간 {now - waited:.0f}초")
            self.callback(ProgressEvent(PipelinePhase.TRANSLATION, 0, 0, message, elapsed_sec=now - self._check_started,
                                        extras={"review_fixed": len(changes)}))
            self._emit_summary()
        for element in elements:
            if element.translation_status == TranslationStatus.TRANSLATED and element.translated_text:
                fixed = translator.hangul_for_kana(element.translated_text)
                if fixed != element.translated_text:
                    logger.info(f"번역문에 남은 일본 글자를 한글로: {element.translated_text} → {fixed}")
                    element.translated_text = fixed
        changed = [element for element in elements
                   if (element.translated_text, element.translation_status) != before[id(element)]]
        flipped = {page.source_page for page in pages for element in page.text_elements()
                   if element.translation_status != before[id(element)][1]}
        return changed, flipped

    @staticmethod
    def _is_page_translated(page_data: PageData) -> bool:
        return page_data.is_translated

    def _count_outcomes(self, pages):
        for page in pages:
            for element in page.text_elements():
                self._stats[element.translation_status.value] += 1

    def _emit_summary(self):
        """이번 실행에서 대사가 어디서 몇 개씩 빠졌는지 한 줄로 알린다 (처리한 게 없으면 생략)."""
        s = self._stats
        if not s:
            return
        parts = []
        if s["boxes"]:
            dropped = s["ocr_empty"] + s["ocr_invalid"] + s["artwork"]
            parts.append(f"글자 박스 {s['boxes']}개 중 {dropped}개 제외 (빈 판독 {s['ocr_empty']} · "
                         f"글자 아님 {s['ocr_invalid']} · 그림으로 판단 {s['artwork']})")
        if s["kept_as_freeform"] or s["merged_blocks"]:
            parts.append(f"말풍선 밖 글자로 살림 {s['kept_as_freeform']} · "
                         f"한 말풍선으로 합친 블록 {s['merged_blocks']}")
        parts.append(f"번역 {s['translated']} · 번역 제외 {s['excluded']} · 실패 {s['failed']}")
        message = "대사 집계 — " + " / ".join(parts)
        logger.info(message)
        self.callback(ProgressEvent(
            PipelinePhase.TRANSLATION, 0, 0, message,
            level=EventLevel.WARNING if s["failed"] else EventLevel.INFO,
            extras=dict(s),
        ))

    @staticmethod
    def _sort_page_data_by_input_order(page_data_list, image_paths):
        ordered = {page_data.source_page: page_data for page_data in page_data_list}
        return [
            ordered[os.path.basename(path)]
            for path in image_paths
            if os.path.basename(path) in ordered
        ]

    def _extract_and_translate_data(self, image_paths, ckpt=None):
        """쪽을 묶음으로 읽고(탐지·인식·글씨체) 읽은 묶음은 번역 창구에 넘긴 뒤 바로 다음 묶음을 읽는다."""
        pool = self._translation_pool()
        pool.reading = True
        try:
            return self._read_pages(image_paths, ckpt)
        finally:
            pool.reading = False

    def _read_pages(self, image_paths, ckpt=None):
        all_page_data = []
        # 읽는 묶음 = 번역 묶음 — 12쪽 이하로 고르게 나누고, 한 묶음을 읽으면 바로 번역에 넘긴다
        sizes = _even_batches(len(image_paths), config.TRANSLATION_BATCH_SIZE)
        total_batches = len(sizes)
        starts = [sum(sizes[:n]) for n in range(total_batches)]

        for batch_idx, (i, batch_size) in enumerate(zip(starts, sizes), start=1):
            self._collect_translations(ckpt)  # 끝난 번역은 그때그때 저장한다
            if self.should_stop():
                logger.info("중단 요청: Pass 1을 배치 경계에서 멈춥니다 (%d/%d 배치 완료).",
                            batch_idx - 1, total_batches)
                self.callback(ProgressEvent(
                    PipelinePhase.PASS1_BATCH, batch_idx - 1, total_batches,
                    "중단 요청 — 끝난 배치까지는 저장되었습니다",
                    level=EventLevel.WARNING,
                ))
                break
            logger.info(f"--- Processing Batch {batch_idx}/{total_batches} ---")
            batch_paths = image_paths[i:i + batch_size]
            batch_started_at = time.perf_counter()

            worker_count = min(max(1, int(config.PASS1_IMAGE_LOAD_WORKERS)), len(batch_paths))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                batch_images_bgr = list(executor.map(read_image_bgr, batch_paths))

            failed_count = sum(1 for img in batch_images_bgr if img is None)
            if failed_count > 0:
                failed_names = [
                    os.path.basename(p) for p, img in zip(batch_paths, batch_images_bgr) if img is None
                ]
                logger.warning(f"{failed_count}개 이미지 로딩 실패: {failed_names}")
                self.callback(ProgressEvent(
                    PipelinePhase.PASS1_BATCH, batch_idx, total_batches,
                    f"배치 {batch_idx}: {failed_count}개 이미지 로딩 실패 ({', '.join(failed_names[:3])}"
                    f"{'…' if failed_count > 3 else ''})",
                    level=EventLevel.WARNING,
                    extras={"failed_count": failed_count, "failed_pages": failed_names},
                ))

            batch_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch_images_bgr if img is not None]
            valid_paths = [p for p, img in zip(batch_paths, batch_images_bgr) if img is not None]
            if not batch_images_rgb:
                continue

            num_pages = len(batch_images_rgb)
            detection_started_at = time.perf_counter()
            self.callback(ProgressEvent(
                PipelinePhase.DETECTION, batch_idx, total_batches,
                f"배치 {batch_idx}/{total_batches}: {num_pages}페이지 탐지 중...",
                extras={"pages": num_pages},
            ))
            all_text_items, all_bubbles_by_page = extractor.detect_objects(self.models["detection"], batch_images_rgb)
            raw_box_count = len(all_text_items)
            bubble_count = sum(len(page_bubbles) for page_bubbles in all_bubbles_by_page)
            merged_text_items = extractor.merge_text_boxes(all_text_items)
            merged_box_count = len(merged_text_items)
            detection_elapsed = time.perf_counter() - detection_started_at
            self.clock.add("글자 찾기", detection_elapsed)
            self.callback(ProgressEvent(
                PipelinePhase.DETECTION, batch_idx, total_batches,
                f"배치 {batch_idx}: 말풍선 {bubble_count} / 글자 박스 {raw_box_count}→{merged_box_count} 병합",
                elapsed_sec=detection_elapsed,
                extras={
                    "pages": num_pages,
                    "bubbles": bubble_count,
                    "raw_boxes": raw_box_count,
                    "merged_boxes": merged_box_count,
                },
            ))

            ocr_started_at = time.perf_counter()
            self.callback(ProgressEvent(
                PipelinePhase.OCR, batch_idx, total_batches,
                f"배치 {batch_idx}: {merged_box_count}개 글자 조각 인식 + 폰트 분석 중...",
                extras={"target": merged_box_count},
            ))
            self._stats["boxes"] += merged_box_count
            processed_text_elements = extractor.extract_text_properties(
                self.models,
                batch_images_rgb,
                merged_text_items,
                stats=self._stats,
                clock=self.clock,
            )
            kept_count = len(processed_text_elements)
            filtered_count = max(0, merged_box_count - kept_count)
            ocr_elapsed = time.perf_counter() - ocr_started_at
            self.callback(ProgressEvent(
                PipelinePhase.OCR, batch_idx, total_batches,
                f"배치 {batch_idx}: 인식 {kept_count}개 / 품질 필터 {filtered_count}개 제외",
                elapsed_sec=ocr_elapsed,
                level=EventLevel.WARNING if filtered_count > 0 and kept_count == 0 else EventLevel.INFO,
                extras={"kept": kept_count, "filtered": filtered_count, "target": merged_box_count},
            ))

            untranslated_page_data = extractor.structure_page_data(
                valid_paths,
                batch_images_rgb,
                all_bubbles_by_page,
                processed_text_elements,
                stats=self._stats,
            )

            # 번역은 뒤에서 — 대사집 저장은 번역을 받은 뒤 (_collect_translations)
            self._send_translation(untranslated_page_data, ckpt, new=True)
            all_page_data.extend(untranslated_page_data)

            batch_elapsed = time.perf_counter() - batch_started_at
            self.callback(ProgressEvent(
                PipelinePhase.PASS1_BATCH, batch_idx, total_batches,
                f"{len(valid_paths)}페이지 · {batch_elapsed:.1f}초 "
                f"(탐지 {detection_elapsed:.1f}s + 인식 {ocr_elapsed:.1f}s) — 번역은 뒤에서 이어집니다",
                elapsed_sec=batch_elapsed,
                extras={
                    "pages": len(valid_paths),
                    "detection_sec": round(detection_elapsed, 2),
                    "ocr_sec": round(ocr_elapsed, 2),
                },
            ))

        return all_page_data
