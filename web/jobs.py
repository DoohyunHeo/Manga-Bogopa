"""번역 작업 — 한 번에 하나만 돈다. 파이프라인을 따로 스레드에서 돌리고, 진행 알림을 화면이 그대로
그릴 수 있는 '지금 상태'(단계·묶음·쪽 수·남은 시간·알림·기록)로 모아 둔다.

화면은 이 상태를 SSE로 받아 다시 그리기만 한다. 브라우저를 닫았다 열어도 상태는 여기 남아 있다.
"""
import logging
import threading
import time
from typing import Dict, List, Optional

from src import config, model_files, translator
from src.progress import EventLevel, PipelinePhase, ProgressEvent
from web import library
from web.state import app_state

logger = logging.getLogger(__name__)

# 1단계(읽기)의 세부 단계 — 번역을 다 받은 뒤 오는 '저장'(SAVING_JSON)은 단계를 되돌리지 않게 뺀다
_READING = {PipelinePhase.DETECTION, PipelinePhase.OCR, PipelinePhase.PASS1_BATCH}
_LOG_LIMIT = 300
_NOTICE_LIMIT = 30


class Job:
    def __init__(self, book: dict, mode: str, pages: Optional[List[str]] = None):
        self.book_id = book["id"]
        self.title = book["title"]
        self.mode = mode
        self.pages = list(pages or [])   # 고른 쪽만 다시 쓸 때 (번역 고치기)
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.status = "running"          # running / done / incomplete / stopped / error
        self.stage = "preparing"         # preparing / reading / translating / typesetting
        self.substage = ""               # 1단계 안에서: 글자 찾기 / 글자 읽기 / 번역 요청
        self.batch = 0
        self.batches = 0
        self.stage_started_at = self.started_at
        self.pages_done = 0
        self.pages_total = 0
        self.pages_skipped = 0            # 전에 끝내서 이번에 건너뛴 쪽
        self.page_seconds: List[float] = []
        self.check_waiting = False        # 먼저 쓸 쪽을 다 쓰고 의심 대사 확인 답을 기다리는 중 — 남은 시간을 셀 수 없다
        self.done_pages: List[Dict] = []  # 이번 작업에서 새로 끝난 쪽 (화면이 그림을 새로 불러온다)
        self.notices: List[Dict] = []
        self.log: List[str] = []
        self.message = ""
        self.error_kind = ""
        self.activity = "시작하는 중이에요"  # 화면에 보일 '지금 하는 일' 한 줄
        self.translation: Optional[Dict] = None  # 뒤에서 도는 번역 — 보낸·받은 묶음 수 (쪽을 읽는 동안에도 온다)
        self.download: Optional[Dict] = None     # 처음 한 번 받는 모델 파일 — 이름·받은 바이트·몇 번째 파일
        self.version = 0

    def snapshot(self) -> dict:
        now = self.finished_at or time.time()
        eta = None
        if self.stage == "typesetting" and self.page_seconds and self.pages_total and not self.check_waiting:
            recent = self.page_seconds[-12:]
            eta = round(sum(recent) / len(recent) * max(0, self.pages_total - self.pages_done))
        # 이어서 할 때는 전에 끝낸 쪽까지 쳐서 보여 준다 — 책장의 'N / M쪽'과 같은 숫자가 되게
        skipped = self.pages_skipped if self.pages_total else 0
        return {
            "book_id": self.book_id, "title": self.title, "mode": self.mode, "pages": self.pages,
            "started_at": self.started_at,
            "status": self.status, "stopping": self.status == "running" and app_state.stop_requested,
            "stage": self.stage, "substage": self.substage, "batch": self.batch, "batches": self.batches,
            "pages_done": skipped + self.pages_done, "pages_total": skipped + self.pages_total,
            "pages_skipped": self.pages_skipped, "eta_sec": eta,
            "elapsed_sec": round(now - self.started_at), "stage_elapsed_sec": round(now - self.stage_started_at),
            "done_pages": self.done_pages[-400:], "notices": self.notices[-_NOTICE_LIMIT:],
            "log": self.log[-_LOG_LIMIT:], "message": self.message, "error_kind": self.error_kind,
            "activity": self.activity, "translation": self.translation, "download": self.download,
            "version": self.version,
        }


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.current: Optional[Job] = None

    def snapshot(self) -> Optional[dict]:
        with self._lock:
            return self.current.snapshot() if self.current else None

    def version(self) -> int:
        job = self.current
        return job.version if job else -1

    def is_running(self) -> bool:
        return app_state.is_running

    def start(self, book: dict, mode: str, pages: Optional[List[str]] = None) -> Job:
        """책 하나의 작업을 시작한다 (pages를 주면 그 쪽만 식자만 다시). 다른 작업이 돌고 있으면 RuntimeError."""
        app_state.initialize_pipeline()
        if not app_state.acquire():
            raise RuntimeError("다른 작업을 하는 중이에요. 끝나거나 멈춘 뒤에 다시 해 주세요.")
        job = Job(book, mode, pages)
        with self._lock:
            self.current = job
        try:
            threading.Thread(target=self._run, args=(job, book), daemon=True).start()
        except Exception:
            app_state.release()  # 작업 스레드를 못 띄웠으면 잡은 잠금을 돌려준다
            raise
        return job

    def stop(self) -> bool:
        stopped = app_state.request_stop()
        with self._lock:
            if self.current:
                self.current.version += 1  # 다음 알림을 기다리지 않고 화면에 '멈추는 중'을 바로 보낸다
        return stopped

    # ── 작업 스레드 ──

    def _run(self, job: Job, book: dict):
        pipeline = app_state.pipeline
        config._config.INPUT_DIR = book["input_dir"]   # 실행 중에만 — config.json에는 쓰지 않는다
        pipeline.output_dir = book["output_dir"]
        pipeline.run_mode = job.mode
        pipeline.only_pages = set(job.pages) if job.pages else None
        pipeline.callback = lambda event: self._on_event(job, event)
        pipeline.should_stop = lambda: app_state.stop_requested
        try:
            pipeline.run()
            if app_state.stop_requested:
                self._finish(job, "stopped", "멈췄어요. '이어서 번역'을 누르면 멈춘 곳부터 다시 해요.")
        except translator.TranslationUnavailable as e:
            logger.error("번역 연결 실패: %s", e)
            job.error_kind = "translation_unavailable"
            self._finish(job, "error", str(e))
        except model_files.ModelDownloadError as e:
            logger.error("모델 파일 받기 실패: %s", e)
            job.error_kind = "model_download"
            self._finish(job, "error", str(e))  # 이미 받은 파일은 남아 있어 다시 하면 나머지만 받는다
        except Exception as e:  # noqa: BLE001 — 어떤 실패든 화면에 알리고 잠금을 푼다
            logger.error("번역 작업 오류: %s", e, exc_info=True)
            job.error_kind = "error"
            with self._lock:
                job.log.append(f"오류: {type(e).__name__}: {e}")  # 원래 오류는 '자세한 기록'에
            self._finish(job, "error", "작업이 예상치 못하게 멈췄어요. 진행한 곳까지는 저장돼 있으니 ‘이어서 번역’으로 "
                                       "다시 해 보세요. 또 멈추면 ‘자세한 기록’의 마지막 줄을 확인해 주세요.")
        finally:
            try:
                if job.status == "running" and job.error_kind:
                    self._finish(job, "error", job.message)
                elif job.status == "running":
                    # 번역을 못 받은 쪽은 식자하지 않으므로, 끝났는지는 책 전체로 본다 (쪽만 다시 쓸 때는 그 쪽들로)
                    state = library.book_state(book)
                    if job.pages:
                        done = {p["name"] for p in state["pages"] if p["typeset"]}
                        finished = all(name in done for name in job.pages)
                    else:
                        finished = state["status"] == "done"
                    self._finish(job, "done" if finished else "incomplete", job.message)
            except Exception as e:  # noqa: BLE001 — 끝난 상태를 읽지 못해도 잠금은 푼다
                logger.error("작업 마무리 중 오류: %s", e, exc_info=True)
                if job.status == "running":
                    self._finish(job, "incomplete", job.message)
            finally:
                pipeline.callback = lambda event: None
                pipeline.only_pages = None
                app_state.release()

    def _finish(self, job: Job, status: str, message: str):
        with self._lock:
            job.status = status
            job.message = message or job.message
            job.finished_at = time.time()
            job.version += 1

    def _on_event(self, job: Job, event: ProgressEvent):
        with self._lock:
            phase = event.phase
            text = (event.message or "").strip()
            extras = event.extras or {}
            if text and not extras.get("waiting"):  # 번역을 기다리는 동안 2초마다 오는 알림은 기록에 남기지 않는다
                job.log.append(text)
            # 멈춰 달라고 한 뒤의 경고와 끝(COMPLETE) 알림은 결과 문구와 겹치므로 기록에만 남긴다
            if (event.level in (EventLevel.WARNING, EventLevel.ERROR) and text and not app_state.stop_requested
                    and phase != PipelinePhase.COMPLETE):
                job.notices.append({"level": str(event.level), "message": text})
            # 확인 답을 기다린다는 알림 다음에 오는 알림(사전 맞춤·고친 수·나머지 쪽 쓰기)은 기다림이 끝났다는 뜻
            job.check_waiting = phase == PipelinePhase.TRANSLATION and bool(extras.get("review_waiting"))
            if "requests_total" in extras:  # 뒤에서 도는 번역 — 보낸·받은 묶음 수
                job.translation = {"done": extras.get("requests_done", 0), "total": extras["requests_total"]}
            # 처음 한 번 모델 파일을 받는 중 — 다른 알림이 오면 받기는 끝난 것
            job.download = ({"label": extras["download"], "done": extras["done_bytes"], "total": extras["total_bytes"],
                             "index": extras["file_index"], "count": extras["file_count"]}
                            if "download" in extras else None)
            if phase == PipelinePhase.LOADING_MODELS:
                if extras.get("base_font"):
                    # 책 전체 글씨체 기준은 쓰기 준비 — 번역을 기다리는 동안 재면(읽기·번역 단계) 단계를 그대로 둔다.
                    # 그 뒤에도 사전 맞춤 같은 번역 알림이 와서, 쓰기로 넘겼다가 번역으로 되돌아가 보이지 않게
                    if job.stage in ("preparing", "typesetting"):
                        self._enter(job, "typesetting", "지우고 쓰기")
                elif job.stage == "preparing":  # 처음 한 번만 — 나중에 불러오는 모델(지우기 등)은 단계를 되돌리지 않는다
                    self._enter(job, "preparing", "모델 준비")
            elif phase in _READING:
                substage = {PipelinePhase.DETECTION: "글자 찾기", PipelinePhase.OCR: "글자 읽기"}
                self._enter(job, "reading", substage.get(phase, job.substage or "글자 찾기"))
                if phase == PipelinePhase.DETECTION and event.total:
                    job.batch, job.batches = event.current, event.total  # 읽기 묶음 번호
            elif phase == PipelinePhase.TRANSLATION:
                # 쪽을 읽는 동안 뒤에서 도는 번역 알림(reading)은 단계를 바꾸지 않는다 — 번역 칸에 받은 수만 곁들인다.
                # 의심 대사 확인의 답(review_waiting·review_fixed)은 확인할 대사가 없는 쪽을 먼저 쓰는 동안 오므로,
                # 쓰기가 시작된 뒤에는 번역으로 되돌리지 않는다
                if (event.level == EventLevel.INFO and not text.startswith("대사 집계") and not extras.get("reading")
                        and job.stage != "typesetting"):
                    self._enter(job, "translating", "번역 요청")
            elif phase == PipelinePhase.PASS2_PAGE:
                self._enter(job, "typesetting", "지우고 쓰기")
                if (event.extras or {}).get("restored_from_disk"):
                    job.pages_skipped += 1  # 전에 끝낸 쪽 — 번호가 책 전체 기준이라 진행 수에 섞지 않는다
                else:
                    if event.total:
                        job.pages_total = event.total  # 이번에 쓸 쪽 수
                    if event.page_name and event.level == EventLevel.INFO:
                        job.pages_done = max(job.pages_done, event.current)
                        if event.elapsed_sec:
                            job.page_seconds.append(float(event.elapsed_sec))
                        job.done_pages.append({"name": event.page_name, "at": time.time()})
            elif phase == PipelinePhase.COMPLETE and not text.startswith("단계별 시간"):  # 시간 기록은 '자세한 기록'에만
                if event.level == EventLevel.INFO and event.current and event.total:
                    job.message = "글자를 다시 썼어요." if job.mode == "retypeset" else "번역을 모두 끝냈어요."
                elif text:
                    job.message = text
                    if event.level == EventLevel.ERROR:
                        job.error_kind = "error"  # 시작하지 못하고 끝냈다 — 작업을 오류로 마친다
            job.activity = _activity(job, event, text) or job.activity
            job.version += 1

    @staticmethod
    def _enter(job: Job, stage: str, substage: str):
        if job.stage != stage:
            job.stage = stage
            job.stage_started_at = time.time()
        job.substage = substage


def _activity(job: Job, event: ProgressEvent, text: str) -> str:
    """파이프라인 알림을 '지금 하는 일' 한 줄로 — 개발용 기록 대신 사용자가 알아듣는 말과 숫자로."""
    phase, extras = event.phase, event.extras or {}
    if phase == PipelinePhase.LOADING_MODELS:
        if event.level != EventLevel.INFO:
            return ""
        if "download" in extras:  # 처음 한 번 모델 파일을 받는 중
            if extras["done_bytes"] >= extras["total_bytes"]:
                return f"{extras['download']}를 받았어요"
            return (f"모델 파일을 받는 중이에요 ({extras['file_index']}/{extras['file_count']}) {extras['download']} "
                    f"{extras['done_bytes'] / 1e6:.0f} / {extras['total_bytes'] / 1e6:.0f}MB")
        if extras.get("base_font"):  # 책 전체로 정하는 글씨체 판정(font_relative) — 기록 파일을 새로 만들 때만 온다
            return "글씨체를 골랐어요" if event.current else "책 전체를 보고 글씨체를 고르는 중이에요"
        if job.stage == "typesetting":  # 지우기 모델을 다 불러오면 바로 첫 묶음을 지운다
            return "쪽마다 원래 글자를 지우는 중이에요. 첫 쪽이 곧 나와요" if event.current else "글자 지우기를 준비하는 중이에요"
        if job.stage == "preparing":
            return "모델을 다 불러왔어요" if event.current else "모델을 불러오는 중이에요"
        return ""  # 번역을 기다리는 동안 미리 불러오는 모델 — '지금 하는 일'은 그대로 둔다
    if phase == PipelinePhase.DETECTION:
        if "bubbles" in extras:
            return f"말풍선 {extras['bubbles']}개와 글자 {extras.get('merged_boxes', 0)}곳을 찾았어요"
        return f"{extras.get('pages', '')}쪽에서 말풍선과 글자를 찾는 중이에요"
    if phase == PipelinePhase.OCR:
        if "kept" in extras:
            return f"글자 {extras['kept']}곳을 읽었어요"
        return f"글자 {extras.get('target', '')}곳을 읽고 글씨체를 살피는 중이에요"
    if phase == PipelinePhase.TRANSLATION:
        if extras.get("reading"):
            return ""  # 쪽을 읽는 동안에는 '읽는 중' 문구를 그대로 두고, 번역은 번역 칸에 숫자로 보인다
        if "glossary_fixed" in extras:
            return f"이름 표기를 맞췄어요 ({extras['glossary_fixed']}곳)"
        # 번역을 다 받은 뒤 쓰기 전까지의 다듬기 — 의심 대사 확인과 짧은 번역은 함께 돌아서 끝나는 순서가 바뀔 수 있다.
        # 의심 대사가 없으면 확인 알림은 아예 오지 않는다
        if extras.get("review_started"):
            return f"틀렸을 수 있는 대사 {extras.get('check_items', 0)}곳을 번역기에 다시 묻는 중이에요"
        if extras.get("review_waiting"):
            return "다시 물은 대사의 답을 기다리는 중이에요"
        if "review_fixed" in extras:
            fixed = extras["review_fixed"]
            return f"다시 물은 대사 가운데 {fixed}곳을 고쳤어요" if fixed else "다시 물은 대사는 고칠 곳이 없었어요"
        if extras.get("fit_started"):
            return "좁은 말풍선에 맞게 더 짧은 번역을 받는 중이에요"
        if "fit_adopted" in extras:
            return f"좁은 말풍선 {extras['fit_adopted']}곳을 더 짧은 번역으로 바꿨어요"
        if "requests_total" in extras:  # 쪽을 다 읽고 남은 번역만 기다리는 중
            left = extras["requests_total"] - extras.get("requests_done", 0)
            return f"남은 번역 {left}묶음을 기다리는 중이에요" if left > 0 else "번역을 다 받았어요"
        if "texts" in extras:
            return f"대사 {extras['texts']}개를 번역기에 보냈어요. 답을 기다리는 중이에요"
        if text.startswith("누락된 번역"):
            return "번역에서 빠진 대사를 다시 요청하는 중이에요"
        if text.startswith("지난 실행에서"):
            return "지난번에 번역을 못 받은 대사를 다시 요청하는 중이에요"
        return ""
    if phase == PipelinePhase.PASS1_BATCH and event.level == EventLevel.INFO and "pages" in extras:
        return f"{event.current}번째 묶음({extras['pages']}쪽)을 읽었어요"  # 번역은 뒤에서 따로 돈다
    if phase == PipelinePhase.SAVING_JSON:
        return "번역을 저장했어요"
    if phase == PipelinePhase.PASS2_PAGE and event.level == EventLevel.INFO:
        if not event.page_name:
            if event.current:
                # 의심 대사 확인 답을 받은 뒤 나머지 — 먼저 쓴 쪽 뒤로 이어 센다. 먼저 쓴 쪽 가운데 답으로 대사가 바뀐
                # 쪽은 다시 써서 번호가 그만큼 앞에서 시작한다 (진행 수는 이미 센 데서 줄지 않는다)
                left, redo = job.pages_total - job.pages_done, job.pages_done - event.current
                if redo > 0:
                    return (f"남은 {left}쪽을 쓰고, 대사가 바뀐 {redo}쪽을 다시 쓰는 중이에요" if left > 0
                            else f"대사가 바뀐 {redo}쪽을 다시 쓰는 중이에요")
                return f"남은 {left}쪽을 마저 쓰는 중이에요" if left > 0 else ""
            return "쪽마다 원래 글자를 지우는 중이에요. 첫 쪽이 곧 나와요"
        if extras.get("restored_from_disk"):
            return f"전에 끝낸 {job.pages_skipped}쪽은 건너뛰어요"
        return f"지금까지 {job.pages_skipped + job.pages_done}쪽을 다 썼어요"  # 화면의 'N / M쪽'과 같은 수
    if phase == PipelinePhase.COMPLETE and job.message:
        return job.message
    return ""


jobs = JobManager()
