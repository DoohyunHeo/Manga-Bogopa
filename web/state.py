"""앱 전역 상태 관리 모듈"""
import logging
import threading

logger = logging.getLogger(__name__)


class AppState:
    """파이프라인 객체 하나와 실행 상태(실행 중인지, 멈춰 달라고 했는지)를 관리합니다."""

    def __init__(self):
        self.pipeline = None
        self.is_running = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def initialize_pipeline(self):
        """파이프라인 객체를 처음 한 번 만듭니다 (모델은 번역을 시작할 때 읽는다)."""
        from pipeline import MangaTranslationPipeline
        if self.pipeline is None:
            logger.info("파이프라인 초기화 중...")
            self.pipeline = MangaTranslationPipeline()
            logger.info("파이프라인 초기화 완료.")

    def acquire(self) -> bool:
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True
            self._stop_event.clear()
            return True

    def release(self):
        with self._lock:
            self.is_running = False

    def request_stop(self) -> bool:
        """실행 중인 파이프라인에 중단 신호를 보냅니다. 신호 성공 여부 반환."""
        with self._lock:
            if not self.is_running:
                return False
            self._stop_event.set()
            return True

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()


app_state = AppState()
