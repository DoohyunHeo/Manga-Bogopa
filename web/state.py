"""앱 전역 상태 관리 모듈"""
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class AppState:
    """모델 싱글톤 및 파이프라인 실행 상태를 관리합니다."""

    def __init__(self):
        self.pipeline = None
        self.is_running = False
        self._lock = threading.Lock()

    def initialize_pipeline(self, **kwargs):
        """모델을 로드하고 파이프라인을 초기화합니다."""
        from pipeline import MangaTranslationPipeline
        if self.pipeline is None:
            logger.info("파이프라인 초기화 중...")
            self.pipeline = MangaTranslationPipeline(**kwargs)
            logger.info("파이프라인 초기화 완료.")

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    def acquire(self) -> bool:
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True
            return True

    def release(self):
        with self._lock:
            self.is_running = False


app_state = AppState()
