"""Manga-Bogopa 애플리케이션 진입점"""
import logging

from src import config
from web.state import app_state
from web.ui import build_ui

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    if config.is_configured():
        logger.info("모델 로딩 중... (10~30초 소요)")
        app_state.initialize_pipeline()
        logger.info("모델 로딩 완료.")
    else:
        logger.info("초기 설정이 필요합니다. 웹 브라우저에서 설정을 완료하세요.")

    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
