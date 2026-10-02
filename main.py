"""Manga-Bogopa 실행 시작점 — 이 컴퓨터에서만 열리는 웹 화면을 띄우고 브라우저를 연다."""
import logging
import os
import threading
import webbrowser

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("MANGA_BOGOPA_PORT", "7860"))


def main():
    import uvicorn

    from web.server import app

    url = f"http://127.0.0.1:{PORT}"
    logger.info("웹 화면: %s (브라우저 창을 닫아도 이 창을 끄기 전까지는 주소로 다시 열 수 있어요)", url)
    if os.environ.get("MANGA_BOGOPA_NO_BROWSER") != "1":
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
