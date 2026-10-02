"""직접 만든 모델 파일을 GitHub 릴리즈에서 받아 제자리(data/models)에 둔다 — 처음 실행하는 사람이 손으로 받지 않게.

Hugging Face에서 받는 모델(보완 탐지·읽기·지우기·글자 모양)은 각 모듈이 알아서 받고, 여기는 이 저장소의
릴리즈에 올리는 파일만 맡는다. 받는 곳은 모델 전용 릴리즈 태그(RELEASE_TAG)에 고정한다 — 프로그램을 새로
릴리즈해도 모델을 다시 올리지 않고, 모델이 바뀌면 새 태그에 올리고 여기 태그·크기·해시만 고친다.

이미 있는 파일은 절대 덮어쓰지 않는다(새로 학습한 모델을 그 자리에 두고 시험하는 경우가 있다). 없는 파일만
받아서 해시가 맞을 때만 제자리로 옮긴다. 릴리즈를 올릴 때는 `python -m src.model_files`로 지금 파일의
크기·해시를 찍어 MODEL_FILES에 옮긴다.
"""
import hashlib
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional

from src import config
from src.progress import PipelinePhase, ProgressEvent
from src.utils import replace_file

logger = logging.getLogger(__name__)

RELEASE_TAG = "models-v1"
RELEASE_BASE = f"https://github.com/DoohyunHeo/Manga-Bogopa/releases/download/{RELEASE_TAG}"
_CHUNK = 1 << 20
_ATTEMPTS = 3
_PROGRESS_EVERY_SEC = 0.5


class ModelDownloadError(RuntimeError):
    """모델 파일을 받지 못했다 — 화면에 그대로 보여 줄 수 있는 문장을 담는다."""


@dataclass(frozen=True)
class ModelFile:
    config_field: str   # 이 파일의 경로를 담은 config 필드
    asset: str          # 릴리즈에 올린 파일 이름
    size: int
    sha256: str
    label: str          # 화면에 보일 이름


MODEL_FILES: List[ModelFile] = [
    ModelFile("MODEL_PATH", "MangaTextExtractor-V2.pt", 44121241,
              "933b63f745161ee6d80d43b78eccc5d399444bc368c8a6af5ecbf087d541f3f0", "글자 찾기 모델"),
    ModelFile("FONT_STYLE6_MODEL_PATH", "font_style6_analyzer.pth", 116104101,
              "8a6093296bf306027d68e88eaae6390002de6325efc334bdc6b7015c427c35e3", "글씨 모양 모델"),
]


def missing_files() -> List[ModelFile]:
    """제자리에 없는 모델 파일."""
    return [item for item in MODEL_FILES if not os.path.exists(getattr(config._config, item.config_field))]


def ensure_model_files(callback: Optional[Callable[[ProgressEvent], None]] = None,
                       should_stop: Optional[Callable[[], bool]] = None) -> bool:
    """없는 모델 파일을 받는다. 다 있으면(또는 다 받으면) True, 멈춰 달라고 해서 그만두면 False.

    받지 못하면 ModelDownloadError를 낸다 — 이미 받은 파일은 그대로 둔다.
    """
    todo = missing_files()
    for index, item in enumerate(todo, start=1):
        if should_stop and should_stop():
            return False
        if not _download(item, index, len(todo), callback, should_stop):
            return False
    return True


def _emit(callback, item, index, count, done, message=""):
    if callback:
        callback(ProgressEvent(PipelinePhase.LOADING_MODELS, done, item.size, message, extras={
            "download": item.label, "file_index": index, "file_count": count,
            "done_bytes": done, "total_bytes": item.size,
        }))


def _download(item: ModelFile, index: int, count: int, callback, should_stop) -> bool:
    path = getattr(config._config, item.config_field)
    url = f"{RELEASE_BASE}/{item.asset}"
    part = path + ".part"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _emit(callback, item, index, count, 0, f"모델 파일 받는 중 ({index}/{count}): {item.asset}")
    last_error = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            digest, done, last_sent = hashlib.sha256(), 0, 0.0
            request = urllib.request.Request(url, headers={"User-Agent": "Manga-Bogopa"})
            with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
                while True:
                    if should_stop and should_stop():
                        out.close()
                        os.remove(part)
                        return False
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if time.monotonic() - last_sent >= _PROGRESS_EVERY_SEC:
                        _emit(callback, item, index, count, done)  # 문구 없는 진행 알림은 '자세한 기록'에 남지 않는다
                        last_sent = time.monotonic()
            if done != item.size or digest.hexdigest() != item.sha256:
                raise ValueError(f"받은 파일이 맞지 않습니다 (크기 {done}/{item.size})")
            replace_file(part, path)
            _emit(callback, item, index, count, item.size, f"모델 파일을 받았습니다: {item.asset}")
            logger.info("모델 파일을 받았습니다: %s -> %s", url, path)
            return True
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 404:
                break  # 릴리즈에 파일이 없다 — 다시 해도 같다
        except (urllib.error.URLError, OSError, ValueError) as e:
            last_error = e
        logger.warning("모델 파일 받기 실패 (%d/%d): %s — %s", attempt, _ATTEMPTS, url, last_error)
        if os.path.exists(part):
            os.remove(part)
        if attempt < _ATTEMPTS:
            time.sleep(2 * attempt)
    raise ModelDownloadError(_message(item, last_error)) from last_error


def _message(item: ModelFile, error) -> str:
    if isinstance(error, urllib.error.HTTPError) and error.code == 404:
        return (f"{item.label} 파일을 받을 곳에 파일이 없어요. 잠시 뒤에 다시 해 보거나, "
                f"GitHub 릴리즈({RELEASE_TAG})에서 {item.asset}를 받아 data/models 폴더에 넣어 주세요.")
    if isinstance(error, OSError) and not isinstance(error, urllib.error.URLError) and getattr(error, "errno", None) == 28:
        return f"{item.label} 파일을 저장할 자리가 모자라요. 디스크 공간을 비운 뒤 다시 해 주세요."
    if isinstance(error, ValueError):  # 크기·해시가 맞지 않음 — 받다 깨졌거나 릴리즈 파일이 바뀌었다
        return (f"{item.label} 파일을 받았지만 내용이 맞지 않아요. 잠시 뒤에 다시 해 보고, 계속되면 "
                f"GitHub 릴리즈({RELEASE_TAG})에서 {item.asset}를 받아 data/models 폴더에 넣어 주세요.")
    return f"{item.label} 파일을 받지 못했어요. 인터넷 연결을 확인하고 다시 해 주세요."


if __name__ == "__main__":
    # 릴리즈에 올릴 지금 파일의 크기·해시 — MODEL_FILES에 옮겨 적는다
    for item in MODEL_FILES:
        path = getattr(config._config, item.config_field)
        if not os.path.exists(path):
            print(f"# 없음: {path}", file=sys.stderr)
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(_CHUNK), b""):
                digest.update(block)
        print(f'ModelFile("{item.config_field}", "{item.asset}", {os.path.getsize(path)}, "{digest.hexdigest()}", "{item.label}"),')
