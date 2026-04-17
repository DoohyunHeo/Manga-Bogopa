import logging
import re
import time
from typing import Callable, List, Optional
from src.data_models import PageData
from src.progress import ProgressEvent, PipelinePhase

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

# Ellipsis normalization: Gemini emits ..., …, ．．．, 。。。, ・・・, 점 사이 공백 등 다양한 형태.
# 모두 U+22EF(⋯)로 통일.
_ELLIPSIS_SPACED = re.compile(r'(?<=[.\u3002\uff0e\u30fb])\s+(?=[.\u3002\uff0e\u30fb])')
_ELLIPSIS_RUNS = re.compile(
    r'[\u2026\u2025]+'                      # …  ‥
    r'|[.\u3002\uff0e\u30fb]{2,}'           # .. .., 。。。, ．．．, ・・・
)


def _normalize_ellipsis(text: str) -> str:
    if not text:
        return text
    collapsed = _ELLIPSIS_SPACED.sub('', text)
    return _ELLIPSIS_RUNS.sub('⋯', collapsed)


def _stream_with_retry(chat_session, request_text, callback=None):
    """스트리밍으로 API 요청을 보내고 응답을 조합합니다. 실패 시 재시도."""
    for attempt in range(MAX_RETRIES):
        try:
            response_stream = chat_session.send_message(request_text, stream=True)

            chunks = []
            lines_received = 0
            start_time = time.time()
            last_report = 0

            for chunk in response_stream:
                chunks.append(chunk.text)
                lines_received += chunk.text.count('\n')
                elapsed = time.time() - start_time

                # 2초마다 또는 새 줄이 올 때 상태 보고
                if callback and (elapsed - last_report >= 2 or chunk.text.count('\n') > 0):
                    last_report = elapsed
                    callback(ProgressEvent(
                        PipelinePhase.TRANSLATION, lines_received, 0,
                        f"Gemini 응답 수신 중... ({lines_received}줄, {elapsed:.0f}초 경과)"
                    ))

            full_text = "".join(chunks)

            if callback:
                elapsed = time.time() - start_time
                callback(ProgressEvent(
                    PipelinePhase.TRANSLATION, 1, 1,
                    f"Gemini 응답 수신 완료 ({lines_received}줄, {elapsed:.1f}초)"
                ))

            return full_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY ** (attempt + 1)
                logger.info(f"번역 API 오류 (시도 {attempt + 1}/{MAX_RETRIES}): {e}. {delay}초 후 재시도...")
                if callback:
                    callback(ProgressEvent(
                        PipelinePhase.TRANSLATION, 0, 0,
                        f"API 오류, {delay}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})..."
                    ))
                time.sleep(delay)
            else:
                raise


def _parse_translation_line(line):
    """응답에서 번호와 번역 텍스트를 분리합니다."""
    num_match = re.match(r'^([\d.]+)\.\s*', line)
    if not num_match:
        return None, ""

    key = num_match.group(1).rstrip('.')
    rest = line[num_match.end():]

    paren_match = re.search(r'\((.*)\)', rest)
    cleaned = paren_match.group(1) if paren_match else rest.strip()

    if cleaned == "번역 불가":
        return key, ""
    return key, _normalize_ellipsis(cleaned)


def translate_pages_in_batch(chat_session, batch_page_data: List[PageData],
                             callback: Optional[Callable] = None):
    """페이지 데이터 리스트를 받아 모든 텍스트를 페이지.텍스트 번호로 일괄 번역합니다."""
    logger.info(f"{len(batch_page_data)} 페이지 데이터 전체 번역 요청...")

    keyed_items = {}
    lines = []

    for page_idx, page_data in enumerate(batch_page_data):
        page_num = page_idx + 1
        text_num = 0

        for bubble in page_data.speech_bubbles:
            if bubble.text_element.original_text:
                text_num += 1
                key = f"{page_num}.{text_num}"
                keyed_items[key] = bubble.text_element
                lines.append(f"{key}.({bubble.text_element.original_text})")

        for freeform_text in page_data.freeform_texts:
            if freeform_text.original_text:
                text_num += 1
                key = f"{page_num}.{text_num}"
                keyed_items[key] = freeform_text
                lines.append(f"{key}.({freeform_text.original_text})")

    if not keyed_items:
        logger.info("번역할 텍스트가 없습니다.")
        return batch_page_data

    request_text = "\n".join(lines)

    if callback:
        callback(ProgressEvent(PipelinePhase.TRANSLATION, 0, 1,
                               f"{len(keyed_items)}개 대사 Gemini에 전송 중..."))

    try:
        response_text = _stream_with_retry(chat_session, request_text, callback)
        response_lines = response_text.strip().split('\n')

        if len(response_lines) != len(keyed_items):
            logger.warning(f"응답 줄 수({len(response_lines)})와 요청 수({len(keyed_items)})가 불일치합니다.")

        for resp_line in response_lines:
            resp_line = resp_line.strip()
            if not resp_line:
                continue
            key, translated = _parse_translation_line(resp_line)
            if key and key in keyed_items:
                keyed_items[key].translated_text = translated
            elif key:
                logger.warning(f"알 수 없는 번호: {key}")

        for key, element in keyed_items.items():
            if element.translated_text is None:
                element.translated_text = ""

        for page_data in batch_page_data:
            page_data.speech_bubbles = [
                bubble for bubble in page_data.speech_bubbles
                if bubble.text_element.translated_text != ""
            ]
            page_data.freeform_texts = [
                text for text in page_data.freeform_texts
                if text.translated_text != ""
            ]

    except Exception as e:
        logger.error(f"통합 번역 중 오류 발생 ({MAX_RETRIES}회 재시도 후 실패): {e}")
        for element in keyed_items.values():
            element.translated_text = "번역 불가"

    logger.info("번역 완료.")
    return batch_page_data
