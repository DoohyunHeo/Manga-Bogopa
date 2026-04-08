import logging
import re
import time
from typing import List
from src.data_models import PageData

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


def _send_with_retry(chat_session, request_text):
    """지수 백오프로 API 요청을 재시도합니다."""
    for attempt in range(MAX_RETRIES):
        try:
            response = chat_session.send_message(request_text)
            return response
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY ** (attempt + 1)
                logger.info(f"번역 API 오류 (시도 {attempt + 1}/{MAX_RETRIES}): {e}. {delay}초 후 재시도...")
                time.sleep(delay)
            else:
                raise


def _parse_translation_line(line):
    """응답 형식에서 번역된 텍스트만 추출합니다. (예: '1.(번역된 문장)')"""
    match = re.search(r'\((.*)\)', line)
    cleaned_text = match.group(1) if match else line.split('.', 1)[-1].strip()
    if cleaned_text == "번역 불가":
        return ""
    return cleaned_text.replace("...", "⋯").replace("…", "⋯")


def translate_pages_in_batch(chat_session, batch_page_data: List[PageData]):
    """페이지 데이터 리스트를 받아 모든 텍스트를 일괄 번역하고 업데이트합니다."""
    logger.info(f"{len(batch_page_data)} 페이지 데이터 전체 번역 요청...")

    # 번역할 텍스트와 해당 텍스트가 속한 원본 객체를 수집
    texts_to_translate = []
    for page_data in batch_page_data:
        for bubble in page_data.speech_bubbles:
            if bubble.text_element.original_text:
                texts_to_translate.append({
                    "source_element": bubble.text_element,
                    "text": bubble.text_element.original_text
                })
        for freeform_text in page_data.freeform_texts:
            if freeform_text.original_text:
                texts_to_translate.append({
                    "source_element": freeform_text,
                    "text": freeform_text.original_text
                })

    if not texts_to_translate:
        logger.info("번역할 텍스트가 없습니다.")
        return batch_page_data

    # API 요청 형식에 맞게 텍스트 조합
    request_text = "\n".join([f"{i + 1}.({item['text']})" for i, item in enumerate(texts_to_translate)])

    try:
        response = _send_with_retry(chat_session, request_text)
        lines = response.text.strip().split('\n')

        if len(lines) != len(texts_to_translate):
            logger.warning(f"응답 줄 수({len(lines)})와 요청 수({len(texts_to_translate)})가 불일치합니다.")

        # 번역 결과를 먼저 모두 객체에 저장
        for i, item in enumerate(texts_to_translate):
            if i < len(lines):
                item['source_element'].translated_text = _parse_translation_line(lines[i])
            else:
                item['source_element'].translated_text = ""

        # 번역 결과가 없는 요소들을 제거
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
        for item in texts_to_translate:
            item['source_element'].translated_text = "번역 불가"

    logger.info("번역 완료.")
    return batch_page_data
