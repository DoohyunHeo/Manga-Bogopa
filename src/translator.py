import logging
import re
import time
from typing import Callable, List, Optional

from google import genai
from google.genai import types as genai_types

from src.data_models import PageData
from src.progress import EventLevel, PipelinePhase, ProgressEvent

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
# 503(서버 과부하)은 수십 초 단위로 풀리는 경우가 많아 점증 대기 (총 ~2분)
RETRY_DELAYS = [3, 8, 20, 45]

# Ellipsis normalization: Gemini emits ..., …, ．．．, 。。。, ・・・, 점 사이 공백 등 다양한 형태.
# 모두 U+22EF(⋯)로 통일.
_ELLIPSIS_SPACED = re.compile(r'(?<=[.。．・])\s+(?=[.。．・])')
_ELLIPSIS_RUNS = re.compile(
    r'[…‥]+'                      # …  ‥
    r'|[.。．・]{2,}'           # .. .., 。。。, ．．．, ・・・
)


class TranslatorSession:
    """google-genai 기반 번역 세션.

    - 시스템 프롬프트는 system_instruction으로 전달 (히스토리 오염 없음)
    - 배치 간 맥락(인명·말투 일관성)을 위해 대화 히스토리를 직접 관리
    - 히스토리는 max_history_exchanges 쌍으로 상한 (장편 작업 시 토큰 폭주 방지)
    - Gemini 3 계열에는 thinking_level(번역 품질/속도 트레이드오프) 전달
    """

    def __init__(self, api_key: str, model: str, system_prompt: str,
                 thinking_level: str = "default", max_history_exchanges: int = 6):
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.thinking_level = str(thinking_level or "default").lower()
        self.max_history_exchanges = max(1, int(max_history_exchanges))
        self._history: List[genai_types.Content] = []

    def _generation_config(self) -> genai_types.GenerateContentConfig:
        cfg = genai_types.GenerateContentConfig(system_instruction=self.system_prompt)
        if self.thinking_level in ("low", "high") and "gemini-3" in self.model.lower():
            cfg.thinking_config = genai_types.ThinkingConfig(thinking_level=self.thinking_level)
        return cfg

    def send_message_stream(self, text: str):
        contents = self._history + [
            genai_types.Content(role="user", parts=[genai_types.Part(text=text)])
        ]
        return self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=self._generation_config(),
        )

    def record_exchange(self, user_text: str, model_text: str) -> None:
        self._history.append(genai_types.Content(role="user", parts=[genai_types.Part(text=user_text)]))
        self._history.append(genai_types.Content(role="model", parts=[genai_types.Part(text=model_text)]))
        max_entries = self.max_history_exchanges * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]


def _normalize_ellipsis(text: str) -> str:
    if not text:
        return text
    collapsed = _ELLIPSIS_SPACED.sub('', text)
    return _ELLIPSIS_RUNS.sub('⋯', collapsed)


def _stream_with_retry(chat_session: TranslatorSession, request_text, callback=None):
    """스트리밍으로 API 요청을 보내고 응답을 조합합니다. 실패 시 재시도."""
    for attempt in range(MAX_RETRIES):
        try:
            response_stream = chat_session.send_message_stream(request_text)

            chunks = []
            lines_received = 0
            start_time = time.time()
            last_report = 0

            for chunk in response_stream:
                piece = chunk.text or ""
                if not piece:
                    continue
                chunks.append(piece)
                lines_received += piece.count('\n')
                elapsed = time.time() - start_time

                # 2초마다 또는 새 줄이 올 때 상태 보고
                if callback and (elapsed - last_report >= 2 or piece.count('\n') > 0):
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
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
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


def _apply_response_lines(response_text, keyed_items):
    """응답 텍스트를 파싱해 keyed_items에 번역을 채웁니다. 알 수 없는 키를 반환."""
    unknown_keys = []
    for resp_line in response_text.strip().split('\n'):
        resp_line = resp_line.strip()
        if not resp_line:
            continue
        key, translated = _parse_translation_line(resp_line)
        if key and key in keyed_items:
            keyed_items[key].translated_text = translated
        elif key:
            unknown_keys.append(key)
            logger.warning(f"알 수 없는 번호: {key}")
    return unknown_keys


def _request_missing_translations(chat_session, keyed_items, callback=None):
    """첫 응답에서 누락된 항목만 모아 한 번 더 요청합니다."""
    missing_keys = [key for key, element in keyed_items.items() if element.translated_text is None]
    if not missing_keys:
        return

    logger.info(f"누락된 번역 {len(missing_keys)}개 재요청: {missing_keys[:10]}")
    if callback:
        callback(ProgressEvent(
            PipelinePhase.TRANSLATION, 0, len(missing_keys),
            f"누락된 번역 {len(missing_keys)}개 재요청 중...",
            level=EventLevel.WARNING,
        ))

    retry_lines = [f"{key}.({keyed_items[key].original_text})" for key in missing_keys]
    retry_request = (
        "다음 항목의 번역이 응답에서 누락되었습니다. "
        "같은 출력 형식(번호.(번역문))으로 누락된 항목만 다시 출력하세요.\n"
        + "\n".join(retry_lines)
    )

    try:
        retry_response = _stream_with_retry(chat_session, retry_request, callback)
        _apply_response_lines(retry_response, keyed_items)
        chat_session.record_exchange(retry_request, retry_response)
    except Exception as e:
        logger.warning(f"누락 번역 재요청 실패 (남은 항목은 원문 유지): {e}")


def translate_pages_in_batch(chat_session: TranslatorSession, batch_page_data: List[PageData],
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
        chat_session.record_exchange(request_text, response_text)

        response_lines = response_text.strip().split('\n')
        if len(response_lines) != len(keyed_items) and callback:
            logger.warning(f"응답 줄 수({len(response_lines)})와 요청 수({len(keyed_items)})가 불일치합니다.")
            callback(ProgressEvent(
                PipelinePhase.TRANSLATION, len(response_lines), len(keyed_items),
                f"번역 응답 줄 수 불일치: 요청 {len(keyed_items)}개 vs 응답 {len(response_lines)}개",
                level=EventLevel.WARNING,
                extras={"requested": len(keyed_items), "received": len(response_lines)},
            ))

        unknown_keys = _apply_response_lines(response_text, keyed_items)

        if unknown_keys and callback:
            callback(ProgressEvent(
                PipelinePhase.TRANSLATION, 0, 0,
                f"번역 응답에 알 수 없는 번호 {len(unknown_keys)}개: {', '.join(unknown_keys[:5])}",
                level=EventLevel.WARNING,
                extras={"unknown_keys": unknown_keys},
            ))

        # 누락분은 한 번 더 요청해 채운다 (그래도 없으면 원문 유지 처리).
        _request_missing_translations(chat_session, keyed_items, callback)

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
