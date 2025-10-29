import re
from typing import List
from src.data_models import PageData


def translate_pages_in_batch(chat_session, batch_page_data: List[PageData]):
    """페이지 데이터 리스트를 받아 모든 텍스트를 일괄 번역하고 업데이트합니다."""
    print(f"-> {len(batch_page_data)} 페이지 데이터 전체 번역 요청...")

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
        print("-> 번역할 텍스트가 없습니다.")
        return batch_page_data

    # API 요청 형식에 맞게 텍스트 조합
    request_text = "\n".join([f"{i + 1}.({item['text']})" for i, item in enumerate(texts_to_translate)])

    try:
        response = chat_session.send_message(request_text)
        lines = response.text.strip().split('\n')

        # 번역 결과를 먼저 모두 객체에 저장
        for i, item in enumerate(texts_to_translate):
            if i < len(lines):
                line = lines[i]
                # 응답 형식에서 번역된 텍스트만 추출 (예: "1.(번역된 문장)")
                match = re.search(r'\((.*)\)', line)
                cleaned_text = match.group(1) if match else line.split('.', 1)[-1].strip()
                # dataclass의 translated_text 필드를 직접 업데이트
                if cleaned_text == "번역 불가":
                    item['source_element'].translated_text = ""
                else:
                    item['source_element'].translated_text = cleaned_text.replace("...", "⋯").replace("…", "⋯")
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
        print(f"-> 통합 번역 중 오류 발생: {e}")
        for item in texts_to_translate:
            item['source_element'].translated_text = "번역 불가"

    print("-> 번역 완료.")
    return batch_page_data
