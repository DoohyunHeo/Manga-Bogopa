from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Dict, List, Optional

import numpy as np


class Attachment(StrEnum):
    """말풍선 말꼬리 방향"""
    LEFT = 'left'
    RIGHT = 'right'
    NONE = 'none'


class TranslationStatus(StrEnum):
    """대사 하나의 번역 상태"""
    PENDING = 'pending'        # 아직 요청 전
    TRANSLATED = 'translated'
    FAILED = 'failed'          # 응답에서 빠졌거나 요청이 실패함 — 다음 실행에서 이 대사만 다시 요청
    EXCLUDED = 'excluded'      # 번역할 수 없다는 답(깨진 글자 등) — 원문을 지우지 않고 둔다


@dataclass
class TextElement:
    """개별 텍스트 요소(말풍선 안 텍스트, 말풍선 밖 텍스트 등)의 정보를 담는 데이터 클래스"""
    text_box: List[int]
    original_text: str
    font_size: int
    font_style: str
    angle: int
    translated_text: Optional[str] = None
    font_char_ratio: Optional[float] = None
    font_style_raw: Optional[str] = None                   # 모양 모델이 1위로 본 글씨 모양 (문턱 규칙 전, 책 단위 판정)
    font_style_reason: Optional[str] = None                # 최종 글씨체를 고른 이유 (font_analysis.STYLE_REASONS)
    font_style_scores: Optional[Dict[str, float]] = None   # 모양 모델이 본 상위 3개 글씨 모양 확률
    look: Optional[Dict] = None                            # 원문 글자 모양 실측 — 지우기 때마다 다시 잰다 (font_attributes)
    furigana: Optional[List[str]] = None                   # 후리가나 '본문(읽기)' 목록 — 번역 요청에 인명 읽기로 싣는다
    attachment: Attachment = Attachment.NONE
    translation_status: TranslationStatus = TranslationStatus.PENDING
    # 말풍선 안쪽 모양 (text_fitting.BubbleRoom) — 식자 때 지운 쪽에서 재어 붙이고 대사집에는 쓰지 않는다
    room: Optional[object] = field(default=None, compare=False, repr=False)

    @property
    def needs_translation(self) -> bool:
        return self.translation_status in (TranslationStatus.PENDING, TranslationStatus.FAILED)

    @classmethod
    def from_dict(cls, d: dict) -> "TextElement":
        # 저장(serialization)은 __dict__를 자동 직렬화하므로 복원도 필드 목록을
        # 순회해 대칭을 유지한다 — 필드를 추가해도 재로드 시 조용히 유실되지 않는다.
        # 필드에 없는 키는 버린다 (그런 키가 든 대사집도 그대로 읽힌다).
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kwargs["attachment"] = Attachment(d.get("attachment", Attachment.NONE.value))
        # 상태가 없으면 아직 요청 전(PENDING)
        kwargs["translation_status"] = TranslationStatus(d.get("translation_status", TranslationStatus.PENDING.value))
        return cls(**kwargs)


@dataclass
class SpeechBubble:
    """말풍선과 그 안의 텍스트 요소 정보를 담는 데이터 클래스"""
    bubble_box: List[int]
    text_element: TextElement
    attachment: Attachment

    @classmethod
    def from_dict(cls, d: dict) -> "SpeechBubble":
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kwargs["text_element"] = TextElement.from_dict(d["text_element"])
        kwargs["attachment"] = Attachment(d["attachment"])
        return cls(**kwargs)


@dataclass
class PageData:
    """페이지 한 장의 모든 정보를 담는 데이터 클래스"""
    source_page: str
    image_rgb: Optional[np.ndarray] = None
    speech_bubbles: List[SpeechBubble] = field(default_factory=list)
    freeform_texts: List[TextElement] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "PageData":
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kwargs["image_rgb"] = None
        kwargs["speech_bubbles"] = [SpeechBubble.from_dict(sb) for sb in d.get("speech_bubbles", [])]
        kwargs["freeform_texts"] = [TextElement.from_dict(ft) for ft in d.get("freeform_texts", [])]
        return cls(**kwargs)

    def text_elements(self) -> List[TextElement]:
        return [bubble.text_element for bubble in self.speech_bubbles] + list(self.freeform_texts)

    @property
    def is_translated(self) -> bool:
        """모든 대사가 번역됐거나 번역 제외로 정해졌는지 (실패·대기가 남으면 False)."""
        return not any(element.needs_translation for element in self.text_elements())

    def typeset_view(self) -> "PageData":
        """번역된 대사만 남긴 사본 — 지우기·식자는 이것만 다룬다 (제외된 대사는 원문 유지)."""
        done = TranslationStatus.TRANSLATED
        return PageData(
            source_page=self.source_page,
            image_rgb=self.image_rgb,
            speech_bubbles=[b for b in self.speech_bubbles if b.text_element.translation_status == done],
            freeform_texts=[t for t in self.freeform_texts if t.translation_status == done],
        )
