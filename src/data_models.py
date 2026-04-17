from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional

import numpy as np


class Attachment(StrEnum):
    """말풍선 말꼬리 방향"""
    LEFT = 'left'
    RIGHT = 'right'
    NONE = 'none'


class FontStyle(StrEnum):
    """폰트 스타일 분류"""
    POP = 'pop'
    ANGRY = 'angry'
    CUTE = 'cute'
    EMBARRASSMENT = 'embarrassment'
    HANDWRITING = 'handwriting'
    NARRATION = 'narration'
    SCARED = 'scared'
    SHOUTING = 'shouting'
    STANDARD = 'standard'


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
    font_stroke_ratio: Optional[float] = None
    font_style_confidence: Optional[float] = None
    expressive_confidence: Optional[float] = None
    attachment: Attachment = Attachment.NONE

    @classmethod
    def from_dict(cls, d: dict) -> "TextElement":
        return cls(
            text_box=d["text_box"],
            original_text=d["original_text"],
            font_size=d["font_size"],
            font_style=d["font_style"],
            angle=d["angle"],
            translated_text=d.get("translated_text"),
            font_char_ratio=d.get("font_char_ratio", d.get("font_stroke_ratio")),
            font_stroke_ratio=d.get("font_stroke_ratio"),
            font_style_confidence=d.get("font_style_confidence"),
            expressive_confidence=d.get("expressive_confidence"),
            attachment=Attachment(d.get("attachment", Attachment.NONE.value)),
        )


@dataclass
class SpeechBubble:
    """말풍선과 그 안의 텍스트 요소 정보를 담는 데이터 클래스"""
    bubble_box: List[int]
    text_element: TextElement
    attachment: Attachment

    @classmethod
    def from_dict(cls, d: dict) -> "SpeechBubble":
        return cls(
            bubble_box=d["bubble_box"],
            text_element=TextElement.from_dict(d["text_element"]),
            attachment=Attachment(d["attachment"]),
        )


@dataclass
class PageData:
    """페이지 한 장의 모든 정보를 담는 데이터 클래스"""
    source_page: str
    image_rgb: Optional[np.ndarray] = None
    speech_bubbles: List[SpeechBubble] = field(default_factory=list)
    freeform_texts: List[TextElement] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "PageData":
        return cls(
            source_page=d["source_page"],
            image_rgb=None,
            speech_bubbles=[SpeechBubble.from_dict(sb) for sb in d.get("speech_bubbles", [])],
            freeform_texts=[TextElement.from_dict(ft) for ft in d.get("freeform_texts", [])],
        )
