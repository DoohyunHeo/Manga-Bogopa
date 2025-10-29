from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class TextElement:
    """개별 텍스트 요소(말풍선 안 텍스트, 자유 텍스트 등)의 정보를 담는 데이터 클래스"""
    text_box: List[int]
    original_text: str
    font_size: int
    font_style: str
    angle: int
    translated_text: Optional[str] = None


@dataclass
class SpeechBubble:
    """말풍선과 그 안의 텍스트 요소 정보를 담는 데이터 클래스"""
    bubble_box: List[int]
    text_element: TextElement
    attachment: str


@dataclass
class PageData:
    """페이지 한 장의 모든 정보를 담는 데이터 클래스"""
    source_page: str
    image_rgb: np.ndarray
    speech_bubbles: List[SpeechBubble] = field(default_factory=list)
    freeform_texts: List[TextElement] = field(default_factory=list)
