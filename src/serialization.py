import json
import os
from typing import List

import numpy as np

from src.data_models import PageData, SpeechBubble, TextElement


class _PageDataEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PageData):
            d = o.__dict__.copy()
            d.pop('image_rgb', None)
            return d
        if isinstance(o, (SpeechBubble, TextElement)):
            return o.__dict__
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_page_data_json(all_page_data: List[PageData], path: str) -> None:
    """PageData 리스트를 JSON 파일로 저장합니다."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_page_data, f, ensure_ascii=False, indent=4, cls=_PageDataEncoder)


def load_page_data_json(path: str) -> List[PageData]:
    """JSON 파일에서 PageData 리스트를 복원합니다."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [PageData.from_dict(d) for d in data]


def append_page_data_json(new_pages: List[PageData], path: str) -> None:
    """기존 JSON 파일에 새 PageData 항목들을 추가합니다. 파일이 없으면 새로 생성합니다."""
    existing = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing + json.loads(json.dumps(new_pages, cls=_PageDataEncoder)),
                  f, ensure_ascii=False, indent=4)
