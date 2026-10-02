import json
import os
from dataclasses import fields
from typing import List

import numpy as np

from src.data_models import PageData, SpeechBubble, TextElement
from src.utils import replace_file


_TEXT_FIELDS = {f.name for f in fields(TextElement)}
_PAGE_FIELDS = {f.name for f in fields(PageData)}


class _PageDataEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PageData):
            d = {k: v for k, v in o.__dict__.items() if k in _PAGE_FIELDS}  # 식자용 임시 값은 빼고
            d.pop('image_rgb', None)
            return d
        if isinstance(o, TextElement):
            # 식자 때 붙이는 임시 값(style_inverted 등)은 대사집에 쓰지 않는다. 원문 글자 모양(look — 테두리 두께 rim_px 포함)은
            # 지우기 때마다 원본·지운 그림으로 다시 재는 실행 중 값이라 비워 둔다 (쓰면 다음 실행이 낡은 값을 이어받는다)
            d = {k: v for k, v in o.__dict__.items() if k in _TEXT_FIELDS}
            d["look"] = None
            d.pop("room", None)  # 식자 때 잰 말풍선 안쪽 모양
            return d
        if isinstance(o, SpeechBubble):
            return o.__dict__
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _write_json_atomic(obj, path: str, cls=None) -> None:
    """임시 파일에 다 쓴 뒤 바꿔치기한다 — 중간에 꺼져도 기존 파일이 깨지지 않는다."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4, cls=cls)
    replace_file(tmp_path, path)  # 웹 화면이 같은 파일을 읽는 순간과 겹쳐도 실패하지 않게


def save_page_data_json(all_page_data: List[PageData], path: str) -> None:
    """PageData 리스트를 JSON 파일로 저장합니다."""
    _write_json_atomic(all_page_data, path, cls=_PageDataEncoder)


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

    _write_json_atomic(existing + json.loads(json.dumps(new_pages, cls=_PageDataEncoder)), path)
