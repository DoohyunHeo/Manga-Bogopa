"""글자 모양 마스크 — comic-text-mask (TareHimself, MIT, UNet·ResNet18 57MB).

말풍선 밖 글자는 이 모양대로 지우고, 말풍선 안팎 글자 모두 원문 글자 모양(극성·외곽선) 실측과
말풍선 안 남은 글자 조각 흡수에 쓴다 (src/inpainter.py).

글자 조각을 384 정사각에 비율을 지켜 넣고(남는 곳은 0) 글자일 확률이 문턱(0.5)을 넘는 픽셀을 글자로 본다.
정규화는 모델 안에 들어 있어 uint8 RGB를 그대로 넣는다. 검증한 버전으로 고정한다.
길쭉한 조각(긴 변 > 짧은 변 2배)은 통째로 줄이면 작은 글자가 뭉개지므로, 짧은 변의 2배 길이 조각으로
반씩 겹쳐 잘라 따로 보고 합친다.
"""
import json
import logging

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

TEXT_MASK_MODEL = ("TareHimself/comic-text-mask", "ebd37f1a9ae9298519678fb96a796573fb602977")


def _download(filename):
    """받아 둔 파일을 먼저 쓰고, 없으면 내려받는다."""
    repo, revision = TEXT_MASK_MODEL
    try:
        return hf_hub_download(repo, filename, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 — 받아 둔 파일이 없으면 아래에서 내려받는다
        return hf_hub_download(repo, filename, revision=revision)


def _tiles(h, w):
    """조각을 나눠 볼 영역들 (y1, y2, x1, x2) — 길쭉하지 않으면 통째로 하나."""
    short, long_ = min(h, w), max(h, w)
    if long_ <= 2 * short:
        return [(0, h, 0, w)]
    tile, step = 2 * short, short
    starts = list(range(0, long_ - tile + 1, step))
    if starts[-1] != long_ - tile:
        starts.append(long_ - tile)
    if w >= h:
        return [(0, h, s, s + tile) for s in starts]
    return [(s, s + tile, 0, w) for s in starts]


class TextMaskModel:
    """글자 조각 여러 개를 16장씩 묶어 한 번에 본다 (묶어도 한 장씩 볼 때와 마스크가 같다)."""

    BATCH = 16

    def __init__(self, device):
        with open(_download("tm_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.size = int(meta["imgsz"])
        self.threshold = float(meta["threshold"])
        self.device = device
        self.model = torch.jit.load(_download("model.pt"), map_location=device).eval()

    def _predict(self, images):
        """RGB 그림들 → 그림마다 같은 크기의 글자 마스크(bool)."""
        masks = []
        for start in range(0, len(images), self.BATCH):
            squares, places = [], []
            for image in images[start:start + self.BATCH]:
                h, w = image.shape[:2]
                scale = self.size / max(h, w)
                nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
                oy, ox = (self.size - nh) // 2, (self.size - nw) // 2
                square = np.zeros((self.size, self.size, 3), np.uint8)
                square[oy:oy + nh, ox:ox + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
                squares.append(square)
                places.append((oy, ox, nh, nw, h, w))
            batch = torch.from_numpy(np.stack(squares)).permute(0, 3, 1, 2).to(self.device)
            with torch.inference_mode():
                prob = self.model(batch)[:, 0].float().cpu().numpy()
            for p, (oy, ox, nh, nw, h, w) in zip(prob, places):
                local = (p[oy:oy + nh, ox:ox + nw] > self.threshold).astype(np.uint8)
                masks.append(cv2.resize(local, (w, h), interpolation=cv2.INTER_NEAREST) > 0)
        return masks

    def __call__(self, crops):
        """RGB 조각 리스트 → 조각마다 같은 크기의 글자 마스크(bool) 리스트."""
        jobs = [(index, area) for index, crop in enumerate(crops) for area in _tiles(*crop.shape[:2])]
        pieces = self._predict([crops[index][y1:y2, x1:x2] for index, (y1, y2, x1, x2) in jobs])
        masks = [np.zeros(crop.shape[:2], bool) for crop in crops]
        for (index, (y1, y2, x1, x2)), piece in zip(jobs, pieces):
            masks[index][y1:y2, x1:x2] |= piece
        return masks
