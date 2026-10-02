import logging
from contextlib import nullcontext

import torch
from manga_ocr.ocr import MangaOcr, MangaOcrModel
from tqdm import tqdm
from transformers import AutoTokenizer, ViTImageProcessor

from src import config

logger = logging.getLogger(__name__)


class BatchMangaOcr(MangaOcr):
    # 원조 manga-ocr — 주간지 흑백 본문 비교(2026-09)에서 manga-ocr-base-2025보다 정확했다 (84.9% 대 79.2%,
    # 말풍선 밖 글자 66% 대 49%)
    def __init__(self, model_name="kha-white/manga-ocr-base", device=None, batch_size=64):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_beams = max(1, int(config.OCR_NUM_BEAMS))

        self.processor, self.tokenizer, self.model, local_only = self._load_pretrained_components(model_name)
        self.model.to(self.device)

        if self.device.type == "cuda":
            logger.info("BatchMangaOcr: FP16 mixed-precision inference enabled.")
            self.model = self.model.half()

        # torch.compile은 사용하지 않는다: generate() 경로는 컴파일을 우회해 효과가
        # 없고, Windows(트리톤 부재) 환경에서 실제 컴파일이 일어나면 크래시한다.

        logger.info(
            "BatchMangaOcr ready. device=%s batch_size=%s num_beams=%s local_only=%s",
            self.device,
            self.batch_size,
            self.num_beams,
            local_only,
        )

    def _load_pretrained_components(self, model_name):
        # 받아 둔 파일을 먼저 쓰고, 없으면 Hugging Face에서 내려받는다
        last_error = None
        for local_only in (True, False):
            load_kwargs = {"local_files_only": True} if local_only else {}
            try:
                processor = ViTImageProcessor.from_pretrained(model_name, **load_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
                model = MangaOcrModel.from_pretrained(model_name, **load_kwargs)
                if local_only:
                    logger.info("BatchMangaOcr: OCR artifacts loaded from local cache only.")
                return processor, tokenizer, model, local_only
            except Exception as exc:
                last_error = exc
                if local_only:
                    logger.warning(
                        "BatchMangaOcr: local cache load failed, retrying default Hugging Face resolution -> %s",
                        exc,
                    )

        raise last_error

    def _autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def __call__(self, image_list):
        """PIL 그림 목록 → 판독 목록."""
        return self.ocr_batch(image_list)

    def ocr_batch(self, image_list, max_length=128):
        results = []
        for i in tqdm(range(0, len(image_list), self.batch_size), desc="OCR"):
            batch_images = image_list[i:i + self.batch_size]
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                with self._autocast_context():
                    generated_ids = self.model.generate(
                        inputs.pixel_values,
                        max_length=max_length,
                        num_beams=self.num_beams,
                        early_stopping=True,
                    )

            texts = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            texts = [text.replace(" ", "") for text in texts]
            results.extend(texts)

        return results
