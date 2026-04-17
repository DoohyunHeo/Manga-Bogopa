import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from manga_ocr.ocr import MangaOcr, MangaOcrModel, post_process
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, ViTImageProcessor

from src import config

logger = logging.getLogger(__name__)


class BatchMangaOcr(MangaOcr):
    def __init__(self, model_name="jzhang533/manga-ocr-base-2025", device=None, batch_size=64):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_beams = max(1, int(getattr(config, "OCR_NUM_BEAMS", 2)))
        self.prefer_local_files = bool(getattr(config, "OCR_PREFER_LOCAL_FILES", True))
        self.warmup_on_load = bool(getattr(config, "OCR_WARMUP_ON_LOAD", False))

        self.processor, self.tokenizer, self.model, local_only = self._load_pretrained_components(model_name)
        self.model.to(self.device)

        if self.device.type == "cuda":
            logger.info("BatchMangaOcr: FP16 mixed-precision inference enabled.")
            self.model = self.model.half()

        torch_version = torch.__version__
        if sys.platform != "darwin" and int(torch_version.split(".")[0]) >= 2:
            try:
                logger.info("BatchMangaOcr: enabling torch.compile() (PyTorch %s)", torch_version)
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as exc:
                logger.warning("BatchMangaOcr: torch.compile() disabled -> %s", exc)

        if self.warmup_on_load:
            self._warmup_model()

        logger.info(
            "BatchMangaOcr ready. device=%s batch_size=%s num_beams=%s local_only=%s warmup=%s",
            self.device,
            self.batch_size,
            self.num_beams,
            local_only,
            self.warmup_on_load,
        )

    def _load_pretrained_components(self, model_name):
        load_modes = [False]
        if self.prefer_local_files:
            load_modes = [True, False]

        last_error = None
        for local_only in load_modes:
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

    def _run_single_ocr(self, image: Image.Image) -> str:
        image = image.convert("L").convert("RGB")
        pixel_values = self._preprocess(image)
        with torch.inference_mode():
            with self._autocast_context():
                generated_ids = self.model.generate(
                    pixel_values[None].to(self.device),
                    max_length=300,
                )[0].cpu()
        return post_process(self.tokenizer.decode(generated_ids, skip_special_tokens=True))

    def _warmup_model(self):
        try:
            self._run_single_ocr(Image.new("RGB", (32, 32), color="white"))
        except Exception as exc:
            logger.warning("BatchMangaOcr: warmup skipped after failure -> %s", exc)

    def __call__(self, image_list):
        if isinstance(image_list, list):
            return self.ocr_batch(image_list)
        if isinstance(image_list, (str, Path)):
            image = Image.open(image_list)
        elif isinstance(image_list, Image.Image):
            image = image_list
        else:
            raise ValueError(f"img_or_path must be a path or PIL.Image, instead got: {image_list}")
        return self._run_single_ocr(image)

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
