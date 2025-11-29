from __future__ import annotations

import inspect
from typing import List, Sequence

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from src import config


class BatchMangaOcr:
    """Batch inference helper that wraps PaddleOCR-VL-For-Manga."""

    def __init__(
        self,
        model_id: str | None = None,
        processor_id: str | None = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size or config.OCR_BATCH_SIZE
        self.max_new_tokens = max_new_tokens or config.PADDLE_OCR_MAX_NEW_TOKENS

        self.model = self._load_model(model_id or config.PADDLE_OCR_MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(
            processor_id or config.PADDLE_OCR_PROCESSOR_ID,
            trust_remote_code=True,
        )

        if self.model.generation_config.pad_token_id is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer and tokenizer.eos_token_id is not None:
                self.model.generation_config.pad_token_id = tokenizer.eos_token_id

        self.prompt = config.PADDLE_OCR_TASK_PROMPT
        self._forward_arg_names = None
        self._forward_accepts_kwargs = False
        self._ignored_input_keys = set()

    def __call__(self, images):
        """Allow class instances to be called like a function."""
        if isinstance(images, list):
            return self.ocr_batch(images)
        return self.ocr_batch([images])[0]

    def ocr_batch(self, images: Sequence[Image.Image]) -> List[str]:
        """Runs OCR on a batch of PIL images."""
        if not images:
            return []

        pil_images = [self._ensure_pil(image) for image in images]
        results: List[str] = []
        for start in range(0, len(pil_images), self.batch_size):
            batch_images = pil_images[start : start + self.batch_size]
            results.extend(self._infer_batch(batch_images))
        return results

    def _load_model(self, model_id: str):
        dtype = self._select_dtype()
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if config.PADDLE_OCR_USE_FLASH_ATTN:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.to(self.device)
        model.eval()
        return model

    def _select_dtype(self):
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 7:
                return torch.float16
        return torch.float32

    def _infer_batch(self, images: Sequence[Image.Image]) -> List[str]:
        chat_texts = [
            self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for image in images
        ]

        inputs = self.processor(
            text=chat_texts,
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        model_inputs = self._filter_supported_inputs(inputs)

        attention_mask = inputs.get("attention_mask")
        with torch.inference_mode():
            sequences = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated_token_ids = []
        for idx, seq in enumerate(sequences):
            if attention_mask is not None:
                input_length = int(attention_mask[idx].sum().item())
            else:
                input_length = inputs["input_ids"].shape[1]
            generated_token_ids.append(seq[input_length:].cpu().tolist())

        decoded_texts = self.processor.batch_decode(
            generated_token_ids, skip_special_tokens=True
        )
        return [self._clean_response(text) for text in decoded_texts]

    def _clean_response(self, text: str) -> str:
        text = text.strip()
        if text.startswith(self.prompt):
            text = text[len(self.prompt) :].strip()
        return text

    def _filter_supported_inputs(self, inputs):
        if self._forward_arg_names is None:
            try:
                signature = inspect.signature(self.model.forward)
                self._forward_accepts_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in signature.parameters.values()
                )
                self._forward_arg_names = set(signature.parameters.keys())
            except (TypeError, ValueError):
                self._forward_arg_names = set()
                self._forward_accepts_kwargs = True

        if self._forward_accepts_kwargs:
            return inputs

        if not self._forward_arg_names:
            return inputs

        filtered = {k: v for k, v in inputs.items() if k in self._forward_arg_names}
        if not filtered:
            return inputs

        for key in inputs:
            if key not in filtered and key not in self._ignored_input_keys:
                print(f"경고: OCR 모델이 '{key}' 입력을 지원하지 않아 무시합니다.")
                self._ignored_input_keys.add(key)
        return filtered

    @staticmethod
    def _ensure_pil(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.fromarray(image).convert("RGB")
