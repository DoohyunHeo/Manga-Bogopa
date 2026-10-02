"""겹쳐 읽기 OCR — 원조 manga-ocr와 Hayai가 같게 읽으면 그대로 쓰고, 다르게 읽은 조각만 VL-For-Manga로 다시 읽는다.

주간지 흑백 본문 284조각 비교(2026-09)에서 manga-ocr 84.9%, Hayai 85.6%, VL-For-Manga 89.4%였고,
이 방식이 92.6%로 가장 정확했다 (VL-For-Manga는 조각의 약 20%만 읽는다). 원격 모델 코드를 쓰는
두 모델은 검증한 버전으로 고정한다 — 새 버전이 설치된 transformers와 어긋나 갑자기 깨지지 않도록.
실제 쪽에서는 조각의 약 35%가 VL-For-Manga로 가서, 읽기 시간의 대부분이 이 다시 읽기였다 (VlMangaOcr 참고).
"""
import logging
import re
import unicodedata

import torch

from src.batch_manga_ocr import BatchMangaOcr

logger = logging.getLogger(__name__)

MANGA_OCR_MODEL = "kha-white/manga-ocr-base"
HAYAI_MODEL = ("JustANormalTinkerer/hayai-ocr-v2.5-nova", "e34d7755ed11e626c5ba39544af5d66f20ee57cc")
HAYAI_PROCESSOR = ("google/siglip2-base-patch16-naflex", "b53b807d3a2d5e2b3911292f2d69e5341cdc064c")
VL_MODEL = ("jzhang533/PaddleOCR-VL-For-Manga", "1e8aa5f1dd90cc86fe9137c9c0b26ebde613cfe8")

_SPACE = re.compile(r"\s+")
_DOTS = re.compile(r"[.・･‥…︙︰⋯]+")
_DASH = re.compile(r"[ー―—─‐‑–－━-]+")
_WAVE = re.compile(r"[〜~～]+")


def agreement_key(text):
    """두 판독이 같은지 볼 때 쓰는 모양 — 전각/반각·공백·말줄임표·줄표 모양과 길이 차이는 무시한다."""
    text = _SPACE.sub("", unicodedata.normalize("NFKC", text or ""))
    return _WAVE.sub("〜", _DASH.sub("ー", _DOTS.sub("…", text)))


def _from_pretrained(loader, repo, revision, **kwargs):
    """받아 둔 파일을 먼저 쓰고, 없으면 내려받는다."""
    try:
        return loader(repo, revision=revision, local_files_only=True, **kwargs)
    except Exception:  # noqa: BLE001 — 받아 둔 파일이 없으면 아래에서 내려받는다
        pass
    return loader(repo, revision=revision, **kwargs)


class HayaiOcr:
    """Hayai OCR v2.5 — 16조각씩 묶어 읽는다 (묶어 읽어도 한 조각씩 읽을 때와 결과가 같다)."""

    BATCH = 16
    MAX_PATCHES = 512

    def __init__(self, device):
        from transformers import AutoModel, AutoProcessor, PreTrainedTokenizerFast
        self.device = device
        self.model = _from_pretrained(AutoModel.from_pretrained, *HAYAI_MODEL, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        if str(device).startswith("cuda"):
            # generate가 fp16 autocast 안에서 돈다 — 가중치를 fp32로 두면 연산마다 변환이 붙고 RMSNorm이 느린 길로 간다.
            # fp16으로 두면 2배 빠르고 판독은 같았다 (정답 284조각·실제 쪽 275조각 모두)
            self.model = self.model.half()
        self.tokenizer = _from_pretrained(PreTrainedTokenizerFast.from_pretrained, *HAYAI_MODEL)
        self.processor = _from_pretrained(AutoProcessor.from_pretrained, *HAYAI_PROCESSOR)

    def __call__(self, images):
        texts = []
        for start in range(0, len(images), self.BATCH):
            inputs = self.processor(images=images[start:start + self.BATCH], max_num_patches=self.MAX_PATCHES,
                                    return_tensors="pt").to(self.device)
            with torch.inference_mode():
                texts += self.model.generate(
                    pixel_values=inputs["pixel_values"], pixel_attention_mask=inputs["pixel_attention_mask"],
                    spatial_shapes=inputs["spatial_shapes"], tokenizer=self.tokenizer,
                    max_new_tokens=128, repetition_penalty=1.0,
                )
        return [_SPACE.sub("", text) for text in texts]


class VlMangaOcr:
    """PaddleOCR-VL-For-Manga — 만화 조각으로 추가 학습한 0.9B 시각언어모델, 한 조각씩 읽는다.

    글자 하나를 만들 때 GPU가 하는 일은 2ms 남짓인데, 작은 연산 수백 개를 CPU가 하나씩 걸어 주느라 20~40ms가 들었다.
    그래서 첫 계산(그림과 질문)은 모델에 그대로 넘기고, 그 뒤 한 글자씩 만드는 걸음은 CUDA 그래프로 한 번 녹화해 재생한다.
    실제 쪽 97조각에서 조각당 889ms → 189ms였고 판독은 96조각이 같았다 (한 조각은 소수점 계산 차이).
    여러 조각을 묶어 읽으면 이 환경에서는 그림끼리 섞여 엉뚱한 글자가 나와 쓰지 않는다.
    """

    MAX_NEW_TOKENS = 256
    CACHE_LEN = 2048  # 질문(그림 포함)과 판독을 합친 최대 길이 — 넘는 큰 조각은 generate로 읽는다

    def __init__(self, device):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.device = device
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.model = _from_pretrained(AutoModelForCausalLM.from_pretrained, *VL_MODEL, trust_remote_code=True,
                                      torch_dtype=dtype)
        self.model = self.model.to(device).eval()
        self.processor = _from_pretrained(AutoProcessor.from_pretrained, *VL_MODEL, trust_remote_code=True)
        self._eos = self.processor.tokenizer.eos_token_id
        self._graph = None
        self._cache = None
        if str(device).startswith("cuda"):
            from transformers.cache_utils import StaticCache
            self._cache = StaticCache(config=self.model.config, max_cache_len=self.CACHE_LEN)
            self._step_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
            self._step_cache_position = torch.zeros((1,), dtype=torch.long, device=device)
            self._step_position_ids = torch.zeros((3, 1, 1), dtype=torch.long, device=device)

    def __call__(self, images):
        texts = []
        for image in images:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "OCR:"}]}]
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                        return_dict=True, return_tensors="pt").to(self.device)
            text = self._read_with_graph(inputs) if self._cache is not None else None
            if text is None:
                with torch.inference_mode():
                    # 모델 설정에 use_cache=False가 들어 있어 토큰마다 그림부터 다시 계산한다 — 켜면 2.2배 빠르고 결과는 같다
                    out = self.model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKENS, do_sample=False,
                                              use_cache=True)
                text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            texts.append(_SPACE.sub("", text))
        return texts

    def _decode_step(self):
        """이미 읽은 글자 뒤에 한 글자 — 언어 모델과 출력층만 (그림은 첫 계산에서 캐시에 들어가 있다)."""
        out = self.model.model(input_ids=self._step_ids, position_ids=self._step_position_ids,
                               past_key_values=self._cache, cache_position=self._step_cache_position, use_cache=True)
        return self.model.lm_head(out.last_hidden_state[:, -1])

    def _capture(self):
        """한 글자 걸음을 CUDA 그래프로 녹화한다 — 캐시 자리가 잡힌 뒤 한 번만 (녹화 중 쓴 캐시는 곧 덮어쓴다)."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._decode_step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step_logits = self._decode_step()
        self._graph = graph

    def _read_with_graph(self, inputs):
        """generate(do_sample=False)와 같은 욕심 디코딩을 CUDA 그래프로. 쓸 수 없으면 None (generate로 읽는다)."""
        length = inputs["input_ids"].shape[1]
        if length + self.MAX_NEW_TOKENS > self.CACHE_LEN:
            return None
        try:
            with torch.inference_mode():
                self._cache.reset()
                self.model.rope_deltas = None
                out = self.model(**inputs, past_key_values=self._cache, use_cache=True,
                                 cache_position=torch.arange(length, device=self.device))
                deltas = self.model.rope_deltas.view(-1)[:1]  # 그림 뒤 글자 위치 = 캐시 위치 + 이 값
                token = out.logits[:, -1].argmax(-1)
                if self._graph is None:
                    self._capture()
                tokens = []
                for step in range(self.MAX_NEW_TOKENS):
                    value = int(token)
                    if value == self._eos:
                        break
                    tokens.append(value)
                    if step == self.MAX_NEW_TOKENS - 1:
                        break
                    position = length + step
                    self._step_ids.copy_(token.view(1, 1))
                    self._step_cache_position.fill_(position)
                    self._step_position_ids.copy_((deltas + position).view(1, 1, 1).expand(3, 1, 1))
                    self._graph.replay()
                    token = self._step_logits.argmax(-1)
        except Exception as exc:  # noqa: BLE001 — 그래프를 못 쓰는 환경이면 예전 방식으로 계속 읽는다
            logger.warning(f"VL-For-Manga 빠른 읽기를 쓰지 못해 예전 방식으로 읽습니다: {exc}")
            self._cache = self._graph = None
            return None
        return self.processor.decode(tokens, skip_special_tokens=True)


class CascadeOcr:
    """겹쳐 읽기 — BatchMangaOcr와 같은 방식으로 부른다: ocr(PIL 목록) → 판독 목록."""

    def __init__(self, device, batch_size):
        self.manga = BatchMangaOcr(model_name=MANGA_OCR_MODEL, batch_size=batch_size)
        self.hayai = HayaiOcr(device)
        self.vl = VlMangaOcr(device)
        # 마지막 호출의 조각별 (manga-ocr, Hayai) 판독 — 오탐 거르기가 두 판독이 서로 닮았는지 본다
        self.last_readings = []

    def __call__(self, images):
        images = [image.convert("RGB") for image in images]
        results = list(self.manga(images))
        second = self.hayai(images)
        self.last_readings = list(zip(results, second))
        disputed = [i for i, (a, b) in enumerate(zip(results, second)) if agreement_key(a) != agreement_key(b)]
        if disputed:
            for i, text in zip(disputed, self.vl([images[i] for i in disputed])):
                results[i] = text or results[i]
        logger.info(f"겹쳐 읽기: {len(images)}조각 중 서로 다르게 읽은 {len(disputed)}조각을 한 번 더 읽었습니다.")
        return results
