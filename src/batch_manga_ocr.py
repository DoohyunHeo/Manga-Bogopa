import torch
from manga_ocr import MangaOcr
from transformers import AutoTokenizer
import sys
from tqdm import tqdm


class BatchMangaOcr(MangaOcr):
    def __init__(self, model_name="kha-white/manga-ocr-base", device=None, batch_size=64):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        super().__init__(model_name, force_cpu=False)  # force_cpu=False로 명시적 GPU 사용

        # 모델을 지정된 디바이스로 이동
        self.model.to(self.device)

        # 1. FP16 (혼합 정밀도) 적용 (CUDA 사용 시)
        if self.device.type == 'cuda':
            print("BatchMangaOcr: FP16 (혼합 정밀도) 추론을 활성화합니다.")
            self.model = self.model.half()

        # 2. torch.compile() 적용 (PyTorch 2.0+ 권장)
        # torch 버전 확인
        torch_version = torch.__version__
        if sys.platform != 'darwin' and int(torch_version.split('.')[0]) >= 2:
            print(f"BatchMangaOcr: torch.compile()을 적용합니다. (PyTorch 버전: {torch_version})")
            # mode="reduce-overhead"는 작은 배치 크기에서 오버헤드를 줄여줌
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # __init__에서 한 번만 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"BatchMangaOcr 초기화 완료. (Device: {self.device}, Batch Size: {self.batch_size})")

    def __call__(self, image_list):
        if isinstance(image_list, list):
            return self.ocr_batch(image_list)
        # 단일 이미지는 기존 로직을 따르되, FP16을 위해 autocast 추가
        if self.device.type == 'cuda':
            with torch.autocast(device_type=self.device.type):
                return super().__call__(image_list)
        else:
            return super().__call__(image_list)

    def ocr_batch(self, image_list, max_length=128):
        results = []
        # 생성자에서 받은 self.batch_size 사용
        for i in tqdm(range(0, len(image_list), self.batch_size), desc="OCR"):
            batch_images = image_list[i:i + self.batch_size]

            # processor는 내부적으로 이미지 리사이즈 및 텐서 변환을 수행
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            # FP16 추론을 위해 autocast 컨텍스트 사용
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.pixel_values,
                    max_length=max_length,
                    num_beams=5,  # 정확도를 위해 beam search 옵션 증가
                    early_stopping=True
                )

            texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
            texts = [text.replace(" ", "") for text in texts]
            results.extend(texts)

        return results