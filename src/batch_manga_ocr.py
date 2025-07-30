import torch
from manga_ocr import MangaOcr
from transformers import AutoTokenizer


class BatchMangaOcr(MangaOcr):
    def __init__(self, model_name="kha-white/manga-ocr-base", device=None):
        super().__init__(model_name)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 직접 tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, image_list):
        """
        클래스 인스턴스를 함수처럼 호출할 때 ocr_batch를 실행하도록 합니다.
        (기존 ocr_model(crops) 호출과 호환성을 위함)
        """
        if isinstance(image_list, list):
            return self.ocr_batch(image_list)
        # 단일 이미지가 들어오면 기존 방식대로 처리
        return super().__call__(image_list)

    def ocr_batch(self, image_list, max_length=128, batch_size=2):
        results = []
        for i in range(0, len(image_list), batch_size):
            batch_images = image_list[i:i + batch_size]
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(inputs.pixel_values, max_length=max_length)

            texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            texts = [text.replace(" ", "") for text in texts]
            results.extend(texts)

        return results