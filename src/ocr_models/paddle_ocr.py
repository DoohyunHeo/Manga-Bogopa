import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

class PaddleOcrVLForManga:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = "jzhang533/PaddleOCR-VL-For-Manga"

        # Load the model with performance optimizations from the user's example
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,  # Use bfloat16 for performance
        ).to(self.device).eval()

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=True
        )

        # Set pad_token_id to avoid warning during generation
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id

        self.prompt_text = "OCR:" # Use the simple prompt from the user's example

    def ocr_batch(self, images):
        results = []
        for image in images:
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 1. Create the structured message format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt_text},
                    ],
                }
            ]

            # 2. Apply the chat template to get the prompt string
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 3. Process the text and image(s) to get inputs
            # Note: text and images are passed as lists
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
            inputs = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

            # 4. Generate token ids
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                )

            # 5. Slice and decode the generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = generated_ids[:, input_length:]
            answer = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            results.append(answer.strip())

        return results

    def __call__(self, images):
        return self.ocr_batch(images)