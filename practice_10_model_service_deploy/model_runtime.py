import time
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


@dataclass
class RuntimeConfig:
    model_path: str
    device_map: str = "auto"
    precision: str = "fp16"
    max_new_tokens: int = 512


class VisionLanguageRuntime:
    def __init__(self, cfg: RuntimeConfig):
        dtype = torch.float16 if cfg.precision == "fp16" and torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=cfg.device_map,
        )
        self.max_new_tokens = cfg.max_new_tokens

    def infer(self, image: Image.Image, prompt: str) -> dict:
        start = time.perf_counter()
        messages = [{"role": "user", "content": [{"type": "image", "image": image.convert("RGB")}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        answer = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        latency_ms = (time.perf_counter() - start) * 1000
        return {"answer": answer, "latency_ms": latency_ms}


def run_end_to_end(runtime: VisionLanguageRuntime, image: Image.Image, prompt: str) -> dict:
    segmentation = {"status": "placeholder", "masks": []}
    ocr = {"status": "placeholder", "texts": []}
    vqa = runtime.infer(image, prompt)
    return {"segmentation": segmentation, "ocr": ocr, "vqa": vqa}

