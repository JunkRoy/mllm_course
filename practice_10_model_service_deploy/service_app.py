import io

import yaml
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from model_runtime import RuntimeConfig, VisionLanguageRuntime, run_end_to_end


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
runtime = VisionLanguageRuntime(
    RuntimeConfig(
        model_path=cfg["model_path"],
        device_map=cfg.get("device_map", "auto"),
        precision=cfg.get("precision", "fp16"),
        max_new_tokens=int(cfg.get("max_new_tokens", 512)),
    )
)

app = FastAPI(title="视觉大模型服务实操")


@app.post("/vqa")
async def vqa(image: UploadFile = File(...), prompt: str = Form(...)):
    content = await image.read()
    pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    return runtime.infer(pil_image, prompt)


@app.post("/pipeline")
async def pipeline(image: UploadFile = File(...), prompt: str = Form("请输出结构化 JSON 结果")):
    content = await image.read()
    pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    return run_end_to_end(runtime, pil_image, prompt)

