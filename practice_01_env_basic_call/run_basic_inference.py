import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def gpu_status() -> dict:
    """返回当前 CUDA/GPU 状态，方便写入推理结果用于排查环境问题。"""

    if not torch.cuda.is_available():
        return {"cuda_available": False, "gpu_count": 0}
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        smi = f"nvidia-smi unavailable: {exc}"
    return {"cuda_available": True, "gpu_count": torch.cuda.device_count(), "nvidia_smi": smi}


def load_config(path: str) -> dict:
    """读取 YAML 配置文件。"""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按配置加载本地模型和处理器；local_files_only 避免服务器离线时误触发下载。
    processor = AutoProcessor.from_pretrained(cfg["model_path"], local_files_only=True, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_path"],
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=cfg.get("device", "auto"),
    )

    # 构造 Qwen-VL 聊天模板：同一轮 user 消息中同时包含图片和文本问题。
    image = Image.open(cfg["image_path"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": cfg["prompt"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=int(cfg.get("max_new_tokens", 256)))

    # generated 中前半段是输入 token，切掉后只保留模型新生成的回答。
    answer = processor.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

    result = {"gpu": gpu_status(), "image_path": cfg["image_path"], "prompt": cfg["prompt"], "answer": answer}
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
