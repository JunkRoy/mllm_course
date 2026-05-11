"""Qwen-VL / InternVL 图片问答示例。"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def load_config(path: str) -> dict:
    """读取 YAML 配置文件。"""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_vqa(cfg: dict, output_dir: Path) -> list[dict]:
    """使用视觉语言模型对同一张图片批量回答 questions。"""

    processor = AutoProcessor.from_pretrained(
        cfg["model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    image = Image.open(cfg["image_path"]).convert("RGB")
    results = []
    for question in cfg.get("questions", []):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=cfg.get("max_new_tokens", 512))

        # 去掉输入 token，只保留模型新生成的回答。
        answer = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        results.append({"question": question, "answer": answer})

    (output_dir / "vqa_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Qwen-VL / InternVL 执行图片问答。")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vl_cfg = cfg["vlm"]

    output_dir = Path(vl_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_vqa(vl_cfg, output_dir)
    print(json.dumps({"vqa_count": len(results), "output": str(output_dir / "vqa_results.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
