import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


# 读取 YAML 配置文件。
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 使用视觉语言模型执行 VQA。
# 输入：图片 + questions 列表；输出：vqa_results.json。
def run_vqa(cfg: dict, output_dir: Path) -> list[dict]:
    # 加载处理器和模型（兼容 Qwen-VL / InternVL 等支持 chat 模板的模型）。
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
        # 构造多模态消息：同一轮中包含图片和文本问题。
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

        # 去除输入 token，仅保留新增回答内容。
        answer = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        results.append({"question": question, "answer": answer})

    # 持久化保存，便于复盘与评估。
    (output_dir / "vqa_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Qwen-VL / InternVL 执行 VQA")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 仅使用配置中的 vlm 区块。
    cfg = load_config(args.config)
    vl_cfg = cfg["vlm"]

    output_dir = Path(vl_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_vqa(vl_cfg, output_dir)
    print(json.dumps({"vqa_count": len(results), "output": str(output_dir / 'vqa_results.json')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
