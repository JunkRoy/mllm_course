import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ask(model, processor, image, history: list[dict], question: str, system_prompt: str) -> str:
    prompt = f"{system_prompt}\n请结合上下文回答：{question}"
    messages = history + [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg["model_path"], local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    image = Image.open(cfg["image_path"]).convert("RGB")
    history, turns = [], []
    for question in cfg["questions"]:
        answer = ask(model, processor, image, history, question, cfg["system_prompt"])
        turns.append({"question": question, "answer": answer})
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    (output_dir / "vqa_turns.json").write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(turns, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

