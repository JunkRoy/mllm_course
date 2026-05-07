import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM, AutoProcessor


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_images(cfg: dict, output_dir: Path) -> list[str]:
    pipe = DiffusionPipeline.from_pretrained(
        cfg["image_model_path"],
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    saved = []
    for idx, prompt in enumerate(cfg.get("generation_prompts", [])):
        image = pipe(prompt=prompt).images[0]
        path = output_dir / f"generated_{idx:03d}.png"
        image.save(path)
        saved.append(str(path))
    return saved


def edit_image(cfg: dict, output_dir: Path) -> str | None:
    if not cfg.get("edit_image_path"):
        return None
    pipe = DiffusionPipeline.from_pretrained(
        cfg["image_model_path"],
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    image = Image.open(cfg["edit_image_path"]).convert("RGB")
    prompt = cfg.get("edit_prompt", "保持主体结构不变，增强画面清晰度和工业质感。")
    result = pipe(prompt=prompt, image=image).images[0] if "image" in pipe.__call__.__code__.co_varnames else pipe(prompt=prompt).images[0]
    path = output_dir / "edited_image.png"
    result.save(path)
    return str(path)


def run_vqa(cfg: dict, output_dir: Path) -> list[dict]:
    processor = AutoProcessor.from_pretrained(cfg["vlm_model_path"], local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["vlm_model_path"],
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    image = Image.open(cfg["vqa_image_path"]).convert("RGB")
    results = []
    for question in cfg.get("vqa_questions", []):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        answer = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        results.append({"question": question, "answer": answer})
    (output_dir / "vqa_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_images(cfg, output_dir)
    edited = edit_image(cfg, output_dir)
    vqa_results = run_vqa(cfg, output_dir)
    print(json.dumps({"generated": generated, "edited": edited, "vqa_count": len(vqa_results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
