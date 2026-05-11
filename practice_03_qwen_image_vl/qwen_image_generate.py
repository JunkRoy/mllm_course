"""Qwen-Image 图像生成和可选图片编辑示例。"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from diffusers import DiffusionPipeline
from PIL import Image


def load_config(path: str) -> dict:
    """读取 YAML 配置文件。"""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline(model_path: str) -> DiffusionPipeline:
    """按配置加载 Qwen-Image 推理管线。"""

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


def generate_images(cfg: dict, output_dir: Path, pipe: DiffusionPipeline) -> list[str]:
    """根据 generation_prompts 批量文生图，并返回保存路径列表。"""

    saved_paths = []
    for idx, prompt in enumerate(cfg.get("generation_prompts", [])):
        image = pipe(prompt=prompt).images[0]
        path = output_dir / f"generated_{idx:03d}.png"
        image.save(path)
        saved_paths.append(str(path))
    return saved_paths


def edit_image(cfg: dict, output_dir: Path, pipe: DiffusionPipeline) -> str | None:
    """执行可选图片编辑；未配置 edit_image_path 时直接跳过。"""

    edit_image_path = cfg.get("edit_image_path")
    if not edit_image_path:
        return None

    source_image = Image.open(edit_image_path).convert("RGB")
    prompt = cfg.get("edit_prompt", "保持主体结构不变，增强画面清晰度和工业质感。")
    call_args = {"prompt": prompt}

    # 不同 diffusion pipeline 的调用签名不同，这里只在支持 image 参数时传入原图。
    if "image" in pipe.__call__.__code__.co_varnames:
        call_args["image"] = source_image

    result = pipe(**call_args).images[0]
    path = output_dir / "edited_image.png"
    result.save(path)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Qwen-Image 进行图像生成与编辑。")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_cfg = cfg["qwen_image"]

    output_dir = Path(image_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(image_cfg["model_path"])
    generated_paths = generate_images(image_cfg, output_dir, pipe)
    edited_path = edit_image(image_cfg, output_dir, pipe)

    summary = {
        "generated_count": len(generated_paths),
        "generated_paths": generated_paths,
        "edited_path": edited_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
