import argparse
import json
from pathlib import Path

import torch
import yaml
from diffusers import DiffusionPipeline
from PIL import Image


# 读取 YAML 配置文件，并返回字典结构。
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 按配置加载 Qwen-Image 推理管线。
# - local_files_only=True: 仅从本地加载模型，避免联网下载。
# - torch_dtype: 有 GPU 时默认 FP16 以减少显存占用。
def build_pipeline(model_path: str) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


# 根据 generation_prompts 批量文生图。
# 每条 prompt 生成一张图片，并按 generated_XXX.png 命名保存。
def generate_images(cfg: dict, output_dir: Path, pipe: DiffusionPipeline) -> list[str]:
    saved_paths = []
    for idx, prompt in enumerate(cfg.get("generation_prompts", [])):
        image = pipe(prompt=prompt).images[0]
        path = output_dir / f"generated_{idx:03d}.png"
        image.save(path)
        saved_paths.append(str(path))
    return saved_paths


# 可选图像编辑流程：当 edit_image_path 为空时直接跳过。
# 部分模型的 pipe.__call__ 支持 image 参数，部分不支持，故做兼容判断。
def edit_image(cfg: dict, output_dir: Path, pipe: DiffusionPipeline) -> str | None:
    edit_image_path = cfg.get("edit_image_path")
    if not edit_image_path:
        return None

    source_image = Image.open(edit_image_path).convert("RGB")
    prompt = cfg.get("edit_prompt", "保持主体结构不变，增强画面清晰度和工业质感。")
    call_args = {"prompt": prompt}
    if "image" in pipe.__call__.__code__.co_varnames:
        call_args["image"] = source_image

    result = pipe(**call_args).images[0]
    path = output_dir / "edited_image.png"
    result.save(path)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Qwen-Image 进行图像生成与编辑")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 仅使用配置中的 qwen_image 区块。
    cfg = load_config(args.config)
    image_cfg = cfg["qwen_image"]

    # 创建输出目录（若不存在则自动创建）。
    output_dir = Path(image_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 先执行文生图，再执行可选编辑流程。
    pipe = build_pipeline(image_cfg["model_path"])
    generated_paths = generate_images(image_cfg, output_dir, pipe)
    edited_path = edit_image(image_cfg, output_dir, pipe)

    # 控制台输出本次任务摘要，方便自动化脚本读取。
    summary = {
        "generated_count": len(generated_paths),
        "generated_paths": generated_paths,
        "edited_path": edited_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
