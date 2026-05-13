"""本地 Qwen-Image 图像生成与可选图片编辑示例。"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from diffusers import DiffusionPipeline
from PIL import Image


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件，空文件时返回空字典。"""

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | None, base_dir: Path) -> Path | None:
    """把配置里的相对路径解析为相对 config.yaml 所在目录的绝对路径。"""

    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def get_local_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """读取本地推理配置，同时兼容旧版 qwen_image 扁平配置。"""

    image_cfg = cfg.get("qwen_image", {})
    return image_cfg.get("local", image_cfg)


def build_pipeline(model_path: Path, dtype_name: str = "auto") -> DiffusionPipeline:
    """加载本地 Qwen-Image diffusion pipeline。"""

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # auto 模式下 GPU 使用 float16，CPU 使用 float32，减少手动配置成本。
    if dtype_name == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = getattr(torch, dtype_name)

    pipe = DiffusionPipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    # 有 CUDA 时把 pipeline 移到显卡，避免后续每次调用时隐式搬运。
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


def supported_call_args(pipe: DiffusionPipeline, call_args: dict[str, Any]) -> dict[str, Any]:
    """过滤当前 pipeline 不支持的参数，兼容不同版本的 diffusers pipeline。"""

    signature = inspect.signature(pipe.__call__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return call_args
    return {key: value for key, value in call_args.items() if key in signature.parameters}


def common_inference_args(cfg: dict[str, Any]) -> dict[str, Any]:
    """从配置中收集通用推理参数。"""

    inference_cfg = cfg.get("inference", {})
    allowed = {
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "true_cfg_scale",
        "negative_prompt",
        "max_sequence_length",
    }
    return {key: value for key, value in inference_cfg.items() if key in allowed and value is not None}


def generator_from_seed(seed: int | None) -> torch.Generator | None:
    """按配置创建随机数生成器，便于复现实验结果。"""

    if seed is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


def run_pipeline(pipe: DiffusionPipeline, call_args: dict[str, Any]) -> Image.Image:
    """在推理模式下执行 pipeline，并返回第一张图片。"""

    call_args = supported_call_args(pipe, call_args)
    with torch.inference_mode():
        return pipe(**call_args).images[0]


def generate_images(cfg: dict[str, Any], output_dir: Path, pipe: DiffusionPipeline) -> list[str]:
    """根据 generation_prompts 批量文生图，并返回保存路径。"""

    prompts = cfg.get("generation_prompts") or []
    inference_args = common_inference_args(cfg)
    generator = generator_from_seed(cfg.get("seed"))

    saved_paths: list[str] = []
    for idx, prompt in enumerate(prompts):
        # 每条 prompt 复用同一组推理参数，输出文件名用序号区分。
        call_args = {"prompt": prompt, **inference_args}
        if generator is not None:
            call_args["generator"] = generator

        image = run_pipeline(pipe, call_args)
        path = output_dir / f"generated_{idx:03d}.png"
        image.save(path)
        saved_paths.append(str(path))
    return saved_paths


def edit_image(
    cfg: dict[str, Any],
    output_dir: Path,
    pipe: DiffusionPipeline,
    config_dir: Path,
) -> str | None:
    """配置 edit_image_path 时执行单张图片编辑。"""

    edit_image_path = resolve_path(cfg.get("edit_image_path"), config_dir)
    if not edit_image_path:
        return None
    if not edit_image_path.exists():
        raise FileNotFoundError(f"Edit input image does not exist: {edit_image_path}")

    source_image = Image.open(edit_image_path).convert("RGB")
    prompt = cfg.get("edit_prompt", "保持主体结构不变，增强画面清晰度和工业质感。")
    # Qwen-Image-Edit 类 pipeline 通常需要 prompt + image；不支持的参数会在 run_pipeline 中过滤。
    call_args = {
        "prompt": prompt,
        "image": source_image,
        **common_inference_args(cfg),
    }
    generator = generator_from_seed(cfg.get("seed"))
    if generator is not None:
        call_args["generator"] = generator

    image = run_pipeline(pipe, call_args)
    path = output_dir / "edited_image.png"
    image.save(path)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地运行 Qwen-Image 图像生成或编辑。")
    parser.add_argument("--config", default="config.yaml", help="config.yaml 配置文件路径")
    parser.add_argument(
        "--task",
        choices=["all", "generate", "edit"],
        default="all",
        help="选择运行文生图、图片编辑或两者都运行。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)
    image_cfg = get_local_config(cfg)

    # 输出目录和模型目录都按配置文件位置解析，避免受当前工作目录影响。
    output_dir = resolve_path(image_cfg.get("output_dir", "outputs"), config_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(image_cfg["model_path"], config_dir)
    assert model_path is not None
    pipe = build_pipeline(model_path, image_cfg.get("torch_dtype", "auto"))

    generated_paths: list[str] = []
    edited_path: str | None = None
    # 按 --task 决定执行哪部分任务，方便课堂上单独调试生成或编辑。
    if args.task in {"all", "generate"}:
        generated_paths = generate_images(image_cfg, output_dir, pipe)
    if args.task in {"all", "edit"}:
        edited_path = edit_image(image_cfg, output_dir, pipe, config_dir)

    summary = {
        "backend": "local",
        "task": args.task,
        "generated_count": len(generated_paths),
        "generated_paths": generated_paths,
        "edited_path": edited_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
