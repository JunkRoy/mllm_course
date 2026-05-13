#!/usr/bin/env python
"""加载训练好的 Qwen3-VL LoRA adapter 进行单图或 JSONL 批量推理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_lora_utils import (
    extract_answer,
    generate_text,
    load_config,
    load_image,
    load_model_and_processor,
    messages_from_image_prompt,
    normalize_messages_images,
    parse_detection_json,
    read_jsonl,
    resolve_path,
    strip_assistant_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL LoRA inference on one image or JSONL.")
    parser.add_argument("--config", default="config.yaml", help="config.yaml 配置文件路径")
    parser.add_argument("--image", default=None, help="覆盖 config.inference.input_image")
    parser.add_argument("--prompt", default=None, help="覆盖 config.inference.prompt")
    parser.add_argument("--output-path", default=None, help="覆盖 config.inference.output_path")
    parser.add_argument("--jsonl-path", default=None, help="使用 Qwen-VL messages JSONL 批量预测")
    parser.add_argument("--max-samples", type=int, default=None, help="批量预测最多处理多少条，0 表示全部")
    return parser.parse_args()


def run_single_image(args: argparse.Namespace, cfg: dict, config_dir: Path) -> None:
    """按 config.inference 或命令行参数执行单图推理。"""

    inference_cfg = cfg.get("inference", {})
    image_value = args.image or inference_cfg["input_image"]
    prompt = args.prompt or inference_cfg["prompt"]
    output_path = resolve_path(args.output_path or inference_cfg.get("output_path"), config_dir)
    if output_path is None:
        raise ValueError("Please configure inference.output_path or pass --output-path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(cfg, config_dir)
    data_root = resolve_path(cfg.get("vlm_data_root"), config_dir)
    image = load_image(image_value, data_root)
    messages = messages_from_image_prompt(image, prompt)
    prediction = generate_text(model, processor, messages, inference_cfg)

    result = {
        "image": image_value,
        "prompt": prompt,
        "prediction": prediction,
        "parsed_prediction": parse_detection_json(prediction),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[done] saved result to: {output_path}")


def run_jsonl(args: argparse.Namespace, cfg: dict, config_dir: Path) -> None:
    """直接读取 qwen_vl_val/test JSONL 批量预测。"""

    inference_cfg = cfg.get("inference", {})
    jsonl_path = resolve_path(args.jsonl_path or inference_cfg.get("jsonl_path"), config_dir)
    output_path = resolve_path(args.output_path or inference_cfg.get("batch_output_path"), config_dir)
    if jsonl_path is None:
        raise ValueError("Please pass --jsonl-path or configure inference.jsonl_path.")
    if output_path is None:
        raise ValueError("Please pass --output-path or configure inference.batch_output_path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(jsonl_path)
    max_samples = args.max_samples if args.max_samples is not None else int(inference_cfg.get("max_samples", 0))
    if max_samples and max_samples > 0:
        records = records[:max_samples]

    model, processor = load_model_and_processor(cfg, config_dir)
    data_root = resolve_path(cfg.get("vlm_data_root"), config_dir) or jsonl_path.parent

    with output_path.open("w", encoding="utf-8") as writer:
        for idx, row in enumerate(records):
            full_messages = normalize_messages_images(row["messages"], data_root)
            user_messages = strip_assistant_messages(full_messages)
            prediction = generate_text(model, processor, user_messages, inference_cfg)
            result = {
                "index": idx,
                "prediction": prediction,
                "parsed_prediction": parse_detection_json(prediction),
                "target_text": extract_answer(full_messages),
            }
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            if (idx + 1) % 10 == 0:
                print(f"[infer] processed {idx + 1}/{len(records)}")

    print(f"[done] saved batch predictions to: {output_path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)

    if args.jsonl_path or cfg.get("inference", {}).get("jsonl_path"):
        run_jsonl(args, cfg, config_dir)
    else:
        run_single_image(args, cfg, config_dir)


if __name__ == "__main__":
    main()
