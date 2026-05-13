#!/usr/bin/env python
"""评估 Qwen3-VL LoRA adapter 在 val/test JSONL 上的检测效果。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from vlm_lora_utils import (
    extract_answer,
    generate_text,
    load_config,
    load_model_and_processor,
    match_detections,
    normalize_messages_images,
    parse_detection_json,
    read_jsonl,
    resolve_path,
    strip_assistant_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL LoRA on val/test JSONL.")
    parser.add_argument("--config", default="config.yaml", help="config.yaml 配置文件路径")
    parser.add_argument("--split", choices=("val", "test"), default=None, help="覆盖 config.evaluation.split")
    parser.add_argument("--jsonl-path", default=None, help="直接指定评估 JSONL")
    parser.add_argument("--max-samples", type=int, default=None, help="覆盖 config.evaluation.max_samples")
    return parser.parse_args()


def metrics_from_counts(counts: Counter[str]) -> dict[str, float | int]:
    """由 TP/FP/FN 计算 precision、recall、F1。"""

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def jsonl_for_split(cfg: dict, config_dir: Path, split: str, override: str | None) -> Path:
    """按 split 选择 val/test JSONL。"""

    if override:
        path = resolve_path(override, config_dir)
    elif split == "test":
        path = resolve_path(cfg.get("vlm_test_jsonl"), config_dir)
    else:
        path = resolve_path(cfg.get("vlm_val_jsonl"), config_dir)
    if path is None:
        raise ValueError(f"Please configure vlm_{split}_jsonl or pass --jsonl-path.")
    return path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)
    eval_cfg = cfg.get("evaluation", {})

    split = args.split or eval_cfg.get("split", "val")
    jsonl_path = jsonl_for_split(cfg, config_dir, split, args.jsonl_path)
    data_root = resolve_path(cfg.get("vlm_data_root"), config_dir) or jsonl_path.parent
    output_path = resolve_path(eval_cfg.get("output_path", "outputs/vlm_lora/eval_predictions.jsonl"), config_dir)
    metrics_path = resolve_path(eval_cfg.get("metrics_path", "outputs/vlm_lora/eval_metrics.json"), config_dir)
    assert output_path is not None and metrics_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(jsonl_path)
    max_samples = args.max_samples if args.max_samples is not None else int(eval_cfg.get("max_samples", 0))
    if max_samples > 0:
        records = records[:max_samples]

    model, processor = load_model_and_processor(cfg, config_dir)
    iou_threshold = float(eval_cfg.get("iou_threshold", 0.5))
    counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {
        "head": Counter(),
        "helmet": Counter(),
    }

    with output_path.open("w", encoding="utf-8") as writer:
        for idx, row in enumerate(tqdm(records, desc=f"Evaluating {split}", unit="sample")):
            full_messages = normalize_messages_images(row["messages"], data_root)
            user_messages = strip_assistant_messages(full_messages)
            target_text = extract_answer(full_messages)
            target_objects = parse_detection_json(target_text)
            prediction = generate_text(model, processor, user_messages, eval_cfg)
            pred_objects = parse_detection_json(prediction)

            matched = match_detections(pred_objects, target_objects, iou_threshold)
            counts.update(matched)
            for label in label_counts:
                label_matched = match_detections(
                    [item for item in pred_objects if item["label"] == label],
                    [item for item in target_objects if item["label"] == label],
                    iou_threshold,
                )
                label_counts[label].update(label_matched)

            writer.write(
                json.dumps(
                    {
                        "index": idx,
                        "prediction": prediction,
                        "parsed_prediction": pred_objects,
                        "target": target_objects,
                        "match": matched,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {
        "split": split,
        "jsonl_path": str(jsonl_path),
        "sample_count": len(records),
        "iou_threshold": iou_threshold,
        "overall": metrics_from_counts(counts),
        "by_label": {label: metrics_from_counts(label_count) for label, label_count in label_counts.items()},
        "predictions_path": str(output_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] saved predictions to: {output_path}")
    print(f"[done] saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
