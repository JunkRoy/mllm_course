import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_trainable(model, mode: str) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False
    keywords = {
        "projector_only": ["projector", "mm_projector", "multi_modal_projector"],
        "projector_llm": ["projector", "mm_projector", "language_model", "model.layers"],
        "vision_partial": ["vision_tower.encoder.layers.20", "vision_model.encoder.layers.20", "projector"],
    }.get(mode, [])
    for name, p in model.named_parameters():
        if any(key in name for key in keywords):
            p.requires_grad = True


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_ratio": trainable / max(1, total)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    set_trainable(model, cfg["tuning_mode"])
    report = count_parameters(model)
    report["tuning_mode"] = cfg["tuning_mode"]
    (output_dir / "parameter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    model.save_pretrained(output_dir / "checkpoint")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

