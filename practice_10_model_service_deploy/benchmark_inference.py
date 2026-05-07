import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from model_runtime import RuntimeConfig, VisionLanguageRuntime


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = VisionLanguageRuntime(
        RuntimeConfig(
            model_path=cfg["model_path"],
            device_map=cfg.get("device_map", "auto"),
            precision=cfg.get("precision", "fp16"),
            max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        )
    )
    image = Image.open(cfg["sample_image_path"]).convert("RGB")
    rows = [runtime.infer(image, "请描述图片并输出 JSON。") for _ in range(int(cfg["benchmark_rounds"]))]
    report = {"rounds": rows, "avg_latency_ms": sum(r["latency_ms"] for r in rows) / max(1, len(rows))}
    (output_dir / "benchmark_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
