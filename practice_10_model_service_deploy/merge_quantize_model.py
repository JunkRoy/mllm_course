import argparse
import json
from pathlib import Path

import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--merged_output", default="outputs/merged_model")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.merged_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], local_files_only=True, trust_remote_code=True, device_map="auto")
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
        model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True, trust_remote_code=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    report = {
        "base_model": cfg["model_path"],
        "adapter_path": args.adapter_path,
        "merged_output": str(output_dir),
        "quantization_note": "如需 int8/int4，请在本地 bitsandbytes、AutoGPTQ 或 AWQ 环境中加载该目录后执行量化保存。",
    }
    (output_dir / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
