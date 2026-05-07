import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> Dataset:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"]) / "vlm_lora"

    processor = AutoProcessor.from_pretrained(cfg["vlm_model_path"], local_files_only=True, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["vlm_model_path"],
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_cfg)
    train_ds = load_jsonl(cfg["vlm_train_jsonl"])

    def collate(batch):
        prompts = [f"<image>\n{row['instruction']}\n{row['answer']}" for row in batch]
        return processor(text=prompts, padding=True, truncation=True, return_tensors="pt")

    args_train = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(cfg["epochs"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        save_steps=20,
        logging_steps=1,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args_train, train_dataset=train_ds, data_collator=collate)
    trainer.train()
    model.save_pretrained(output_dir / "adapter")


if __name__ == "__main__":
    main()

