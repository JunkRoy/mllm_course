"""使用 LoRA 微调 Qwen3-VL-8B-Instruct 的训练入口。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

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


def load_jsonl(path: Path) -> Dataset:
    """读取 JSONL 训练集，每行是一条 image + instruction + answer 样本。"""

    return JsonlDataset(read_jsonl_records(path))


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 原始记录列表，数据检查和 Dataset 构造共用。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class JsonlDataset(Dataset):
    """保持 JSONL 原始嵌套结构的 PyTorch Dataset。"""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def choose_dtype(dtype_name: str) -> torch.dtype | str:
    """按配置选择模型加载精度。"""

    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, dtype_name)


def load_local_image(path: Path) -> Any:
    """用 PIL 读取本地图片并转成 RGB，兼容 BMP 等 torchvision 不直接支持的格式。"""

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("缺少依赖：Pillow。请先运行 pip install Pillow") from exc

    with Image.open(path) as image:
        return image.convert("RGB")


def row_images(row: dict[str, Any], data_root: Path | None, load_images: bool) -> list[Any]:
    """兼容 image、image_path、images 三种常见图片字段。"""

    value = row.get("images", row.get("image", row.get("image_path")))
    if value is None:
        return []
    images = value if isinstance(value, list) else [value]

    resolved: list[str] = []
    for image in images:
        image_text = str(image)
        if image_text.startswith(("http://", "https://", "data:")):
            resolved.append(image_text)
            continue

        image_path = Path(image_text).expanduser()
        if not image_path.is_absolute() and data_root is not None:
            image_path = data_root / image_path
        image_path = image_path.resolve()
        resolved.append(load_local_image(image_path) if load_images else str(image_path))
    return resolved


def row_text(row: dict[str, Any], keys: tuple[str, ...], field_name: str) -> str:
    """从一组候选字段中读取文本，缺失时给出清晰错误。"""

    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError(f"训练样本缺少 {field_name} 字段，可用字段：{', '.join(keys)}")


def resolve_image_value(value: str, data_root: Path | None, load_images: bool) -> Any:
    """把 messages 中的图片路径解析成绝对路径，URL 和 data URL 保持原样。"""

    if value.startswith(("http://", "https://", "data:")):
        return value

    image_path = Path(value).expanduser()
    if not image_path.is_absolute() and data_root is not None:
        image_path = data_root / image_path
    image_path = image_path.resolve()
    return load_local_image(image_path) if load_images else str(image_path)


def normalize_content(content: Any, data_root: Path | None, load_images: bool) -> list[dict[str, Any]]:
    """规范化一条 message 的 content，并修正其中的图片路径。"""

    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    normalized: list[dict[str, Any]] = []
    for item in content or []:
        if not isinstance(item, dict):
            normalized.append({"type": "text", "text": str(item)})
            continue

        new_item = dict(item)
        if new_item.get("type") == "image" and new_item.get("image"):
            new_item["image"] = resolve_image_value(str(new_item["image"]), data_root, load_images)
        normalized.append(new_item)
    return normalized


def normalize_messages(
    messages: list[dict[str, Any]],
    data_root: Path | None,
    include_answer: bool,
    load_images: bool,
) -> list[dict[str, Any]]:
    """规范化现成的 Qwen-VL messages 数据；可按需去掉 assistant 答案。"""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if not include_answer and role == "assistant":
            continue
        normalized.append(
            {
                "role": role,
                "content": normalize_content(message.get("content"), data_root, load_images),
            }
        )
    return normalized


def build_flat_messages(
    row: dict[str, Any],
    data_root: Path | None,
    include_answer: bool,
    load_images: bool,
) -> list[dict[str, Any]]:
    """把 image/instruction/answer 扁平样本转成 Qwen3-VL messages。"""

    instruction = row_text(row, ("instruction", "question", "prompt"), "instruction/question/prompt")
    answer = row_text(row, ("answer", "output", "response"), "answer/output/response")

    content: list[dict[str, str]] = []
    for image in row_images(row, data_root, load_images):
        # Qwen3-VL 的 processor 支持 type=image 的消息；本地路径和 URL 都可放在 image 字段。
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": instruction})

    messages = [{"role": "user", "content": content}]
    if include_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def build_messages(
    row: dict[str, Any],
    data_root: Path | None,
    include_answer: bool,
    load_images: bool = True,
) -> list[dict[str, Any]]:
    """构造 Qwen3-VL chat template 需要的多模态消息。"""

    # practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py 输出的就是 messages 格式。
    if row.get("messages"):
        return normalize_messages(row["messages"], data_root, include_answer, load_images)
    return build_flat_messages(row, data_root, include_answer, load_images)


def split_messages_for_processor(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    """拆分 messages：模板只保留 image 占位符，真实图片单独交给 processor。"""

    template_messages: list[dict[str, Any]] = []
    images: list[Any] = []

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            template_messages.append(message)
            continue

        template_content: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                if item.get("image") is None:
                    # 没有真实图片值时不要生成 image 占位符，否则模板和图片数量会错位。
                    continue
                images.append(item["image"])
                # 模板里只需要知道这里有一张图，图片对象不要再塞进模板。
                template_content.append({"type": "image"})
            else:
                template_content.append(item)
        template_messages.append({"role": message.get("role"), "content": template_content})

    return template_messages, images


def batch_encode_messages(
    processor: AutoProcessor,
    messages: list[list[dict[str, Any]]],
    *,
    add_generation_prompt: bool,
) -> dict[str, torch.Tensor]:
    """先渲染 chat template，再单独传入图片做编码。"""

    texts: list[str] = []
    images: list[Any] = []
    for sample_messages in messages:
        template_messages, sample_images = split_messages_for_processor(sample_messages)
        texts.append(
            processor.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )
        images.extend(sample_images)

    processor_kwargs: dict[str, Any] = {
        "text": texts,
        "padding": True,
        "return_tensors": "pt",
    }
    if images:
        processor_kwargs["images"] = images
    return processor(**processor_kwargs)


def count_rendered_image_tokens(text: str) -> int:
    """粗略统计渲染后模板中的图片占位符数量。"""

    tokens = ("<|image_pad|>", "<image>")
    return sum(text.count(token) for token in tokens)


def describe_record_images(row: dict[str, Any]) -> list[str]:
    """返回样本中的原始图片引用，便于错误定位。"""

    return image_values_from_record(row)


class Qwen3VLCollator:
    """把 JSONL 样本转换成 Qwen3-VL 训练 batch。"""

    def __init__(self, processor: AutoProcessor, data_root: Path | None, max_length: int) -> None:
        self.processor = processor
        self.data_root = data_root
        self.max_length = max_length
        self._warned_long_sequence = False

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # full_messages 包含 assistant 答案，用作模型输入和 label 来源。
        full_messages = [build_messages(row, self.data_root, include_answer=True) for row in batch]
        # prompt_messages 不包含答案，只用来计算需要 mask 的用户提示长度。
        prompt_messages = [build_messages(row, self.data_root, include_answer=False) for row in batch]

        model_inputs = batch_encode_messages(
            self.processor,
            full_messages,
            add_generation_prompt=False,
        )
        prompt_inputs = batch_encode_messages(
            self.processor,
            prompt_messages,
            add_generation_prompt=True,
        )

        labels = model_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        # 只训练 assistant 的回答；图片 token、用户指令和 system/chat 模板都不计入 loss。
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_length in enumerate(prompt_lengths):
            labels[row_idx, : int(prompt_length)] = -100

        if self.max_length > 0 and labels.shape[1] > self.max_length and not self._warned_long_sequence:
            # 多模态序列不能像纯文本一样直接截断，否则 image token 和 image_grid_thw 会不匹配。
            print(
                f"[warn] encoded sequence length {labels.shape[1]} exceeds training.max_length={self.max_length}; "
                "not truncating multimodal inputs. Reduce model.max_pixels if this causes OOM."
            )
            self._warned_long_sequence = True

        model_inputs["labels"] = labels
        return model_inputs


def debug_scan_collate(
    records: list[dict[str, Any]],
    collator: Qwen3VLCollator,
    processor: AutoProcessor,
    limit: int,
) -> None:
    """逐条测试 processor 编码，定位第一条无法编码的样本。"""

    total = len(records) if limit <= 0 else min(limit, len(records))
    for idx in range(total):
        row = records[idx]
        try:
            messages = build_messages(row, collator.data_root, include_answer=True)
            template_messages, sample_images = split_messages_for_processor(messages)
            rendered = processor.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            rendered_image_tokens = count_rendered_image_tokens(rendered)
            if rendered_image_tokens != len(sample_images):
                print(
                    json.dumps(
                        {
                            "bad_line": idx + 1,
                            "reason": "image token count mismatch before processor call",
                            "rendered_image_tokens": rendered_image_tokens,
                            "actual_images": len(sample_images),
                            "images": describe_record_images(row),
                            "rendered_prefix": rendered[:500],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return

            collator([row])
        except Exception as exc:  # noqa: BLE001 - 调试入口要展示真实异常。
            print(
                json.dumps(
                    {
                        "bad_line": idx + 1,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "images": describe_record_images(row),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise

        if (idx + 1) % 100 == 0:
            print(f"[debug-scan] checked {idx + 1}/{total}")

    print(f"[debug-scan] all {total} samples passed processor encoding")


def image_values_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    """从 messages 样本中抽取图片字段，供数据检查使用。"""

    images: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and item.get("image"):
                    images.append(str(item["image"]))
    return images


def image_values_from_record(row: dict[str, Any]) -> list[str]:
    """从一条训练样本中抽取原始图片引用。"""

    if row.get("messages"):
        return image_values_from_messages(row["messages"])
    value = row.get("images", row.get("image", row.get("image_path")))
    if value is None:
        return []
    return [str(item) for item in (value if isinstance(value, list) else [value])]


def resolve_image_path_for_check(value: str, data_root: Path | None) -> Path | None:
    """数据检查时把本地图片引用解析成路径；URL 和 data URL 返回 None。"""

    if value.startswith(("http://", "https://", "data:")):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and data_root is not None:
        path = data_root / path
    return path.resolve()


def check_data(records: list[dict[str, Any]], data_root: Path | None, max_bad_examples: int = 10) -> None:
    """检查 JSONL 格式、图片路径和图片可读性。"""

    format_counter: Counter[str] = Counter()
    suffix_counter: Counter[str] = Counter()
    missing_images: list[str] = []
    unreadable_images: list[str] = []
    no_image_count = 0
    no_answer_count = 0

    for row_idx, row in enumerate(records):
        format_counter["messages" if row.get("messages") else "flat"] += 1
        images = image_values_from_record(row)
        if not images:
            no_image_count += 1

        if row.get("messages"):
            has_answer = any(message.get("role") == "assistant" for message in row["messages"])
        else:
            has_answer = any(row.get(key) is not None for key in ("answer", "output", "response"))
        if not has_answer:
            no_answer_count += 1

        for image in images:
            path = resolve_image_path_for_check(image, data_root)
            if path is None:
                suffix_counter["<url_or_data>"] += 1
                continue

            suffix_counter[path.suffix.lower() or "<no_suffix>"] += 1
            if not path.exists():
                missing_images.append(f"line {row_idx + 1}: {path}")
                continue
            try:
                load_local_image(path)
            except Exception as exc:  # noqa: BLE001 - 这里要把坏样本完整报出来。
                unreadable_images.append(f"line {row_idx + 1}: {path} ({exc})")

    print("[check] samples:", len(records))
    print("[check] formats:", dict(format_counter))
    print("[check] image suffixes:", dict(suffix_counter))
    print("[check] no image samples:", no_image_count)
    print("[check] no answer samples:", no_answer_count)
    print("[check] missing images:", len(missing_images))
    for item in missing_images[:max_bad_examples]:
        print("  -", item)
    print("[check] unreadable images:", len(unreadable_images))
    for item in unreadable_images[:max_bad_examples]:
        print("  -", item)


def build_training_args(cfg: dict[str, Any], output_dir: Path) -> TrainingArguments:
    """从配置生成 Hugging Face Trainer 参数。"""

    train_cfg = cfg.get("training", {})
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("epochs", cfg.get("epochs", 3))),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", cfg.get("batch_size", 1))),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", cfg.get("learning_rate", 1e-4))),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        save_steps=int(train_cfg.get("save_steps", 100)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 4)),
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        ddp_find_unused_parameters=False,
    )


def processor_load_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """构造 processor 加载参数，用 min/max pixels 控制视觉 token 数。"""

    kwargs: dict[str, Any] = {}
    if model_cfg.get("min_pixels") is not None:
        kwargs["min_pixels"] = int(model_cfg["min_pixels"])
    if model_cfg.get("max_pixels") is not None:
        kwargs["max_pixels"] = int(model_cfg["max_pixels"])
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 LoRA 微调 Qwen3-VL-8B-Instruct。")
    parser.add_argument("--config", default="config.yaml", help="config.yaml 配置文件路径")
    parser.add_argument("--check-data", action="store_true", help="只检查训练 JSONL 和图片，不加载模型。")
    parser.add_argument("--debug-collate", type=int, default=0, help="只测试前 N 条样本的 processor 编码，不启动训练。")
    parser.add_argument("--debug-scan", type=int, default=0, help="逐条扫描前 N 条样本；0 表示不扫描，-1 表示扫描全部。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)

    model_path = resolve_path(cfg["vlm_model_path"], config_dir)
    train_jsonl = resolve_path(cfg["vlm_train_jsonl"], config_dir)
    data_root = resolve_path(cfg.get("vlm_data_root"), config_dir)
    output_dir = resolve_path(cfg.get("output_dir", "outputs"), config_dir)
    assert model_path is not None and train_jsonl is not None and output_dir is not None
    output_dir = output_dir / "vlm_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl_records(train_jsonl)
    if args.check_data:
        check_data(records, data_root or train_jsonl.parent)
        return

    model_cfg = cfg.get("model", {})
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=bool(cfg.get("local_files_only", True)),
        trust_remote_code=True,
        **processor_load_kwargs(model_cfg),
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        local_files_only=bool(cfg.get("local_files_only", True)),
        trust_remote_code=True,
        dtype=choose_dtype(model_cfg.get("dtype", "bfloat16")),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    if bool(cfg.get("training", {}).get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"].get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 不使用 datasets.Dataset.from_list，避免 Arrow 改写嵌套 messages/content 结构。
    train_ds = JsonlDataset(records)
    collator = Qwen3VLCollator(
        processor=processor,
        data_root=data_root or train_jsonl.parent,
        max_length=int(cfg.get("training", {}).get("max_length", 8192)),
    )
    if args.debug_collate > 0:
        batch = [records[idx] for idx in range(min(args.debug_collate, len(records)))]
        encoded = collator(batch)
        summary = {
            key: list(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            for key, value in encoded.items()
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.debug_scan != 0:
        debug_scan_collate(records, collator, processor, args.debug_scan)
        return

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg, output_dir),
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(output_dir / "adapter")
    # 多卡训练时只让主进程保存 processor，避免多个 rank 同时写同一批文件。
    if trainer.is_world_process_zero():
        processor.save_pretrained(output_dir / "adapter")


if __name__ == "__main__":
    main()
