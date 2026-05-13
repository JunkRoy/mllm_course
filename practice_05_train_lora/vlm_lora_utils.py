"""Qwen3-VL LoRA 推理和评估共用工具。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | None, base_dir: Path) -> Path | None:
    """把相对路径解析为相对 config.yaml 所在目录的绝对路径。"""

    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def choose_dtype(dtype_name: str) -> torch.dtype:
    """选择模型加载精度。"""

    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, dtype_name)


def processor_load_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """构造 processor 加载参数，用 min/max pixels 控制视觉 token 数。"""

    kwargs: dict[str, Any] = {}
    if model_cfg.get("min_pixels") is not None:
        kwargs["min_pixels"] = int(model_cfg["min_pixels"])
    if model_cfg.get("max_pixels") is not None:
        kwargs["max_pixels"] = int(model_cfg["max_pixels"])
    return kwargs


def load_image(image: str | Path, data_root: Path | None = None) -> Image.Image:
    """读取本地图片为 RGB。"""

    image_text = str(image)
    image_path = Path(image_text).expanduser()
    if not image_path.is_absolute() and data_root is not None:
        image_path = data_root / image_path
    with Image.open(image_path) as pil_image:
        return pil_image.convert("RGB")


def load_model_and_processor(
    cfg: dict[str, Any],
    config_dir: Path,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """加载 Qwen3-VL 基座模型、LoRA adapter 和 processor。"""

    model_cfg = cfg.get("model", {})
    model_path = resolve_path(cfg["vlm_model_path"], config_dir)
    adapter_path = resolve_path(cfg.get("lora_adapter_path"), config_dir)
    if model_path is None or adapter_path is None:
        raise ValueError("vlm_model_path and lora_adapter_path must be configured.")

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
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, processor


def messages_from_image_prompt(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    """用单张图片和提示词构造 Qwen3-VL messages。"""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def strip_assistant_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """评估时只保留 user 消息，避免把标准答案送进模型。"""

    return [message for message in messages if message.get("role") != "assistant"]


def extract_answer(messages: list[dict[str, Any]]) -> str:
    """从 messages 中取 assistant 文本答案。"""

    for message in messages:
        if message.get("role") != "assistant":
            continue
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                return str(item.get("text", ""))
    return ""


def normalize_messages_images(messages: list[dict[str, Any]], data_root: Path | None) -> list[dict[str, Any]]:
    """把 messages 中的本地图片路径替换为 PIL 图片。"""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = []
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "image":
                content.append({"type": "image", "image": load_image(str(item["image"]), data_root)})
            else:
                content.append(item)
        normalized.append({"role": message.get("role"), "content": content})
    return normalized


def split_messages_for_processor(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    """模板里保留图片占位符，真实图片单独传给 processor。"""

    template_messages: list[dict[str, Any]] = []
    images: list[Any] = []
    for message in messages:
        template_content: list[dict[str, Any]] = []
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "image":
                images.append(item["image"])
                template_content.append({"type": "image"})
            else:
                template_content.append(item)
        template_messages.append({"role": message.get("role"), "content": template_content})
    return template_messages, images


def generate_text(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    generation_cfg: dict[str, Any],
) -> str:
    """对一条 messages 执行生成，并返回新增文本。"""

    template_messages, images = split_messages_for_processor(messages)
    text = processor.apply_chat_template(template_messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 512)),
        "do_sample": bool(generation_cfg.get("do_sample", False)),
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs["temperature"] = float(generation_cfg.get("temperature", 0.7))

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_detection_json(text: str) -> list[dict[str, Any]]:
    """从模型输出文本中解析检测 JSON 数组。"""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(parsed, list):
        return []

    objects: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        bbox = item.get("bbox")
        if label not in {"head", "helmet"} or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            coords = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            continue
        xmin, ymin, xmax, ymax = coords
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append({"label": label, "bbox": coords})
    return objects


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    """计算两个 bbox 的 IoU。"""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def match_detections(preds: list[dict[str, Any]], targets: list[dict[str, Any]], iou_threshold: float) -> dict[str, int]:
    """按 label 和 IoU 贪心匹配预测框与标注框。"""

    used_targets: set[int] = set()
    tp = 0
    for pred in preds:
        best_idx = -1
        best_iou = 0.0
        for idx, target in enumerate(targets):
            if idx in used_targets or pred["label"] != target["label"]:
                continue
            iou = bbox_iou(pred["bbox"], target["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            used_targets.add(best_idx)
    fp = len(preds) - tp
    fn = len(targets) - tp
    return {"tp": tp, "fp": fp, "fn": fn}
