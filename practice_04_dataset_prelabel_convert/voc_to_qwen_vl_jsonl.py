#!/usr/bin/env python
"""把合并后的 VOC 训练集转换为 Qwen-VL 指令微调 JSONL。"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 只是进度条，缺失时不影响主流程。
    def tqdm(iterable, **_: object):
        return iterable


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TRAIN_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
# 默认约定：dataset 与 practice_04_dataset_prelabel_convert 位于同一级目录。
# DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"
DEFAULT_PROMPT = (
    "请检测图片中的安全帽和未佩戴安全帽的头部。"
    "请只输出 JSON 数组，每个元素包含 label 和 bbox，"
    "bbox 格式为 [xmin, ymin, xmax, ymax]。"
)


def text_or_empty(element: ET.Element | None) -> str:
    """安全读取 XML 节点文本。"""

    return "" if element is None or element.text is None else element.text.strip()


def int_text(element: ET.Element | None, default: int = 0) -> int:
    """安全读取 XML 节点中的整数。"""

    try:
        return int(float(text_or_empty(element)))
    except ValueError:
        return default


def find_image(output_dir: Path, filename: str, xml_stem: str) -> Path | None:
    """从 VOC 输出目录中找到 XML 对应的图片。"""

    image_dir = output_dir / "JPEGImages"
    if filename:
        candidate = image_dir / filename
        if candidate.exists():
            return candidate

    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{xml_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def import_pillow():
    """延迟导入 Pillow，用于图片读取、尺寸检查和格式规范化。"""

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("This script needs Pillow. Install it with: pip install pillow") from exc
    return Image


def read_image_size(image_path: Path) -> tuple[int, int]:
    """读取图片尺寸，同时验证图片文件可被 Pillow 打开。"""

    Image = import_pillow()
    with Image.open(image_path) as image:
        return image.size


def normalize_image_for_training(image_path: Path, image_dir: Path, sample_id: str, image_format: str) -> Path:
    """把训练图片规范化为 RGB JPEG/PNG/WEBP，避免训练阶段遇到不支持的图片格式。"""

    target_suffix = "." + image_format.lower().replace("jpeg", "jpg")
    target_path = image_dir / f"{sample_id}{target_suffix}"
    source_suffix = image_path.suffix.lower()

    if source_suffix in TRAIN_IMAGE_EXTENSIONS and image_path.resolve() == target_path.resolve():
        return image_path
    if source_suffix in TRAIN_IMAGE_EXTENSIONS and not target_path.exists():
        shutil.copy2(image_path, target_path)
        return target_path

    Image = import_pillow()
    with Image.open(image_path) as image:
        image.convert("RGB").save(target_path, format=image_format.upper(), quality=95)
    return target_path


def clip_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    """把 bbox 裁剪到图片范围内；裁剪后无效则返回 None。"""

    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(0, min(xmax, width - 1))
    ymax = max(0, min(ymax, height - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


def read_voc_sample(xml_path: Path, output_dir: Path) -> tuple[Path | None, list[dict]]:
    """读取一条 VOC 样本，返回图片路径和目标列表。"""

    root = ET.parse(xml_path).getroot()
    filename = text_or_empty(root.find("filename"))
    image_path = find_image(output_dir, filename, xml_path.stem)

    objects: list[dict] = []
    for obj in root.findall("object"):
        label = text_or_empty(obj.find("name"))
        box = obj.find("bndbox")
        if not label or box is None:
            continue
        xmin = int_text(box.find("xmin"))
        ymin = int_text(box.find("ymin"))
        xmax = int_text(box.find("xmax"))
        ymax = int_text(box.find("ymax"))
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

    return image_path, objects


def validate_objects(objects: list[dict], width: int, height: int, allowed_labels: set[str]) -> tuple[list[dict], Counter]:
    """检查类别和 bbox，并把 bbox 裁剪到图片尺寸内。"""

    stats: Counter[str] = Counter()
    valid_objects: list[dict] = []
    for item in objects:
        label = item["label"]
        if label not in allowed_labels:
            stats["invalid_label"] += 1
            continue
        clipped = clip_bbox(item["bbox"], width, height)
        if clipped is None:
            stats["invalid_bbox"] += 1
            continue
        if clipped != item["bbox"]:
            stats["clipped_bbox"] += 1
        valid_objects.append({"label": label, "bbox": clipped})
    return valid_objects, stats


def load_split_ids(output_dir: Path, split: str) -> list[str]:
    """读取 ImageSets/Main 下的 train/val/test 划分。"""

    split_path = output_dir / "ImageSets" / "Main" / f"{split}.txt"
    if not split_path.exists():
        raise SystemExit(f"Split file not found: {split_path}")
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_reference(image_path: Path, output_dir: Path, absolute_image_path: bool) -> str:
    """生成 JSONL 中的图片路径，默认使用相对 output-dir 的路径，便于迁移数据集。"""

    if absolute_image_path:
        return str(image_path.resolve())
    return image_path.resolve().relative_to(output_dir.resolve()).as_posix()


def build_qwen_record(image_path: Path, output_dir: Path, objects: list[dict], prompt: str, absolute_image_path: bool) -> dict:
    """构造 Qwen2-VL/Qwen-VL 常见 messages 格式的一条样本。"""

    answer = json.dumps(objects, ensure_ascii=False)
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_reference(image_path, output_dir, absolute_image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]
    }


def validate_qwen_record(record: dict) -> None:
    """生成后立即校验 messages 结构，防止坏格式进入 JSONL。"""

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("record.messages must contain user and assistant messages")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("record messages must be ordered as user then assistant")

    user_content = user.get("content")
    assistant_content = assistant.get("content")
    if not isinstance(user_content, list) or not isinstance(assistant_content, list):
        raise ValueError("message.content must be a list")

    image_items = [item for item in user_content if isinstance(item, dict) and item.get("type") == "image"]
    text_items = [item for item in user_content if isinstance(item, dict) and item.get("type") == "text"]
    if len(image_items) != 1 or not image_items[0].get("image"):
        raise ValueError("user message must contain exactly one image item with image path")
    if len(text_items) != 1 or not text_items[0].get("text"):
        raise ValueError("user message must contain exactly one text prompt")

    answer_items = [item for item in assistant_content if isinstance(item, dict) and item.get("type") == "text"]
    if len(answer_items) != 1 or not answer_items[0].get("text"):
        raise ValueError("assistant message must contain exactly one text answer")
    answer = json.loads(answer_items[0]["text"])
    if not isinstance(answer, list) or not answer:
        raise ValueError("assistant answer must be a non-empty JSON array")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert merged VOC train split to Qwen-VL JSONL.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Merged VOC dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Merged VOC dataset directory.")
    parser.add_argument("--split", choices=("train", "val", "test", "trainval"), default="train")
    parser.add_argument("--jsonl-path", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--absolute-image-path", action="store_true")
    parser.add_argument("--image-format", choices=("jpeg", "png", "webp"), default="jpeg")
    parser.add_argument("--normalize-images", action="store_true", help="Write RGB training images to qwen_vl_images.")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root or DEFAULT_DATASET_ROOT
    output_dir = dataset_root / "merged_helmet_voc"
    output_dir = output_dir.resolve()
    print(f"[info] using output directory: {output_dir}")
    jsonl_path = (args.jsonl_path or (output_dir / f"qwen_vl_{args.split}.jsonl")).resolve()
    qwen_image_dir = output_dir / "qwen_vl_images"
    summary_path = (args.summary_path or (output_dir / f"qwen_vl_{args.split}_summary.json")).resolve()

    sample_ids = load_split_ids(output_dir, args.split)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if args.normalize_images:
        qwen_image_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    stats: Counter[str] = Counter()
    label_stats: Counter[str] = Counter()
    suffix_stats: Counter[str] = Counter()
    with jsonl_path.open("w", encoding="utf-8") as writer:
        for sample_id in tqdm(sample_ids, desc=f"Converting {args.split}", unit="sample"):
            xml_path = output_dir / "Annotations" / f"{sample_id}.xml"
            if not xml_path.exists():
                skipped += 1
                stats["missing_xml"] += 1
                print(f"[warn] missing xml: {xml_path}")
                continue

            image_path, objects = read_voc_sample(xml_path, output_dir)
            if image_path is None or not objects:
                skipped += 1
                stats["missing_image_or_objects"] += 1
                print(f"[warn] skipped sample without image or objects: {sample_id}")
                continue

            try:
                width, height = read_image_size(image_path)
            except Exception as exc:  # noqa: BLE001 - 数据转换阶段要报告坏图。
                skipped += 1
                stats["unreadable_image"] += 1
                print(f"[warn] unreadable image: {image_path} ({exc})")
                continue

            valid_objects, object_stats = validate_objects(objects, width, height, {"head", "helmet"})
            stats.update(object_stats)
            if not valid_objects:
                skipped += 1
                stats["no_valid_objects"] += 1
                print(f"[warn] skipped sample without valid objects: {sample_id}")
                continue

            train_image_path = image_path
            if args.normalize_images:
                train_image_path = normalize_image_for_training(image_path, qwen_image_dir, sample_id, args.image_format)

            suffix_stats[train_image_path.suffix.lower()] += 1
            label_stats.update(item["label"] for item in valid_objects)

            record = build_qwen_record(train_image_path, output_dir, valid_objects, args.prompt, args.absolute_image_path)
            try:
                validate_qwen_record(record)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                stats["invalid_record"] += 1
                print(f"[warn] invalid generated record for {sample_id}: {exc}")
                continue

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "split": args.split,
        "jsonl_path": str(jsonl_path),
        "sample_count": len(sample_ids),
        "written": written,
        "skipped": skipped,
        "stats": dict(stats),
        "label_stats": dict(label_stats),
        "image_suffix_stats": dict(suffix_stats),
        "normalize_images": args.normalize_images,
        "image_format": args.image_format,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] wrote {written} samples to: {jsonl_path}")
    print(f"[done] summary: {summary_path}")
    if skipped:
        print(f"[done] skipped {skipped} samples")


if __name__ == "__main__":
    main()
