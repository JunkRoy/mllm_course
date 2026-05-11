#!/usr/bin/env python
"""把合并后的 VOC 训练集转换为 Qwen-VL 指令微调 JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 只是进度条，缺失时不影响主流程。
    def tqdm(iterable, **_: object):
        return iterable


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# 默认约定：dataset 与 practice_04_dataset_prelabel_convert 位于同一级目录。
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert merged VOC train split to Qwen-VL JSONL.")
    default_output = DEFAULT_DATASET_ROOT / "merged_helmet_voc"
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Merged VOC dataset directory.")
    parser.add_argument("--split", choices=("train", "val", "test", "trainval"), default="train")
    parser.add_argument("--jsonl-path", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--absolute-image-path", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    jsonl_path = (args.jsonl_path or (output_dir / f"qwen_vl_{args.split}.jsonl")).resolve()

    sample_ids = load_split_ids(output_dir, args.split)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with jsonl_path.open("w", encoding="utf-8") as writer:
        for sample_id in tqdm(sample_ids, desc=f"Converting {args.split}", unit="sample"):
            xml_path = output_dir / "Annotations" / f"{sample_id}.xml"
            if not xml_path.exists():
                skipped += 1
                print(f"[warn] missing xml: {xml_path}")
                continue

            image_path, objects = read_voc_sample(xml_path, output_dir)
            if image_path is None or not objects:
                skipped += 1
                print(f"[warn] skipped sample without image or objects: {sample_id}")
                continue

            record = build_qwen_record(image_path, output_dir, objects, args.prompt, args.absolute_image_path)
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[done] wrote {written} samples to: {jsonl_path}")
    if skipped:
        print(f"[done] skipped {skipped} samples")


if __name__ == "__main__":
    main()
