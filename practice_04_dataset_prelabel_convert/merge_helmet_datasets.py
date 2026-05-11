#!/usr/bin/env python
"""整合安全帽检测数据集，并去除近似重复图片。

默认输入：
  dataset/Safety Helmet Detection_datasets_
  dataset/VOC2028

默认输出：
  dataset/merged_helmet_voc

类别归一化：
  head   -> head
  helmet -> helmet
  hat    -> helmet

输出仍然使用 Pascal VOC XML 标注格式，并额外生成 train/val/test 数据划分文件。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.dom import minidom
import xml.etree.ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 只是进度条，缺失时不影响主流程。
    def tqdm(iterable, **_: object):
        return iterable

# 支持查找的常见图片后缀。XML 中的 filename 有时会缺后缀或后缀大小写不一致。
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# 默认约定：dataset 与 practice_04_dataset_prelabel_convert 位于同一级目录。
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"

# 最终训练只保留这两个类别：未戴安全帽的头部、安全帽。
CLASSES = ("head", "helmet")

# 不同数据集对同一语义的命名不同，这里统一映射到最终类别。
CLASS_MAP = {
    "head": "head",
    "helmet": "helmet",
    "hat": "helmet",
}


@dataclass
class ObjectBox:
    """单个目标框，坐标采用 Pascal VOC 的像素坐标：xmin/ymin/xmax/ymax。"""

    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class Record:
    """一张图片及其对应的有效标注信息。"""

    source: str
    xml_path: Path
    image_path: Path
    original_filename: str
    width: int
    height: int
    depth: int
    objects: list[ObjectBox]
    image_hash: int | None = None


class UnionFind:
    """并查集，用来把互相近似重复的图片归为同一组。"""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


class BKNode:
    """BK-tree 节点，用于按汉明距离快速检索相似哈希。"""

    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, BKNode] = {}


class BKTree:
    """BK-tree 适合做小阈值的相似哈希搜索，比两两比较更快。"""

    def __init__(self) -> None:
        self.root: BKNode | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = BKNode(value, index)
            return

        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value, index)
                return
            node = child

    def query(self, value: int, max_distance: int) -> list[int]:
        if self.root is None:
            return []

        result: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)
            if distance <= max_distance:
                result.extend(node.indices)

            lower = distance - max_distance
            upper = distance + max_distance
            for child_distance, child in node.children.items():
                if lower <= child_distance <= upper:
                    stack.append(child)
        return result


def import_pillow_image():
    """延迟导入 Pillow，使 --help 等不读图的命令不受依赖影响。"""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - friendly runtime guard
        raise SystemExit(
            "This script needs Pillow for image hashing/size checks. Install it with: pip install pillow"
        ) from exc
    return Image


def hamming_distance(left: int, right: int) -> int:
    """计算两个整数哈希之间不同 bit 的数量。"""

    return (left ^ right).bit_count()


def image_dhash(path: Path, hash_size: int = 8) -> int:
    """计算 64 位 dHash；图片轻微缩放、压缩后通常仍能保持较近的哈希距离。"""

    Image = import_pillow_image()
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())

    value = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            value = (value << 1) | int(left > right)
    return value


def text_or_empty(element: ET.Element | None) -> str:
    """安全读取 XML 节点文本，缺失时返回空字符串。"""

    return "" if element is None or element.text is None else element.text.strip()


def int_text(element: ET.Element | None, default: int = 0) -> int:
    """安全读取 XML 节点中的数字，兼容 '12.0' 这类文本。"""

    text = text_or_empty(element)
    try:
        return int(float(text))
    except ValueError:
        return default


def find_image_for_xml(xml_path: Path, image_dir: Path, filename: str) -> Path | None:
    """根据 XML 中的 filename 或 XML 文件名，在图片目录中找到真实图片。"""

    if filename:
        candidate = image_dir / filename
        if candidate.exists():
            return candidate

        stem_candidate = image_dir / Path(filename).stem
        for suffix in IMAGE_EXTENSIONS:
            found = stem_candidate.with_suffix(suffix)
            if found.exists():
                return found

    stem = xml_path.stem
    for suffix in IMAGE_EXTENSIONS:
        found = image_dir / f"{stem}{suffix}"
        if found.exists():
            return found
    return None


def read_image_size(path: Path) -> tuple[int, int, int]:
    """当 XML 缺少 size 字段时，从图片本身读取宽高和通道数。"""

    Image = import_pillow_image()
    with Image.open(path) as image:
        width, height = image.size
        bands = image.getbands()
        depth = len(bands) if bands else 3
    return width, height, depth


def parse_record(xml_path: Path, image_dir: Path, source: str) -> tuple[Record | None, Counter]:
    """解析单个 VOC XML，并过滤无效类别或无效 bbox。"""

    stats: Counter = Counter()
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        stats["bad_xml"] += 1
        return None, stats

    original_filename = text_or_empty(root.find("filename"))
    image_path = find_image_for_xml(xml_path, image_dir, original_filename)
    if image_path is None:
        stats["missing_image"] += 1
        return None, stats

    size = root.find("size")
    width = int_text(size.find("width") if size is not None else None)
    height = int_text(size.find("height") if size is not None else None)
    depth = int_text(size.find("depth") if size is not None else None, 3)
    if width <= 0 or height <= 0:
        width, height, depth = read_image_size(image_path)

    objects: list[ObjectBox] = []
    for obj in root.findall("object"):
        raw_name = text_or_empty(obj.find("name")).lower()
        name = CLASS_MAP.get(raw_name)
        if name is None:
            stats[f"skipped_class:{raw_name or '<empty>'}"] += 1
            continue

        box = obj.find("bndbox")
        if box is None:
            stats["missing_box"] += 1
            continue

        # 将框裁剪到图片范围内，避免后续训练或可视化时坐标越界。
        xmin = max(1, min(width, int_text(box.find("xmin"), 1)))
        ymin = max(1, min(height, int_text(box.find("ymin"), 1)))
        xmax = max(1, min(width, int_text(box.find("xmax"), width)))
        ymax = max(1, min(height, int_text(box.find("ymax"), height)))

        if xmax <= xmin or ymax <= ymin:
            stats["bad_box"] += 1
            continue
        objects.append(ObjectBox(name, xmin, ymin, xmax, ymax))

    if not objects:
        stats["empty_after_filter"] += 1
        return None, stats

    return (
        Record(
            source=source,
            xml_path=xml_path,
            image_path=image_path,
            original_filename=original_filename or image_path.name,
            width=width,
            height=height,
            depth=depth,
            objects=objects,
        ),
        stats,
    )


def load_records(root_dir: Path) -> tuple[list[Record], Counter]:
    """加载两个原始数据集，返回统一后的 Record 列表和统计信息。"""

    safety_root = root_dir / "Safety Helmet Detection_datasets_"
    voc_root = root_dir / "VOC2028"

    dataset_specs = [
        ("safety_helmet", safety_root / "annotations", safety_root / "images"),
        ("voc2028", voc_root / "Annotations", voc_root / "JPEGImages"),
    ]

    records: list[Record] = []
    stats: Counter = Counter()
    for source, annotation_dir, image_dir in dataset_specs:
        if not annotation_dir.exists():
            stats[f"missing_annotation_dir:{source}"] += 1
            continue
        if not image_dir.exists():
            stats[f"missing_image_dir:{source}"] += 1
            continue

        for xml_path in tqdm(sorted(annotation_dir.glob("*.xml")), desc=f"Loading {source}", unit="file"):
            stats[f"xml_seen:{source}"] += 1
            record, record_stats = parse_record(xml_path, image_dir, source)
            stats.update(record_stats)
            if record is not None:
                records.append(record)
                stats[f"record_loaded:{source}"] += 1

    return records, stats


def choose_best_record(indices: Iterable[int], records: list[Record]) -> int:
    """近似重复图片中，优先保留标注框更多的一张。"""

    def score(index: int) -> tuple[int, int, str]:
        record = records[index]
        source_score = 1 if record.source == "safety_helmet" else 0
        return (len(record.objects), source_score, record.image_path.name)

    return max(indices, key=score)


def deduplicate_records(records: list[Record], threshold: int) -> tuple[list[Record], list[dict]]:
    """按 dHash 汉明距离去重；threshold 越大，去重越激进。"""

    if threshold < 0:
        return records, []

    tree = BKTree()
    union_find = UnionFind(len(records))
    duplicate_edges: list[tuple[int, int, int]] = []

    for index, record in enumerate(tqdm(records, desc="Deduplicating", unit="record")):
        try:
            image_hash = image_dhash(record.image_path)
        except Exception as exc:  # Keep unreadable hashes out, but do not lose data.
            print(f"[warn] cannot hash {record.image_path}: {exc}", file=sys.stderr)
            continue

        record.image_hash = image_hash
        # 先查询已有图片中与当前图片相近的样本，再把当前图片加入索引。
        for match_index in tree.query(image_hash, threshold):
            match_hash = records[match_index].image_hash
            if match_hash is None:
                continue
            distance = hamming_distance(image_hash, match_hash)
            union_find.union(index, match_index)
            duplicate_edges.append((index, match_index, distance))
        tree.add(image_hash, index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union_find.find(index)].append(index)

    kept_indices = set()
    duplicates: list[dict] = []
    for indices in groups.values():
        best = choose_best_record(indices, records)
        kept_indices.add(best)
        for index in indices:
            if index != best:
                duplicates.append(
                    {
                        "dropped": str(records[index].image_path),
                        "kept": str(records[best].image_path),
                        "dropped_source": records[index].source,
                        "kept_source": records[best].source,
                    }
                )

    kept_records = [records[index] for index in range(len(records)) if index in kept_indices]
    return kept_records, duplicates


def make_xml(record: Record, new_filename: str) -> str:
    """把统一后的 Record 重新写成规范的 Pascal VOC XML 字符串。"""

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "JPEGImages"
    ET.SubElement(annotation, "filename").text = new_filename
    ET.SubElement(annotation, "path").text = f"JPEGImages/{new_filename}"

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = record.source

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(record.width)
    ET.SubElement(size, "height").text = str(record.height)
    ET.SubElement(size, "depth").text = str(record.depth)
    ET.SubElement(annotation, "segmented").text = "0"

    for item in record.objects:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = item.name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(item.xmin)
        ET.SubElement(box, "ymin").text = str(item.ymin)
        ET.SubElement(box, "xmax").text = str(item.xmax)
        ET.SubElement(box, "ymax").text = str(item.ymax)

    rough = ET.tostring(annotation, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    return pretty.decode("utf-8")


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    """准备输出目录；开启 --overwrite 时会删除旧的合并结果。"""

    if output_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    (output_dir / "Annotations").mkdir(parents=True, exist_ok=True)
    (output_dir / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (output_dir / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)


def write_split_files(image_ids: list[str], split_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> None:
    """按照固定随机种子生成 train/val/test/trainval 四个划分文件。"""

    ids = image_ids[:]
    random.Random(seed).shuffle(ids)

    train_count = int(len(ids) * train_ratio)
    val_count = int(len(ids) * val_ratio)
    train_ids = ids[:train_count]
    val_ids = ids[train_count : train_count + val_count]
    test_ids = ids[train_count + val_count :]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "trainval": train_ids + val_ids,
    }
    for name, values in splits.items():
        (split_dir / f"{name}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")


def write_records(records: list[Record], output_dir: Path, seed: int, train_ratio: float, val_ratio: float) -> Counter:
    """复制图片、写 XML、写 classes.txt 和数据划分文件。"""

    image_dir = output_dir / "JPEGImages"
    annotation_dir = output_dir / "Annotations"
    split_dir = output_dir / "ImageSets" / "Main"

    stats: Counter = Counter()
    image_ids: list[str] = []

    for index, record in enumerate(records):
        image_id = f"{index:06d}"
        suffix = record.image_path.suffix.lower()
        if suffix == ".jpeg":
            suffix = ".jpg"

        new_filename = f"{image_id}{suffix}"
        shutil.copy2(record.image_path, image_dir / new_filename)
        (annotation_dir / f"{image_id}.xml").write_text(make_xml(record, new_filename), encoding="utf-8")

        image_ids.append(image_id)
        stats[f"output_source:{record.source}"] += 1
        for obj in record.objects:
            stats[f"class:{obj.name}"] += 1

    (output_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    write_split_files(image_ids, split_dir, train_ratio, val_ratio, seed)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Safety Helmet Detection_datasets_ and VOC2028 into one VOC dataset."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dedupe-threshold", type=int, default=4, help="DHash Hamming threshold. Use -1 to disable.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """命令行入口：加载、去重、写出合并数据集和处理摘要。"""

    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = (args.output_dir or (dataset_root / "merged_helmet_voc")).resolve()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("--train-ratio and --val-ratio must leave a positive test split.")

    print(f"[info] dataset root: {dataset_root}")
    records, load_stats = load_records(dataset_root)
    print(f"[info] loaded records: {len(records)}")

    deduped_records, duplicates = deduplicate_records(records, args.dedupe_threshold)
    print(f"[info] kept records after dedupe: {len(deduped_records)}")
    print(f"[info] near-duplicates removed: {len(duplicates)}")

    reset_output_dir(output_dir, args.overwrite)
    output_stats = write_records(deduped_records, output_dir, args.seed, args.train_ratio, args.val_ratio)

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "class_map": CLASS_MAP,
        "classes": CLASSES,
        "dedupe_threshold": args.dedupe_threshold,
        "input_records": len(records),
        "output_records": len(deduped_records),
        "duplicates_removed": len(duplicates),
        "load_stats": dict(load_stats),
        "output_stats": dict(output_stats),
        "duplicates_sample": duplicates[:100],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] wrote merged dataset to: {output_dir}")
    print(f"[done] summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
