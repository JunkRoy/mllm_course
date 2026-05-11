import argparse
import json
import random
import shutil
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """读取数据构造与格式转换配置。"""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_images(image_dir: Path, ratios: dict) -> dict[str, list[Path]]:
    """按配置比例将图片划分为 train/val/test。"""

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    random.seed(42)
    random.shuffle(images)
    n = len(images)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])
    return {"train": images[:train_end], "val": images[train_end:val_end], "test": images[val_end:]}


def write_coco(records: list[dict], class_names: list[str], path: Path) -> None:
    """将内部 records 写成 COCO 检测格式。"""

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    images, annotations = [], []
    for image_id, item in enumerate(records, start=1):
        images.append({"id": image_id, "file_name": item["file_name"], "width": item.get("width", 0), "height": item.get("height", 0)})
        for ann in item.get("annotations", []):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": class_names.index(ann["label"]) + 1,
                    "bbox": ann["bbox"],
                    "area": ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd": 0,
                }
            )
    path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(records: list[dict], path: Path) -> None:
    """将 records 按一行一个 JSON 样本写成 JSONL。"""

    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def prelabel_placeholder(records: list[dict], class_names: list[str]) -> list[dict]:
    """为没有标注的样本补一个占位框，便于演示完整转换流程。"""

    prepared = []
    default_label = class_names[0] if class_names else "object"
    for item in records:
        if item.get("annotations"):
            prepared.append(item)
            continue
        prepared.append(
            {
                **item,
                "annotations": [
                    {
                        "label": default_label,
                        "bbox": [0, 0, max(1, int(item.get("width", 1)) // 2), max(1, int(item.get("height", 1)) // 2)],
                        "source": "prelabel_placeholder",
                    }
                ],
            }
        )
    return prepared


def write_yolo(records: list[dict], class_names: list[str], output_dir: Path) -> None:
    """将 bbox 转成 YOLO 归一化格式并逐图写入 txt。"""

    label_dir = output_dir / "labels_yolo"
    label_dir.mkdir(parents=True, exist_ok=True)
    for item in records:
        width, height = max(1, item.get("width", 1)), max(1, item.get("height", 1))
        lines = []
        for ann in item.get("annotations", []):
            x, y, w, h = ann["bbox"]
            cls = class_names.index(ann["label"])
            lines.append(f"{cls} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}")
        Path(label_dir / f"{Path(item['file_name']).stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """命令行入口：划分图片并导出 COCO、YOLO 和 JSONL。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = split_images(Path(cfg["raw_image_dir"]), cfg["splits"])
    for split, paths in splits.items():
        target = output_dir / split / "images"
        target.mkdir(parents=True, exist_ok=True)
        for src in paths:
            shutil.copy2(src, target / src.name)

    records = json.loads(Path(cfg["raw_annotation_path"]).read_text(encoding="utf-8"))
    records = prelabel_placeholder(records, cfg["class_names"])
    write_coco(records, cfg["class_names"], output_dir / "annotations_coco.json")
    write_yolo(records, cfg["class_names"], output_dir)
    write_jsonl(records, output_dir / "samples.jsonl")
    print(f"dataset written to {output_dir}")


if __name__ == "__main__":
    main()
