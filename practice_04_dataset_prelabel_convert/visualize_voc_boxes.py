#!/usr/bin/env python
"""将 VOC XML 标注框和类别名画到图片上，便于人工检查数据质量。"""

from __future__ import annotations
import argparse
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
CLASS_COLORS = {
    "head": (255, 99, 71),
    "helmet": (50, 180, 90),
}


def import_pillow():
    """延迟导入 Pillow，缺依赖时给出清晰提示。"""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("This script needs Pillow. Install it with: pip install pillow") from exc
    return Image, ImageDraw, ImageFont


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
    """优先使用 XML 中的 filename；找不到时退回到 XML 文件名匹配。"""

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


def read_objects(xml_path: Path) -> tuple[str, list[dict]]:
    """从单个 VOC XML 中读取图片名、类别名和 bbox。"""

    root = ET.parse(xml_path).getroot()
    filename = text_or_empty(root.find("filename"))
    objects: list[dict] = []

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        objects.append(
            {
                "label": text_or_empty(obj.find("name")),
                "bbox": [
                    int_text(box.find("xmin")),
                    int_text(box.find("ymin")),
                    int_text(box.find("xmax")),
                    int_text(box.find("ymax")),
                ],
            }
        )
    return filename, objects


def draw_one(xml_path: Path, output_dir: Path, visual_dir: Path, line_width: int) -> bool:
    """绘制单张图片的所有标注框。"""

    # main 中会传入 Path；这里再包一层，兼容外部调用时误传字符串。
    xml_path = Path(xml_path)
    output_dir = Path(output_dir)
    visual_dir = Path(visual_dir)

    Image, ImageDraw, ImageFont = import_pillow()
    filename, objects = read_objects(xml_path)
    image_path = find_image(output_dir, filename, xml_path.stem)
    if image_path is None:
        print(f"[warn] missing image for {xml_path}")
        return False

    with Image.open(image_path) as image:
        canvas = image.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for item in objects:
        label = item["label"]
        xmin, ymin, xmax, ymax = item["bbox"]
        color = CLASS_COLORS.get(label, (255, 215, 0))

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)

        # 画一个实心标签背景，避免类别文字被图片内容淹没。
        text = f"{label} [{xmin},{ymin},{xmax},{ymax}]"
        left, top, right, bottom = draw.textbbox((xmin, ymin), text, font=font)
        text_height = bottom - top
        text_width = right - left
        text_y = max(0, ymin - text_height - 4)
        draw.rectangle([xmin, text_y, xmin + text_width + 6, text_y + text_height + 4], fill=color)
        draw.text((xmin + 3, text_y + 2), text, fill=(0, 0, 0), font=font)

    visual_dir.mkdir(parents=True, exist_ok=True)

    save_path = visual_dir / f"{xml_path.stem}.jpg"
    canvas.save(save_path, quality=95)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw VOC boxes and labels on merged dataset images.")
    # default_output = DEFAULT_DATASET_ROOT / "merged_helmet_voc"
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None, help="Merged VOC dataset directory.")
    parser.add_argument("--visual-dir", type=Path, default=None, help="Directory for rendered images.")
    parser.add_argument("--max-images", type=int, default=20, help="0 means visualize all images.")
    parser.add_argument("--line-width", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_output = args.dataset_root / "merged_helmet_voc"
    output_dir = args.output_dir.resolve() if args.output_dir else default_output

    visual_dir = args.visual_dir.resolve() if args.visual_dir else output_dir / "visualization"
    annotation_dir = output_dir / "Annotations"

    xml_files = sorted(annotation_dir.glob("*.xml"))
    if args.max_images > 0:
        xml_files = xml_files[: args.max_images]

    ok_count = 0
    for xml_path in tqdm(xml_files, desc="Visualizing", unit="image"):
        ok_count += int(draw_one(xml_path, output_dir, visual_dir, args.line_width))

    print(f"[done] rendered {ok_count} images to: {visual_dir}")


if __name__ == "__main__":
    main()
