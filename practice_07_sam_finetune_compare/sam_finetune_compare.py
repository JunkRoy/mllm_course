import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_sam_adapter_placeholder(cfg: dict, output_dir: Path) -> Path:
    # Replace with SAM/SAM3 mask decoder or adapter training after local weights are prepared.
    checkpoint = output_dir / "sam_adapter.pt"
    checkpoint.write_text(json.dumps({"source": cfg["train_annotation"], "epochs": cfg["epochs"]}, ensure_ascii=False), encoding="utf-8")
    return checkpoint


def predict_placeholder(image: Image.Image, mode: str) -> np.ndarray:
    w, h = image.size
    mask = np.zeros((h, w), dtype=np.uint8)
    pad = 80 if mode == "before" else 50
    cv2.rectangle(mask, (pad, pad), (w - pad, h - pad), 255, -1)
    return mask


def export_coco_mask(mask: np.ndarray, image_name: str, path: Path) -> None:
    ys, xs = np.where(mask > 0)
    bbox = [0, 0, 0, 0] if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]
    coco = {
        "images": [{"id": 1, "file_name": image_name, "width": int(mask.shape[1]), "height": int(mask.shape[0])}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": bbox, "area": int((mask > 0).sum()), "iscrowd": 0}],
        "categories": [{"id": 1, "name": "target"}],
    }
    path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = train_sam_adapter_placeholder(cfg, output_dir)
    image = Image.open(cfg["val_image_path"]).convert("RGB")
    before = predict_placeholder(image, "before")
    after = predict_placeholder(image, "after")
    cv2.imwrite(str(output_dir / "mask_before.png"), before)
    cv2.imwrite(str(output_dir / "mask_after.png"), after)
    export_coco_mask(after, Path(cfg["val_image_path"]).name, output_dir / "sam_result_coco.json")
    diff = cv2.absdiff(before, after)
    cv2.imwrite(str(output_dir / "mask_diff.png"), diff)
    summary = {"checkpoint": str(checkpoint), "before_area": int(before.sum() / 255), "after_area": int(after.sum() / 255)}
    (output_dir / "compare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
