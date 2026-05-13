"""Run SAM2 prompt segmentation with point and box prompts.

This script is intentionally separate from the SAM3 script because SAM2 and
SAM3 use different processor/model calling conventions in transformers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - only needed when running inference.
    cv2 = None


def load_config(path: str) -> dict[str, Any]:
    """Load YAML config."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing dependency: please install PyYAML first.") from exc

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    """Resolve a config path relative to the config file directory."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def import_sam2():
    """Import SAM2 classes lazily so --help stays lightweight."""

    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 requires torch and a transformers version with Sam2Model/Sam2Processor."
        ) from exc
    return torch, Sam2Model, Sam2Processor


def import_image():
    """Import Pillow lazily."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing dependency: please install Pillow first.") from exc
    return Image


def require_cv2():
    """Ensure OpenCV is available before writing visual outputs."""

    if cv2 is None:
        raise RuntimeError("Missing dependency: please install opencv-python first.")


def build_model_and_processor(cfg: dict[str, Any]):
    """Load SAM2 model and processor."""

    torch, Sam2Model, Sam2Processor = import_sam2()
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = str(cfg.get("sam2_model_name") or cfg.get("sam_model_name") or "facebook/sam2.1-hiera-large")
    model = Sam2Model.from_pretrained(model_name).to(device)
    processor = Sam2Processor.from_pretrained(model_name)
    return torch, model, processor, device


def to_numpy(value: Any) -> np.ndarray:
    """Convert torch/list/numpy values to numpy arrays."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def flatten_masks_and_scores(masks: Any, scores: Any | None) -> tuple[np.ndarray, np.ndarray]:
    """Flatten SAM2 multimask outputs to [N, H, W] plus a score per mask."""

    masks_np = to_numpy(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    elif masks_np.ndim > 3:
        masks_np = masks_np.reshape((-1,) + masks_np.shape[-2:])

    if scores is None:
        scores_np = np.ones(len(masks_np), dtype=np.float32)
    else:
        scores_np = to_numpy(scores).reshape(-1)
        if scores_np.size == 0:
            scores_np = np.ones(len(masks_np), dtype=np.float32)
        elif scores_np.size < len(masks_np):
            scores_np = np.pad(scores_np, (0, len(masks_np) - scores_np.size), constant_values=float(scores_np[-1]))
        else:
            scores_np = scores_np[: len(masks_np)]
    return masks_np, scores_np


def postprocess_sam2(processor, outputs, inputs) -> tuple[np.ndarray, np.ndarray]:
    """Restore SAM2 masks to original image size."""

    # SAM2 post_process_masks only needs original_sizes. The older SAM
    # processor also required reshaped_input_sizes, but SAM2 does not return it.
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
    )[0]
    scores = getattr(outputs, "iou_scores", None)
    if scores is not None:
        scores = scores[0]
    return flatten_masks_and_scores(masks, scores)


def run_box_prompt(image: Image.Image, box: list[float], model, processor, torch, device: str):
    """Run one SAM2 box prompt."""

    box_xyxy = [float(v) for v in box]
    # Sam2Processor expects box nesting as:
    # [image level, box level, box coordinates].
    inputs = processor(images=image, input_boxes=[[box_xyxy]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return postprocess_sam2(processor, outputs, inputs)


def run_point_prompt(image: Image.Image, point: list[float], model, processor, torch, device: str):
    """Run one SAM2 point prompt."""

    label = int(point[2]) if len(point) > 2 else 1
    # Sam2Processor expects point nesting as:
    # [image level, object level, point level, xy coordinate].
    inputs = processor(
        images=image,
        input_points=[[[[float(point[0]), float(point[1])]]]],
        input_labels=[[[label]]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return postprocess_sam2(processor, outputs, inputs)


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    """Convert a binary mask to an xyxy bbox."""

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def draw_mask(image: np.ndarray, mask: np.ndarray, color=(30, 180, 255), alpha=0.5) -> np.ndarray:
    """Draw a semi-transparent mask and contour."""

    vis = image.copy()
    color_layer = np.zeros_like(vis)
    color_layer[mask > 0] = color
    vis = cv2.addWeighted(color_layer, alpha, vis, 1 - alpha, 0)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
    return vis


def save_outputs(image_np: np.ndarray, output_dir: Path, items: list[dict[str, Any]]) -> None:
    """Save masks, visualization images, and segments.json."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, item in enumerate(items):
        mask = item["mask"]
        bbox = mask_to_bbox(mask)

        cv2.imwrite(str(output_dir / f"mask_{idx:03d}.png"), mask)
        vis = draw_mask(image_np, mask)
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (30, 255, 30), 2)
        cv2.putText(
            vis,
            f"{item['source']} score={item['score']:.3f}",
            (bbox[0], max(20, bbox[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(output_dir / f"visual_{idx:03d}.jpg"), vis)
        records.append(
            {
                "id": idx,
                "model": "sam2",
                "source": item["source"],
                "prompt": item["prompt"],
                "bbox": bbox,
                "score": item["score"],
            }
        )

    (output_dir / "segments.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(records)} SAM2 masks to {output_dir}")


def collect_predictions(image: Image.Image, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Run all configured SAM2 prompts."""

    torch, model, processor, device = build_model_and_processor(cfg)
    threshold = float(cfg.get("confidence_threshold", 0.5))
    items: list[dict[str, Any]] = []

    for box in cfg.get("box_prompts", []):
        masks, scores = run_box_prompt(image, box, model, processor, torch, device)
        for idx, mask in enumerate(masks):
            binary_mask = (np.squeeze(mask) > threshold).astype(np.uint8) * 255
            items.append({"source": "box", "prompt": box, "mask": binary_mask, "score": float(scores[idx])})

    for point in cfg.get("point_prompts", []):
        masks, scores = run_point_prompt(image, point, model, processor, torch, device)
        for idx, mask in enumerate(masks):
            binary_mask = (np.squeeze(mask) > threshold).astype(np.uint8) * 255
            items.append({"source": "point", "prompt": point, "mask": binary_mask, "score": float(scores[idx])})

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM2 segmentation with point and box prompts.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    args = parser.parse_args()
    require_cv2()

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    config_dir = config_path.parent

    Image = import_image()
    image = Image.open(resolve_path(cfg["image_path"], config_dir)).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    output_dir = resolve_path(cfg.get("output_dir", "outputs"), config_dir) / "sam2"

    items = collect_predictions(image, cfg)
    save_outputs(image_np, output_dir, items)


if __name__ == "__main__":
    main()
