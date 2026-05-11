"""Run SAM3 segmentation with box prompts.

SAM3 in transformers uses a different API from SAM2:
`processor(images=..., input_boxes=..., input_boxes_labels=...)`.
This script keeps the SAM3 path explicit for classroom comparison.
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


def import_sam3():
    """Import SAM3-compatible auto classes lazily."""

    try:
        import torch
        from transformers import AutoModelForMaskGeneration, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 requires torch and a transformers version with AutoModelForMaskGeneration/AutoProcessor support."
        ) from exc
    return torch, AutoModelForMaskGeneration, AutoProcessor


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
    """Load SAM3 model and processor."""

    torch, AutoModelForMaskGeneration, AutoProcessor = import_sam3()
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = str(cfg.get("sam3_model_name") or cfg.get("sam_model_name") or "facebook/sam3")
    model = AutoModelForMaskGeneration.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return torch, model, processor, device


def to_numpy(value: Any) -> np.ndarray:
    """Convert torch/list/numpy values to numpy arrays."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def to_numpy_scores(scores: Any | None, count: int) -> np.ndarray:
    """Convert SAM3 optional scores to a score per mask."""

    if scores is None:
        return np.ones(count, dtype=np.float32)
    scores_np = to_numpy(scores).reshape(-1)
    if scores_np.size == 0:
        return np.ones(count, dtype=np.float32)
    if scores_np.size < count:
        scores_np = np.pad(scores_np, (0, count - scores_np.size), constant_values=float(scores_np[-1]))
    return scores_np[:count]


def postprocess_sam3(processor, outputs, inputs, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert SAM3 raw outputs to original-size binary masks."""

    target_sizes = inputs.get("original_sizes")
    if hasattr(target_sizes, "tolist"):
        target_sizes = target_sizes.tolist()

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    masks = results.get("masks")
    if masks is None:
        masks = []
    masks_np = to_numpy(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    elif masks_np.ndim > 3:
        masks_np = masks_np.reshape((-1,) + masks_np.shape[-2:])

    scores = results.get("scores")
    if scores is None:
        scores = results.get("confidence_scores")
    return masks_np, to_numpy_scores(scores, len(masks_np))


def point_to_box(point: list[float], image: Image.Image, radius: int = 12) -> list[float]:
    """Convert a point to a tiny box so SAM3 can consume it as a box prompt."""

    x, y = float(point[0]), float(point[1])
    width, height = image.size
    return [
        max(0.0, x - radius),
        max(0.0, y - radius),
        min(float(width - 1), x + radius),
        min(float(height - 1), y + radius),
    ]


def run_box_prompt(image: Image.Image, box: list[float], model, processor, torch, device: str, threshold: float):
    """Run one positive SAM3 box prompt."""

    box_xyxy = [float(v) for v in box]
    inputs = processor(
        images=image,
        input_boxes=[[box_xyxy]],
        input_boxes_labels=[[1]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return postprocess_sam3(processor, outputs, inputs, threshold)


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    """Convert a binary mask to an xyxy bbox."""

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def draw_mask(image: np.ndarray, mask: np.ndarray, color=(255, 170, 40), alpha=0.5) -> np.ndarray:
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
                "model": "sam3",
                "source": item["source"],
                "prompt": item["prompt"],
                "bbox": bbox,
                "score": item["score"],
            }
        )

    (output_dir / "segments.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(records)} SAM3 masks to {output_dir}")


def collect_predictions(image: Image.Image, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Run configured SAM3 prompts."""

    torch, model, processor, device = build_model_and_processor(cfg)
    threshold = float(cfg.get("confidence_threshold", 0.5))
    items: list[dict[str, Any]] = []

    for prompt in cfg.get("text_prompts", []):
        print(f"warning: text prompt '{prompt}' is kept for teaching only; this script visualizes box prompts.")

    for box in cfg.get("box_prompts", []):
        masks, scores = run_box_prompt(image, box, model, processor, torch, device, threshold)
        for idx, mask in enumerate(masks):
            binary_mask = (np.squeeze(mask) > 0).astype(np.uint8) * 255
            items.append({"source": "box", "prompt": box, "mask": binary_mask, "score": float(scores[idx])})

    for point in cfg.get("point_prompts", []):
        pseudo_box = point_to_box(point, image)
        masks, scores = run_box_prompt(image, pseudo_box, model, processor, torch, device, threshold)
        for idx, mask in enumerate(masks):
            binary_mask = (np.squeeze(mask) > 0).astype(np.uint8) * 255
            items.append({"source": "point_as_box", "prompt": point, "box": pseudo_box, "mask": binary_mask, "score": float(scores[idx])})

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3 segmentation with box prompts.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    args = parser.parse_args()
    require_cv2()

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    config_dir = config_path.parent

    Image = import_image()
    image = Image.open(resolve_path(cfg["image_path"], config_dir)).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    output_dir = resolve_path(cfg.get("output_dir", "outputs"), config_dir) / "sam3"

    items = collect_predictions(image, cfg)
    save_outputs(image_np, output_dir, items)


if __name__ == "__main__":
    main()
