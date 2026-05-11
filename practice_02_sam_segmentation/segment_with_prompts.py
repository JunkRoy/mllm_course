"""SAM prompt segmentation demo based on Hugging Face transformers.

The script supports two common processor styles:
1. SAM/SAM2 style processors that accept point and box prompts directly.
2. SAM3 style processors that use `images=...`, `input_boxes=...`, and
   `input_boxes_labels=...`, followed by `post_process_instance_segmentation`.

For SAM3, point prompts are converted to small positive boxes because the
current SAM3 processor focuses on concept/text/box prompts.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image


def require_sam_transformers():
    """Import heavy dependencies lazily and report a clear error if missing."""

    try:
        import torch
        from transformers import AutoModelForMaskGeneration, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Missing dependencies. Please install PyTorch and transformers first.") from exc
    return torch, AutoModelForMaskGeneration, AutoProcessor


def load_config(path: str) -> dict[str, Any]:
    """Load the YAML config file."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    """Resolve config paths relative to the config file, not the shell cwd."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_model_name(cfg: dict[str, Any]) -> str:
    """Read the SAM model id or local model path from config."""

    return str(cfg.get("sam_model_name") or "facebook/sam3")


def build_model_and_processor(cfg: dict[str, Any]):
    """Create model, processor, torch module, and device string."""

    torch, AutoModelForMaskGeneration, AutoProcessor = require_sam_transformers()
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = resolve_model_name(cfg)
    model = AutoModelForMaskGeneration.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return torch, model, processor, device


def to_numpy_masks(masks: Any) -> np.ndarray:
    """Convert torch/list/numpy masks to a numpy array."""

    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    return np.asarray(masks)


def to_numpy_scores(scores: Any, count: int) -> np.ndarray:
    """Convert optional scores to a 1-D numpy array."""

    if scores is None:
        return np.ones(count, dtype=np.float32)
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores).reshape(-1)
    if scores.size == 0:
        return np.ones(count, dtype=np.float32)
    if scores.size < count:
        scores = np.pad(scores, (0, count - scores.size), constant_values=float(scores[-1]))
    return scores[:count]


def postprocess_legacy_masks(processor, outputs, inputs) -> tuple[np.ndarray, np.ndarray]:
    """Post-process masks for SAM/SAM2 style outputs."""

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
        reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = getattr(outputs, "iou_scores", None)
    if scores is not None:
        scores = scores[0].detach().cpu().numpy()
    masks_np = to_numpy_masks(masks)
    return masks_np, to_numpy_scores(scores, len(masks_np))


def postprocess_sam3_masks(processor, outputs, inputs, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Post-process masks for SAM3 style outputs."""

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
    scores = results.get("scores")
    if scores is None:
        scores = results.get("confidence_scores")
    masks_np = to_numpy_masks(masks)
    return masks_np, to_numpy_scores(scores, len(masks_np))


def run_box_prompt(image: Image.Image, box: list[float], model, processor, torch, device: str, threshold: float):
    """Run one positive box prompt and return masks/scores."""

    box_xyxy = [float(v) for v in box]

    if hasattr(processor, "post_process_instance_segmentation"):
        inputs = processor(
            images=image,
            input_boxes=[[box_xyxy]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return postprocess_sam3_masks(processor, outputs, inputs, threshold)

    inputs = processor(image, input_boxes=[[[box_xyxy]]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return postprocess_legacy_masks(processor, outputs, inputs)


def point_to_box(point: list[float], image: Image.Image, radius: int = 12) -> list[float]:
    """Convert a point prompt to a small xyxy box for SAM3 compatibility."""

    x, y = float(point[0]), float(point[1])
    width, height = image.size
    return [
        max(0.0, x - radius),
        max(0.0, y - radius),
        min(float(width - 1), x + radius),
        min(float(height - 1), y + radius),
    ]


def run_point_prompt(image: Image.Image, point: list[float], model, processor, torch, device: str, threshold: float):
    """Run one point prompt; SAM3 falls back to a small box around the point."""

    if hasattr(processor, "post_process_instance_segmentation"):
        return run_box_prompt(image, point_to_box(point, image), model, processor, torch, device, threshold)

    label = int(point[2]) if len(point) > 2 else 1
    inputs = processor(
        image,
        input_points=[[[[float(point[0]), float(point[1])]]]],
        input_labels=[[[label]]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return postprocess_legacy_masks(processor, outputs, inputs)


def collect_predictions(image: Image.Image, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect all segmentation candidates from configured prompts."""

    torch, model, processor, device = build_model_and_processor(cfg)
    threshold = float(cfg.get("confidence_threshold", 0.5))
    items: list[dict[str, Any]] = []

    for prompt in cfg.get("text_prompts", []):
        print(f"warning: text prompt '{prompt}' is kept for teaching only and is skipped by this script.")

    for box in cfg.get("box_prompts", []):
        masks, scores = run_box_prompt(image, box, model, processor, torch, device, threshold)
        for idx, mask in enumerate(masks):
            binary_mask = (np.squeeze(mask) > 0).astype(np.uint8) * 255
            items.append({"source": "box", "prompt": box, "mask": binary_mask, "score": float(scores[idx])})

    for point in cfg.get("point_prompts", []):
        masks, scores = run_point_prompt(image, point, model, processor, torch, device, threshold)
        for idx, mask in enumerate(masks):
            binary_mask = (np.squeeze(mask) > 0).astype(np.uint8) * 255
            items.append({"source": "point", "prompt": point, "mask": binary_mask, "score": float(scores[idx])})

    return items


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    """Compute an xyxy bounding box from a binary mask."""

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def draw_mask_like_modelscope(image: np.ndarray, mask: np.ndarray, color=(30, 180, 255), alpha=0.5) -> np.ndarray:
    """Draw a semi-transparent mask with a white contour."""

    vis = image.copy()
    color_layer = np.zeros_like(vis)
    color_layer[mask > 0] = color
    vis = cv2.addWeighted(color_layer, alpha, vis, 1 - alpha, 0)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
    return vis


def save_outputs(image_np: np.ndarray, output_dir: Path, items: list[dict[str, Any]]) -> None:
    """Save mask images, visualization images, and segments.json."""

    records = []
    for idx, item in enumerate(items):
        mask = item["mask"]
        bbox = mask_to_bbox(mask)

        cv2.imwrite(str(output_dir / f"mask_{idx:03d}.png"), mask)

        vis = draw_mask_like_modelscope(image_np, mask)
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (30, 255, 30), 2)
        score = item.get("score")
        if score is not None:
            cv2.putText(
                vis,
                f"score={score:.3f}",
                (bbox[0], max(20, bbox[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imwrite(str(output_dir / f"visual_{idx:03d}.jpg"), vis)

        record = {"id": idx, "source": item["source"], "prompt": item["prompt"], "bbox": bbox}
        if score is not None:
            record["score"] = score
        records.append(record)

    (output_dir / "segments.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(records)} masks to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM segmentation with point/box prompts.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    config_dir = config_path.parent

    output_dir = resolve_path(cfg["output_dir"], config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(resolve_path(cfg["image_path"], config_dir)).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    items = collect_predictions(image, cfg)
    save_outputs(image_np, output_dir, items)


if __name__ == "__main__":
    main()
