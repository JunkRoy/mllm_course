import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


def require_sam3():
    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 is not installed in this Python environment. Install Meta's "
            "facebookresearch/sam3 package and its PyTorch/CUDA dependencies first."
        ) from exc
    return torch, build_sam3_image_model, Sam3Processor


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 180, 255), alpha=0.45) -> np.ndarray:
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def xyxy_to_normalized_cxcywh(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        max(0.0, (x2 - x1) / width),
        max(0.0, (y2 - y1) / height),
    ]


def point_to_normalized_xy(point: list[float], width: int, height: int) -> list[float]:
    x, y = [float(v) for v in point[:2]]
    return [x / width, y / height]


def resolve_checkpoint_path(cfg: dict) -> str | None:
    configured = cfg.get("sam_checkpoint_path")
    if not configured:
        return None

    path = Path(configured)
    if path.is_file():
        return str(path)
    return str(path)


def state_to_items(output: dict, source: str, prompt) -> list[dict]:
    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")
    if masks is None:
        return []

    masks_np = masks.detach().cpu().numpy()
    boxes_np = None if boxes is None else boxes.detach().cpu().numpy()
    scores_np = None if scores is None else scores.detach().cpu().numpy()

    items = []
    for idx, mask in enumerate(masks_np):
        mask_2d = np.squeeze(mask).astype(bool)
        item = {
            "source": source,
            "prompt": prompt,
            "mask": (mask_2d.astype(np.uint8) * 255),
        }
        if boxes_np is not None and idx < len(boxes_np):
            item["model_box"] = [int(v) for v in boxes_np[idx].tolist()]
        if scores_np is not None and idx < len(scores_np):
            item["score"] = float(scores_np[idx])
        items.append(item)
    return items


def add_point_prompt(processor, model, torch, point: list[int], state: dict, device: str) -> dict:
    if "language_features" not in state["backbone_out"]:
        dummy_text_outputs = model.backbone.forward_text(["visual"], device=device)
        state["backbone_out"].update(dummy_text_outputs)
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = model._get_dummy_prompt()

    width = state["original_width"]
    height = state["original_height"]
    xy = point_to_normalized_xy(point, width, height)
    label = bool(int(point[2])) if len(point) > 2 else True
    points = torch.tensor(xy, device=device, dtype=torch.float32).view(1, 1, 2)
    labels = torch.tensor([label], device=device, dtype=torch.long).view(1, 1)
    state["geometric_prompt"].append_points(points, labels)
    return processor._forward_grounding(state)


def run_sam3(image: Image.Image, cfg: dict) -> list[dict]:
    torch, build_sam3_image_model, Sam3Processor = require_sam3()
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(cfg)
    load_from_hf = bool(cfg.get("load_from_hf", checkpoint_path is None))

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=float(cfg.get("confidence_threshold", 0.5)),
    )

    width, height = image.size
    items = []

    for prompt in cfg.get("text_prompts", []):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=str(prompt))
        items.extend(state_to_items(output, "text", prompt))

    for box in cfg.get("box_prompts", []):
        state = processor.set_image(image)
        normalized_box = xyxy_to_normalized_cxcywh(box, width, height)
        output = processor.add_geometric_prompt(
            box=normalized_box,
            label=True,
            state=state,
        )
        items.extend(state_to_items(output, "box", box))

    for point in cfg.get("point_prompts", []):
        state = processor.set_image(image)
        output = add_point_prompt(processor, model, torch, point, state, device)
        items.extend(state_to_items(output, "point", point))

    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(cfg["image_path"]).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    records = []
    for idx, item in enumerate(run_sam3(image, cfg)):
        mask = item["mask"]
        bbox = mask_to_bbox(mask)
        cv2.imwrite(str(output_dir / f"mask_{idx:03d}.png"), mask)
        vis = overlay_mask(image_np, mask)
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (30, 255, 30), 2)
        cv2.imwrite(str(output_dir / f"visual_{idx:03d}.jpg"), vis)
        record = {"id": idx, "source": item["source"], "prompt": item["prompt"], "bbox": bbox}
        if "model_box" in item:
            record["model_box"] = item["model_box"]
        if "score" in item:
            record["score"] = item["score"]
        records.append(record)

    (output_dir / "segments.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(records)} masks to {output_dir}")


if __name__ == "__main__":
    main()
