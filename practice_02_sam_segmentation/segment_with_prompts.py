"""基于 transformers 的 SAM3 提示分割示例。

教学流程：
1）读取配置
2）使用点/框提示执行分割
3）保存 mask/可视化/JSON 结果
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
    """按需导入 torch 与 transformers 组件，缺依赖时给出明确报错。"""
    try:
        import torch
        from transformers import AutoModelForMaskGeneration, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "缺少依赖：请先安装 PyTorch 和 transformers。"
        ) from exc
    return torch, AutoModelForMaskGeneration, AutoProcessor


def load_config(path: str) -> dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_name(cfg: dict[str, Any]) -> str:
    """从配置中获取模型 ID，默认使用 ModelScope/HF 可用的 `facebook/sam3`。"""
    return str(cfg.get("sam_model_name") or "facebook/sam3")


def build_model_and_processor(cfg: dict[str, Any]):
    """创建模型、处理器与设备信息。

    返回：
        torch 模块、模型实例、处理器实例、设备字符串
    """
    torch, AutoModelForMaskGeneration, AutoProcessor = require_sam_transformers()
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = resolve_model_name(cfg)
    model = AutoModelForMaskGeneration.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return torch, model, processor, device


def postprocess_masks(processor, outputs, inputs) -> tuple[np.ndarray, np.ndarray]:
    """将 SAM 原始输出还原为原图尺寸的 mask，并返回对应 IoU 分数。"""
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
        reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores[0].detach().cpu().numpy()
    return masks.detach().cpu().numpy(), scores


def predict_with_box_prompt(image: Image.Image, box: list[float], model, processor, torch, device: str):
    """对单个框提示执行 SAM 预测。"""
    inputs = processor(image, input_boxes=[[[float(v) for v in box]]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return postprocess_masks(processor, outputs, inputs)


def predict_with_point_prompt(image: Image.Image, point: list[float], model, processor, torch, device: str):
    """对单个点提示执行 SAM 预测。

    点格式：[x, y, label]，其中 label=1 表示前景，0 表示背景。
    """
    label = int(point[2]) if len(point) > 2 else 1
    inputs = processor(
        image,
        input_points=[[[[float(point[0]), float(point[1])]]]],
        input_labels=[[[label]]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return postprocess_masks(processor, outputs, inputs)


def collect_predictions(image: Image.Image, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """收集所有提示的预测结果，统一整理为列表。"""
    torch, model, processor, device = build_model_and_processor(cfg)
    threshold = float(cfg.get("confidence_threshold", 0.5))
    items: list[dict[str, Any]] = []

    # 文本提示当前不参与推理：标准 SAM 接口主要面向点/框几何提示。
    for prompt in cfg.get("text_prompts", []):
        print(f"warning: 文本提示 '{prompt}' 当前不支持，将跳过")

    for box in cfg.get("box_prompts", []):
        masks, scores = predict_with_box_prompt(image, box, model, processor, torch, device)
        for idx, mask in enumerate(masks):
            binary_mask = (np.squeeze(mask) > threshold).astype(np.uint8) * 255
            items.append({"source": "box", "prompt": box, "mask": binary_mask, "score": float(scores[idx])})

    for point in cfg.get("point_prompts", []):
        masks, scores = predict_with_point_prompt(image, point, model, processor, torch, device)
        for idx, mask in enumerate(masks):
            binary_mask = (np.squeeze(mask) > threshold).astype(np.uint8) * 255
            items.append({"source": "point", "prompt": point, "mask": binary_mask, "score": float(scores[idx])})

    return items


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    """根据二值 mask 计算外接框 bbox=[x_min,y_min,x_max,y_max]。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def draw_mask_like_modelscope(image: np.ndarray, mask: np.ndarray, color=(30, 180, 255), alpha=0.5) -> np.ndarray:
    """以半透明填充+白色轮廓的方式可视化 mask（接近 ModelScope 风格）。"""
    vis = image.copy()
    color_layer = np.zeros_like(vis)
    color_layer[mask > 0] = color
    vis = cv2.addWeighted(color_layer, alpha, vis, 1 - alpha, 0)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
    return vis


def save_outputs(image_np: np.ndarray, output_dir: Path, items: list[dict[str, Any]]) -> None:
    """保存 mask 图片、可视化图片与 segments.json。"""
    records = []
    for idx, item in enumerate(items):
        mask = item["mask"]
        bbox = mask_to_bbox(mask)

        cv2.imwrite(str(output_dir / f"mask_{idx:03d}.png"), mask)

        vis = draw_mask_like_modelscope(image_np, mask)
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (30, 255, 30), 2)
        score = item.get("score")
        if score is not None:
            cv2.putText(vis, f"score={score:.3f}", (bbox[0], max(20, bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / f"visual_{idx:03d}.jpg"), vis)

        record = {"id": idx, "source": item["source"], "prompt": item["prompt"], "bbox": bbox}
        if score is not None:
            record["score"] = score
        records.append(record)

    (output_dir / "segments.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(records)} masks to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(cfg["image_path"]).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    items = collect_predictions(image, cfg)
    save_outputs(image_np, output_dir, items)


if __name__ == "__main__":
    main()
