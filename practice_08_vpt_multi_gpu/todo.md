# 实操 8 准备事项

## 模型

- 视觉模型或 VLM 视觉主干：`../models/vision-backbone`。

## 数据

- 图片分类数据：`../data/vision_dataset`。

## 环境依赖

- Python 运行环境。
- PyTorch、TorchVision、PyYAML。
- 可用多 GPU 环境。
- 可选 Accelerate。
- Python 包见 `requirements.txt`。

## 课前检查

```bash
ls ../models/vision-backbone
find ../data/vision_dataset -maxdepth 2 -type f | head
sed -n '1,120p' requirements.txt
```

