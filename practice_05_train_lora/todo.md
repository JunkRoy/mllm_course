# 实操 5 准备事项

## 模型

- 基础视觉模型 backbone：`../models/vision-backbone`。
- 视觉语言模型：`../models/vlm`。

## 数据

- 基础视觉训练数据：`../data/vision_dataset`。
- VLM 微调数据：`../data/vlm_lora/train.jsonl`。

## 环境依赖

- Python 运行环境。
- PyTorch、TorchVision、Transformers、Datasets、PEFT、PyYAML。
- Python 包见 `requirements.txt`。

## 课前检查

```bash
ls ../models/vision-backbone
ls ../models/vlm
find ../data/vision_dataset -maxdepth 2 -type f | head
head ../data/vlm_lora/train.jsonl
```

