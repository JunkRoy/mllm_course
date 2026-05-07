# 实操 10 准备事项

## 模型

- VLM 服务模型：`../models/vlm`。
- 可选分割模型：`../models/sam`。
- 可选 OCR 模型：`../models/ocr`。
- 可选合并或量化模型：`../models/vlm-merged-int4`。

## 数据

- benchmark 样例图片：`../data/images/service_demo.jpg`。

## 环境依赖

- Python 运行环境。
- FastAPI、Uvicorn、PyTorch、Transformers、PEFT、Pillow、PyYAML。
- 可用 GPU 环境。
- Python 包见 `requirements.txt`。

## 课前检查

```bash
ls ../models/vlm
ls ../data/images/service_demo.jpg
sed -n '1,160p' requirements.txt
```

