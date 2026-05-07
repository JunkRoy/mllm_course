# 实操 4 准备事项

## 模型

- 可选 SAM/SAM3：`../models/sam`。
- 可选 Grounding DINO：`../models/grounding-dino`。

## 数据

- 原始图片目录：`../data/raw/images`。
- 原始标注文件：`../data/raw/annotations.json`。

## 环境依赖

- Python 运行环境。
- PyYAML。
- Python 包见 `requirements.txt`。

## 课前检查

```bash
ls ../data/raw/images
cat ../data/raw/annotations.json
sed -n '1,120p' requirements.txt
```

