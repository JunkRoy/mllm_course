# 实操 4：检测数据集整理、可视化与 Qwen-VL JSONL 转换

本节把传统 VOC 检测数据集整理成 Qwen-VL 可以训练的 `messages` JSONL。

最终目标：

```text
/autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_train.jsonl
/autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_val.jsonl
/autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_images/
```

## 1. 本节文件

| 文件 | 作用 |
| --- | --- |
| `merge_helmet_datasets.py` | 合并安全帽 VOC 数据集，统一类别，划分 train/val/test |
| `visualize_voc_boxes.py` | 把 XML bbox 画到图片上，人工检查标注质量 |
| `voc_to_qwen_vl_jsonl.py` | 把 VOC XML 转成 Qwen-VL `messages` JSONL |
| `build_and_convert_dataset.py` | 通用格式转换示例 |
| `config.yaml` | 通用格式转换示例配置 |
| `requirements.txt` | 本节依赖 |

## 2. 安装依赖

```bash
pip install -r practice_04_dataset_prelabel_convert/requirements.txt
```

检查：

```bash
python -c "from PIL import Image; import tqdm; print('ok')"
```

## 3. 数据目录约定

服务器数据盘路径建议使用：

```text
/autodl-fs/data/dataset/
```

合并后数据集目录：

```text
/autodl-fs/data/dataset/merged_helmet_voc/
```

里面应该有：

```text
Annotations/
JPEGImages/
ImageSets/Main/train.txt
ImageSets/Main/val.txt
ImageSets/Main/test.txt
classes.txt
summary.json
```

## 4. 合并数据集

如果还没有 `merged_helmet_voc`，先运行合并：

```bash
python practice_04_dataset_prelabel_convert/merge_helmet_datasets.py \
  --dataset-root /autodl-fs/data/dataset \
  --overwrite
```

输出默认写到：

```text
/autodl-fs/data/dataset/merged_helmet_voc/
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--dataset-root` | 原始数据和输出数据所在根目录 |
| `--output-dir` | 自定义合并输出目录 |
| `--dedupe-threshold` | 图片去重阈值，`-1` 表示关闭去重 |
| `--overwrite` | 覆盖已有输出 |

## 5. 可视化检查

先看 100 张：

```bash
python practice_04_dataset_prelabel_convert/visualize_voc_boxes.py \
  --dataset-root /autodl-fs/data/dataset \
  --max-images 100
```

输出：

```text
/autodl-fs/data/dataset/merged_helmet_voc/visualization/
```

如果想可视化全部：

```bash
python practice_04_dataset_prelabel_convert/visualize_voc_boxes.py \
  --dataset-root /autodl-fs/data/dataset \
  --max-images 0
```

人工检查重点：

- `helmet` 是否框住安全帽。
- `head` 是否框住未戴安全帽的头。
- bbox 是否偏移。
- 类别是否反了。

## 6. 转换为 Qwen-VL JSONL

生成训练集，并把图片统一转成 RGB JPEG：

```bash
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py \
  --dataset-root /autodl-fs/data/dataset \
  --split train \
  --normalize-images
```

生成验证集：

```bash
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py \
  --dataset-root /autodl-fs/data/dataset \
  --split val \
  --normalize-images
```

生成测试集：

```bash
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py \
  --dataset-root /autodl-fs/data/dataset \
  --split test \
  --normalize-images
```

输出文件：

```text
/autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_train.jsonl
/autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_val.jsonl
/autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_test.jsonl
/autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_images/
/autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_train_summary.json
```

`--normalize-images` 很重要：它会把 BMP 等图片转成 RGB JPEG，避免 Qwen3-VL 训练时遇到不支持的图片格式。

## 7. 查看 JSONL

```bash
head -n 1 /autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_train.jsonl
```

每行结构类似：

```json
{"messages":[{"role":"user","content":[{"type":"image","image":"qwen_vl_images/xxx.jpg"},{"type":"text","text":"请检测图片中的安全帽和未佩戴安全帽的头部。请只输出 JSON 数组，每个元素包含 label 和 bbox，bbox 格式为 [xmin, ymin, xmax, ymax]。"}]},{"role":"assistant","content":[{"type":"text","text":"[{\"label\":\"helmet\",\"bbox\":[10,20,80,100]}]"}]}]}
```

含义：

| 字段 | 含义 |
| --- | --- |
| `user.content.image` | 输入图片，相对 `merged_helmet_voc` |
| `user.content.text` | 给模型的检测指令 |
| `assistant.content.text` | 标准答案，JSON 数组 |
| `bbox` | `[xmin, ymin, xmax, ymax]` 像素坐标 |

## 8. 完整推荐流程

```bash
python practice_04_dataset_prelabel_convert/merge_helmet_datasets.py \
  --dataset-root /autodl-fs/data/dataset \
  --overwrite

python practice_04_dataset_prelabel_convert/visualize_voc_boxes.py \
  --dataset-root /autodl-fs/data/dataset \
  --max-images 100

python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py \
  --dataset-root /autodl-fs/data/dataset \
  --split train \
  --normalize-images

python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py \
  --dataset-root /autodl-fs/data/dataset \
  --split val \
  --normalize-images
```

## 9. 常见问题

问题：找不到 `train.txt`。

解决：先运行 `merge_helmet_datasets.py`，生成 `ImageSets/Main/train.txt`。

问题：可视化没有框。

解决：检查 XML 中是否有 `object/bndbox`。

问题：转换时跳过很多样本。

解决：查看 `qwen_vl_train_summary.json`，里面有跳过原因，比如坏图、无有效框、类别不合法。

问题：训练时报图片格式不支持。

解决：重新转换 JSONL 时加 `--normalize-images`。
