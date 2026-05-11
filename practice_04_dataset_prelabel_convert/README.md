# 实操 4：检测数据集整理、可视化检查与 Qwen-VL 训练数据转换

本章节面向刚接触多模态训练数据处理的同学。你会把传统目标检测数据集整理成统一格式，再转换为 Qwen-VL 指令微调常用的 JSONL 数据。

本节默认约定：

- `dataset` 目录与 `practice_04_dataset_prelabel_convert` 目录位于同一级。
- 所有命令在 Ubuntu 环境中执行。
- 所有路径都使用相对路径，方便未来迁移到服务器。

推荐项目结构：

```text
mllm_course
├─ dataset
│  ├─ Safety Helmet Detection_datasets_
│  ├─ VOC2028
│  └─ merged_helmet_voc
└─ practice_04_dataset_prelabel_convert
   ├─ merge_helmet_datasets.py
   ├─ visualize_voc_boxes.py
   └─ voc_to_qwen_vl_jsonl.py
```

## 1. 本节目标

完成本节后，你应该能做到：

- 理解图片和 Pascal VOC XML 标注文件的一一对应关系。
- 将 `head`、`helmet`、`hat` 等类别名统一成训练需要的类别。
- 合并两个数据集并去除近似重复图片。
- 把 XML 中的类别和 bbox 画到图片上，人工检查标注质量。
- 把训练集转换成 Qwen-VL 微调常见的 messages JSONL 格式。

## 2. 当前目录文件

从项目根目录执行：

```bash
ls practice_04_dataset_prelabel_convert
```

主要文件：

| 文件 | 作用 |
| --- | --- |
| `merge_helmet_datasets.py` | 合并两个 VOC 数据集，统一类别，去除近似重复图片 |
| `visualize_voc_boxes.py` | 把 XML 标签和检测框画到图片上 |
| `voc_to_qwen_vl_jsonl.py` | 把 VOC 训练集转换为 Qwen-VL JSONL |
| `requirements.txt` | 本节依赖 |
| `build_and_convert_dataset.py` | 课程通用格式转换示例脚本 |
| `config.yaml` | 通用格式转换示例配置 |

## 3. 原始数据目录要求

请确认项目根目录下存在 `dataset`：

```bash
ls dataset
```

其中应包含：

```text
dataset
├─ Safety Helmet Detection_datasets_
│  ├─ images
│  └─ annotations
└─ VOC2028
   ├─ JPEGImages
   ├─ Annotations
   └─ ImageSets
```

两个数据集的类别命名不同：

- `Safety Helmet Detection_datasets_` 中常见类别：`head`、`helmet`。
- `VOC2028` 中常见类别：`hat`。
- 本节会把 `hat` 统一映射为 `helmet`。

最终只保留：

```text
head
helmet
```

## 4. 安装依赖

```bash
pip install -r practice_04_dataset_prelabel_convert/requirements.txt
```

检查依赖：

```bash
python -c "import PIL; import tqdm; print('依赖检查通过')"
```

## 5. 第一步：合并数据集

执行：

```bash
python practice_04_dataset_prelabel_convert/merge_helmet_datasets.py --overwrite
```

脚本默认读取：

```text
dataset/Safety Helmet Detection_datasets_
dataset/VOC2028
```

默认输出：

```text
dataset/merged_helmet_voc
```

输出结构：

```text
dataset/merged_helmet_voc
├─ Annotations
├─ JPEGImages
├─ ImageSets
│  └─ Main
│     ├─ train.txt
│     ├─ val.txt
│     ├─ test.txt
│     └─ trainval.txt
├─ classes.txt
└─ summary.json
```

## 6. 合并脚本参数

指定数据根目录：

```bash
python practice_04_dataset_prelabel_convert/merge_helmet_datasets.py \
  --dataset-root dataset \
  --overwrite
```

指定输出目录：

```bash
python practice_04_dataset_prelabel_convert/merge_helmet_datasets.py \
  --output-dir dataset/merged_helmet_voc_v2 \
  --overwrite
```

关闭近似重复图片去重：

```bash
python practice_04_dataset_prelabel_convert/merge_helmet_datasets.py \
  --dedupe-threshold -1 \
  --overwrite
```

调整去重阈值：

```bash
python practice_04_dataset_prelabel_convert/merge_helmet_datasets.py \
  --dedupe-threshold 6 \
  --overwrite
```

说明：

- `--dedupe-threshold` 越大，去重越激进。
- 默认值是 `4`，适合先做温和去重。
- 如果担心误删，可以先用 `-1` 关闭去重。

## 7. 检查合并结果

查看输出目录：

```bash
ls dataset/merged_helmet_voc
```

查看类别：

```bash
cat dataset/merged_helmet_voc/classes.txt
```

正常输出：

```text
head
helmet
```

查看处理摘要：

```bash
cat dataset/merged_helmet_voc/summary.json
```

重点关注：

- `input_records`：合并前读到的有效样本数。
- `output_records`：合并和去重后保留的样本数。
- `duplicates_removed`：去掉的近似重复图片数。
- `output_stats`：每个类别的目标框数量。

## 8. 第二步：可视化检查检测框

先可视化 100 张图片：

```bash
python practice_04_dataset_prelabel_convert/visualize_voc_boxes.py --max-images 100
```

默认输出：

```text
dataset/merged_helmet_voc/visualizations
```

查看输出文件：

```bash
ls dataset/merged_helmet_voc/visualizations | head
```

如果前 100 张看起来正常，再可视化全部：

```bash
python practice_04_dataset_prelabel_convert/visualize_voc_boxes.py
```

人工检查时重点看：

- `helmet` 是否框住安全帽。
- `head` 是否框住未戴安全帽的头部。
- bbox 是否明显偏移。
- 是否有类别明显标错。

## 9. 第三步：转换为 Qwen-VL JSONL

生成训练集：

```bash
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py --split train
```

生成验证集：

```bash
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py --split val
```

默认输出：

```text
dataset/merged_helmet_voc/qwen_vl_train.jsonl
dataset/merged_helmet_voc/qwen_vl_val.jsonl
```

查看前两行：

```bash
head -n 2 dataset/merged_helmet_voc/qwen_vl_train.jsonl
```

## 10. Qwen-VL JSONL 格式说明

每一行是一条训练样本，结构类似：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "JPEGImages/000001.jpg"},
        {"type": "text", "text": "请检测图片中的安全帽和未佩戴安全帽的头部。请只输出 JSON 数组，每个元素包含 label 和 bbox，bbox 格式为 [xmin, ymin, xmax, ymax]。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "[{\"label\": \"helmet\", \"bbox\": [10, 20, 80, 100]}]"}
      ]
    }
  ]
}
```

字段含义：

- `user`：输入图片和任务指令。
- `assistant`：期望模型学习输出的答案。
- `label`：目标类别。
- `bbox`：像素坐标 `[xmin, ymin, xmax, ymax]`。

## 11. 自定义输出路径和提示词

指定 JSONL 输出位置：

```bash
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py \
  --split train \
  --jsonl-path dataset/merged_helmet_voc/qwen_train.jsonl
```

自定义 prompt：

```bash
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py \
  --split train \
  --prompt "请找出图片中的 helmet 和 head，输出 JSON 数组，字段包括 label 和 bbox。"
```

如果训练框架要求绝对图片路径，可以加：

```bash
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py \
  --split train \
  --absolute-image-path
```

一般建议优先使用相对路径，方便数据迁移。

## 12. 推荐完整流程

第一次操作建议按下面顺序执行：

```bash
python -c "import PIL; import tqdm; print('依赖检查通过')"
python practice_04_dataset_prelabel_convert/merge_helmet_datasets.py --overwrite
python practice_04_dataset_prelabel_convert/visualize_voc_boxes.py --max-images 100
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py --split train
python practice_04_dataset_prelabel_convert/voc_to_qwen_vl_jsonl.py --split val
```

## 13. 常见问题

问题：找不到 `dataset`。

解决：确认 `dataset` 与 `practice_04_dataset_prelabel_convert` 在同一级目录。

问题：找不到 `train.txt`。

解决：先运行合并脚本，生成 `ImageSets/Main/train.txt`。

问题：可视化图片没有框。

解决：检查对应 XML 是否有有效 `object` 和 `bndbox`：

```bash
sed -n '1,120p' dataset/merged_helmet_voc/Annotations/000000.xml
```

问题：去重后图片少了很多。

解决：降低阈值或关闭去重：

```bash
python practice_04_dataset_prelabel_convert/merge_helmet_datasets.py \
  --dedupe-threshold 2 \
  --overwrite
```

## 14. 学习任务

请完成下面练习：

1. 解释为什么要把 `hat` 映射为 `helmet`。
2. 查看 `summary.json`，记录最终保留了多少张图片。
3. 打开 10 张可视化图片，判断 bbox 是否合理。
4. 查看 `qwen_vl_train.jsonl` 的前两行，说明 `user` 和 `assistant` 的作用。

## 15. 最终产物检查清单

完成本节后，请确认这些文件或目录存在：

```text
dataset/merged_helmet_voc/Annotations
dataset/merged_helmet_voc/JPEGImages
dataset/merged_helmet_voc/ImageSets/Main/train.txt
dataset/merged_helmet_voc/classes.txt
dataset/merged_helmet_voc/summary.json
dataset/merged_helmet_voc/visualizations
dataset/merged_helmet_voc/qwen_vl_train.jsonl
```

看到这些产物，说明你已经完成了从传统检测数据集到多模态大模型训练数据的完整转换流程。
