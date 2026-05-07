# 实操 4：视觉任务数据构造、自动预标注与格式转换

课时：1 小时

本节默认原始图片、标注文件和环境已经就绪。学习重点是理解视觉任务训练前的数据组织方式，并把原始标注转换成常见训练格式。

## 学习目标

- 理解训练集、验证集、测试集的作用。
- 认识分类、检测、分割、OCR、VQA 的数据差异。
- 理解预标注的意义和人工修正流程。
- 学会查看 COCO、YOLO、JSONL 三类输出格式。

## 背景知识

训练模型之前，数据必须整齐。图片、标注、类别名和数据划分都要统一，否则训练脚本无法正确读取。

常见格式：

- COCO：适合检测和分割，信息集中在一个 JSON 文件中。
- YOLO：适合检测，每张图片通常对应一个标签文本文件。
- JSONL：适合 VQA、OCR、指令微调，每行是一条样本。

预标注是让模型先给出初始框、mask 或文字区域，再由人工检查和修正。

## 课时安排

| 时间 | 内容 |
| --- | --- |
| 0-10 分钟 | 讲解训练集、验证集、测试集 |
| 10-20 分钟 | 对比分类、检测、分割、OCR、VQA 数据 |
| 20-35 分钟 | 阅读原始标注结构和类别配置 |
| 35-50 分钟 | 执行数据划分和格式转换 |
| 50-60 分钟 | 检查 COCO、YOLO、JSONL 输出 |

## 文件认知

- `config.yaml`：保存原始数据路径、输出目录、类别名和划分比例。
- `build_and_convert_dataset.py`：完成数据划分、预标注占位和格式转换。
- `outputs/dataset/annotations_coco.json`：COCO 格式结果。
- `outputs/dataset/labels_yolo/`：YOLO 标签目录。
- `outputs/dataset/samples.jsonl`：JSONL 样本。
- `todo.md`：课前准备事项。

## 实验详细步骤

1. 进入目录：

```bash
cd practice_04_dataset_prelabel_convert
```

2. 阅读配置：

```bash
sed -n '1,200p' config.yaml
```

3. 查看原始标注：

```bash
cat ../data/raw/annotations.json
```

4. 执行数据构造和格式转换：

```bash
python build_and_convert_dataset.py --config config.yaml
```

5. 查看输出文件：

```bash
find outputs/dataset -maxdepth 3 -type f
```

6. 查看 COCO：

```bash
cat outputs/dataset/annotations_coco.json
```

7. 查看 JSONL：

```bash
head outputs/dataset/samples.jsonl
```

8. 查看 YOLO 标签：

```bash
find outputs/dataset/labels_yolo -maxdepth 1 -type f
```

## 观察记录

| 格式 | 文件位置 | 主要字段 | 适合任务 |
| --- | --- | --- | --- |
| COCO |  |  |  |
| YOLO |  |  |  |
| JSONL |  |  |  |

## 课后练习

- 为样本增加 OCR 字段。
- 为样本增加 VQA 的 `question` 和 `answer`。
- 思考自动预标注后哪些内容必须人工检查。

