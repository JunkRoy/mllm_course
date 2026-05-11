# 实操 2：SAM2 与 SAM3 目标分割对比

本章节学习如何使用 SAM 系列模型做“提示分割”。为了避免 SAM2 和 SAM3 的接口混在一起，本节提供两个独立脚本：

- `segment_with_sam2.py`：专门用于 SAM2，支持点提示和框提示。
- `segment_with_sam3.py`：专门用于 SAM3，使用新版 `input_boxes` / `input_boxes_labels` 接口。

两个脚本会把结果保存到不同目录，方便你对比。

## 1. 本节目标

完成本节后，你应该能理解：

- SAM2 和 SAM3 的调用接口为什么不同。
- 点提示、框提示、mask、bbox、score 分别是什么。
- 如何通过 `config.yaml` 修改图片、模型路径和提示位置。
- 如何查看分割结果图片和 `segments.json`。

## 2. 目录结构

从项目根目录查看：

```bash
ls practice_02_sam_segmentation
```

主要文件：

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | 保存 SAM2/SAM3 模型名、图片路径、点提示、框提示和输出目录 |
| `segment_with_sam2.py` | 使用 SAM2 推理点提示和框提示 |
| `segment_with_sam3.py` | 使用 SAM3 推理框提示，并把点提示转换为小框 |
| `segment_with_prompts.py` | 兼容旧入口，建议课堂上优先使用上面两个独立脚本 |
| `requirements.txt` | 本节依赖 |
| `outputs/sam2` | SAM2 输出结果 |
| `outputs/sam3` | SAM3 输出结果 |

## 3. 安装依赖

在 Ubuntu 环境中执行：

```bash
pip install -r practice_02_sam_segmentation/requirements.txt
```

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果服务器不能联网，请提前下载模型，并把 `config.yaml` 中的模型名改成本地路径。

## 4. 配置文件说明

查看配置：

```bash
sed -n '1,240p' practice_02_sam_segmentation/config.yaml
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `sam2_model_name` | SAM2 模型 ID 或本地模型目录 |
| `sam3_model_name` | SAM3 模型 ID 或本地模型目录 |
| `device` | 推理设备，通常是 `cuda` 或 `cpu` |
| `image_path` | 输入图片路径，相对 `config.yaml` 所在目录解析 |
| `output_dir` | 输出根目录 |
| `confidence_threshold` | 将概率 mask 转成二值 mask 的阈值 |
| `point_prompts` | 点提示，格式为 `[x, y, label]` |
| `box_prompts` | 框提示，格式为 `[x1, y1, x2, y2]` |
| `text_prompts` | 教学保留字段，当前脚本只打印提示 |

点提示中的 `label`：

- `1` 表示前景点。
- `0` 表示背景点。

注意：SAM3 当前 transformers 接口更偏向 box/concept prompt。本课程脚本会把 `point_prompts` 自动转成围绕该点的小框，便于和 SAM2 输出做课堂对比。

## 5. 运行 SAM2 推理

从项目根目录执行：

```bash
python practice_02_sam_segmentation/segment_with_sam2.py \
  --config practice_02_sam_segmentation/config.yaml
```

输出目录：

```text
practice_02_sam_segmentation/outputs/sam2
```

查看输出：

```bash
ls practice_02_sam_segmentation/outputs/sam2
cat practice_02_sam_segmentation/outputs/sam2/segments.json
```

SAM2 会分别使用：

- `box_prompts`
- `point_prompts`

因此它适合演示“点一下目标”和“框住目标”两种分割方式。

## 6. 运行 SAM3 推理

从项目根目录执行：

```bash
python practice_02_sam_segmentation/segment_with_sam3.py \
  --config practice_02_sam_segmentation/config.yaml
```

输出目录：

```text
practice_02_sam_segmentation/outputs/sam3
```

查看输出：

```bash
ls practice_02_sam_segmentation/outputs/sam3
cat practice_02_sam_segmentation/outputs/sam3/segments.json
```

SAM3 脚本使用新版接口：

```python
processor(
    images=image,
    input_boxes=[[box]],
    input_boxes_labels=[[1]],
    return_tensors="pt",
)
```

这也是之前 `input_boxes` 报错的原因：SAM3 不能再用旧的 `processor(image, input_boxes=...)` 位置参数写法。

## 7. 对比 SAM2 和 SAM3 输出

分别运行两个脚本后，查看：

```bash
ls practice_02_sam_segmentation/outputs/sam2
ls practice_02_sam_segmentation/outputs/sam3
```

重点对比：

- `visual_*.jpg` 中 mask 是否覆盖目标。
- `segments.json` 中 bbox 是否合理。
- `score` 是否和肉眼质量一致。
- SAM2 的点提示结果是否比 SAM3 的点转小框结果更自然。

## 8. 输出文件说明

两个脚本都会输出：

| 文件 | 含义 |
| --- | --- |
| `mask_000.png` | 二值 mask |
| `visual_000.jpg` | 原图叠加 mask、bbox 和 score |
| `segments.json` | 结构化结果 |

`segments.json` 中常见字段：

- `model`：`sam2` 或 `sam3`。
- `source`：`box`、`point` 或 `point_as_box`。
- `prompt`：原始提示。
- `bbox`：由 mask 计算出的外接矩形。
- `score`：模型分数。

## 9. 推荐实验

实验 1：只跑框提示。

把配置改成：

```yaml
point_prompts: []
```

然后分别运行 SAM2 和 SAM3，对比输出。

实验 2：只跑点提示。

把配置改成：

```yaml
box_prompts: []
```

运行 SAM2 和 SAM3。注意 SAM3 会把点转换成小框，所以两者结果可能不同。

实验 3：调整阈值。

把：

```yaml
confidence_threshold: 0.5
```

分别改成 `0.3` 和 `0.7`，观察 mask 面积变化。

## 10. 常见问题

问题：SAM2 提示 `Sam2Model` 找不到。

解决：升级 transformers，或确认当前环境安装的是支持 SAM2 的版本。

问题：SAM3 报 `unexpected keyword argument 'input_boxes'`。

解决：请使用 `segment_with_sam3.py`。该脚本使用 `processor(images=..., input_boxes=..., input_boxes_labels=...)`，不要用旧写法。

问题：服务器无法下载模型。

解决：提前下载模型到服务器，然后修改：

```yaml
sam2_model_name: /path/to/local/sam2
sam3_model_name: /path/to/local/sam3
```

问题：输出多个 mask。

说明：SAM 可能返回多个候选结果，脚本会全部保存，方便比较。

## 11. 学习任务

请完成下面练习：

1. 分别运行 SAM2 和 SAM3。
2. 各选择一张 `visual_*.jpg`，说明哪个结果更贴合目标。
3. 修改一个框提示，观察 bbox 和 score 的变化。
4. 修改一个点提示，比较 SAM2 和 SAM3 的差异。
