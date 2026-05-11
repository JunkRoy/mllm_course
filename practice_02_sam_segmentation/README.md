# 实操 2：SAM 目标分割与可视化检查

本章节学习如何使用 SAM 类模型完成“提示分割”。你会通过点提示或框提示告诉模型要分割哪里，脚本会输出 mask、bbox、score 和可视化图片。

## 1. 本节目标

完成本节后，你应该能理解：

- 点提示和框提示分别是什么。
- mask、bbox、score 分别代表什么。
- 如何通过 `config.yaml` 调整输入图片和提示位置。
- 如何查看分割结果图片和结构化 JSON。

## 2. 目录结构

从项目根目录查看本节文件：

```bash
ls practice_02_sam_segmentation
```

主要文件：

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | 保存模型名称、输入图片、点提示、框提示和输出目录 |
| `segment_with_prompts.py` | SAM 提示分割脚本 |
| `requirements.txt` | 本节依赖 |
| `outputs/mask_*.png` | 输出的二值 mask |
| `outputs/visual_*.jpg` | 叠加 mask、bbox 和 score 的可视化图片 |
| `outputs/segments.json` | 分割结果的结构化记录 |

## 3. 安装依赖

在 Ubuntu 环境中执行：

```bash
pip install -r practice_02_sam_segmentation/requirements.txt
```

如果模型需要 GPU，先检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. 查看配置文件

```bash
sed -n '1,220p' practice_02_sam_segmentation/config.yaml
```

关键字段说明：

| 字段 | 含义 |
| --- | --- |
| `sam_model_name` | SAM 模型 ID 或本地路径 |
| `confidence_threshold` | 将概率 mask 转成二值 mask 的阈值 |
| `device` | 推理设备，通常为 `cuda` 或 `cpu` |
| `image_path` | 输入图片路径 |
| `output_dir` | 输出目录 |
| `point_prompts` | 点提示，格式为 `[x, y, label]` |
| `box_prompts` | 框提示，格式为 `[x1, y1, x2, y2]` |
| `text_prompts` | 教学保留字段，当前脚本会跳过 |

点提示中的 `label` 含义：

- `1` 表示前景点，也就是“我要分割这里”。
- `0` 表示背景点，也就是“不要分割这里”。

## 5. 运行分割脚本

从项目根目录执行：

```bash
python practice_02_sam_segmentation/segment_with_prompts.py \
  --config practice_02_sam_segmentation/config.yaml
```

也可以进入本节目录：

```bash
cd practice_02_sam_segmentation
python segment_with_prompts.py --config config.yaml
```

## 6. 查看输出

查看输出目录：

```bash
ls practice_02_sam_segmentation/outputs
```

查看结构化结果：

```bash
cat practice_02_sam_segmentation/outputs/segments.json
```

重点字段：

- `id`：候选结果编号。
- `source`：来自点提示还是框提示。
- `prompt`：原始提示信息。
- `bbox`：由 mask 计算出的外接矩形。
- `score`：模型对该候选 mask 的置信分数。

## 7. 人工检查可视化结果

打开 `visual_*.jpg`，观察：

- 彩色区域是否覆盖了你想分割的目标。
- 白色轮廓是否贴合目标边界。
- 绿色 bbox 是否包住目标。
- score 高的结果是否真的更好。

如果结果不好，优先调整：

- `box_prompts` 的框是否太大或太小。
- `point_prompts` 是否点在目标中心。
- `confidence_threshold` 是否过高或过低。

## 8. 推荐实验

实验 1：只使用框提示。

把 `point_prompts` 改为空列表：

```yaml
point_prompts: []
```

重新运行脚本，观察输出变化。

实验 2：只使用点提示。

把 `box_prompts` 改为空列表：

```yaml
box_prompts: []
```

重新运行脚本，比较 mask 是否稳定。

实验 3：调整阈值。

把 `confidence_threshold` 从 `0.5` 分别改成 `0.3` 和 `0.7`，观察 mask 面积变化。

## 9. 常见问题

问题：提示下载模型失败。

解决：服务器不能联网时，请提前把模型下载到本地，并把 `sam_model_name` 改成本地模型路径。

问题：输出了多个 mask。

说明：SAM 可能返回多个候选分割结果，脚本会全部保存，方便你比较。

问题：text prompt 为什么没有参与推理？

说明：当前脚本演示的是 SAM 的几何提示接口，主要支持点和框。`text_prompts` 只是为了课堂扩展保留。

## 10. 学习任务

请完成下面练习：

1. 修改点提示坐标，让点落在目标边缘，观察 score 变化。
2. 修改框提示大小，观察 mask 是否更完整。
3. 找出你认为最好的 `visual_*.jpg`，并在 `segments.json` 中找到对应记录。
