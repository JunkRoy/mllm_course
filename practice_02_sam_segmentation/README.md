# 实操 2：SAM3 目标分割（Transformers 版）

课时：1 小时

本节使用 `transformers` 加载 `facebook/sam3`，重点理解「提示 -> mask -> bbox -> 可视化」完整流程。

## 学习目标

- 理解点提示、框提示在分割任务中的作用。
- 理解 `mask / bbox / score / visual` 四类输出。
- 会修改 `config.yaml` 并观察分割结果变化。
- 会读取 `segments.json` 做结果分析。

## 文件说明

- `config.yaml`：模型 ID、阈值、输入输出路径、点/框提示配置。
- `segment_with_prompts.py`：教学版脚本（已拆分为“加载模型、预测、可视化、保存”四个阶段）。
- `outputs/mask_*.png`：每个候选结果的二值掩码。
- `outputs/visual_*.jpg`：掩码叠加图（半透明填充 + 白色轮廓 + bbox + score）。
- `outputs/segments.json`：结构化结果（id/source/prompt/bbox/score）。
- `todo.md`：课前准备。

## 环境安装

```bash
cd practice_02_sam_segmentation
pip install -r requirements.txt
```

## 配置说明（`config.yaml`）

关键字段：

- `sam_model_name`：模型 ID，默认 `facebook/sam3`。
- `confidence_threshold`：掩码二值化阈值。
- `device`：`cuda` 或 `cpu`。
- `point_prompts`：格式 `[x, y, label]`。
- `box_prompts`：格式 `[x1, y1, x2, y2]`。
- `text_prompts`：教学保留字段，当前脚本会跳过并提示 warning。

## 运行步骤

1. 进入目录：

```bash
cd practice_02_sam_segmentation
```

2. 检查配置：

```bash
sed -n '1,220p' config.yaml
```

3. 执行分割：

```bash
python segment_with_prompts.py --config config.yaml
```

4. 查看输出：

```bash
ls outputs
cat outputs/segments.json
```

## 课堂建议实验

1. 固定 `point_prompts`，只改 `box_prompts`，观察 bbox 与 mask 稳定性。
2. 固定 `box_prompts`，改点位到目标边界和背景区域，观察 `score` 变化。
3. 对比 `mask_*.png` 与 `visual_*.jpg`，总结高质量分割的判断标准。

## 常见问题

- **Q: 为什么 text prompt 没有参与预测？**  
  A: 当前脚本使用的是 SAM 的几何提示接口（点/框）。`text_prompts` 字段保留用于课堂对比与后续扩展。

- **Q: 为什么会输出多个 mask？**  
  A: SAM 会返回多个候选分割，脚本会分别保存并附带 `score` 供筛选。
