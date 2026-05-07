# 实操 7：SAM/SAM3 分割模块微调与效果对比

课时：1 小时

本节默认模型、标注、图片和环境已经就绪。学习重点是理解分割微调的目的，并学会从 mask 边界、完整度、小目标和遮挡目标几个角度评价效果。

## 学习目标

- 理解 mask / polygon 数据在分割微调中的作用。
- 理解微调前后对比为什么重要。
- 学会查看 `mask_before.png`、`mask_after.png` 和 `mask_diff.png`。
- 学会查看导出的 COCO 格式分割结果。

## 背景知识

通用分割模型在很多图片上表现不错，但遇到特殊行业场景时可能边界不准、小目标漏分或遮挡目标分不完整。分割微调的目标是让模型更适应当前业务图像。

评价分割效果时，可以先看四个方面：

- 边界是否贴合。
- 目标是否完整。
- 小目标是否漏掉。
- 遮挡区域是否仍能分出主体。

## 课时安排

| 时间 | 内容 |
| --- | --- |
| 0-10 分钟 | 讲解 mask、polygon 和分割微调 |
| 10-25 分钟 | 讲解微调前后对比指标 |
| 25-40 分钟 | 运行脚本生成 before、after、diff |
| 40-52 分钟 | 人工检查分割质量 |
| 52-60 分钟 | 查看 COCO 导出并讨论数据质量提升 |

## 文件认知

- `config.yaml`：保存模型路径、标注路径、验证图片和训练参数。
- `sam_finetune_compare.py`：分割微调占位、效果对比和 COCO 导出。
- `outputs/mask_before.png`：微调前 mask。
- `outputs/mask_after.png`：微调后 mask。
- `outputs/mask_diff.png`：前后差异。
- `outputs/sam_result_coco.json`：导出的 COCO 结果。
- `todo.md`：课前准备事项。

## 实验详细步骤

1. 进入目录：

```bash
cd practice_07_sam_finetune_compare
```

2. 阅读配置：

```bash
sed -n '1,160p' config.yaml
```

3. 运行对比脚本：

```bash
python sam_finetune_compare.py --config config.yaml
```

4. 查看输出：

```bash
ls outputs
cat outputs/compare_summary.json
cat outputs/sam_result_coco.json
```

5. 对比三张 mask：

- `mask_before.png`
- `mask_after.png`
- `mask_diff.png`

6. 记录观察结果。

## 观察记录

| 指标 | 微调前 | 微调后 | 结论 |
| --- | --- | --- | --- |
| 边界贴合度 |  |  |  |
| 目标完整度 |  |  |  |
| 小目标 |  |  |  |
| 遮挡目标 |  |  |  |

## 课后练习

- 换一张小目标图片进行对比。
- 换一张遮挡图片进行对比。
- 思考分割结果如何提升检测数据标注质量。
