# 实操 8：Visual Prompt Tuning 与多 GPU 训练

课时：1 小时

本节默认模型、数据、多 GPU 环境和依赖已经就绪。学习重点是理解 Visual Prompt Tuning 的训练思想，并观察单 GPU 与多 GPU 训练的差异。

## 学习目标

- 理解冻结视觉主干、只训练 prompt token 的思路。
- 学会查看 prompt token 的可训练参数量。
- 学会使用单 GPU、多 GPU两种方式启动训练。
- 学会比较训练速度、显存、global batch size 和 checkpoint。

## 背景知识

Visual Prompt Tuning 简称 VPT。它不大规模更新视觉模型，而是在输入侧增加少量可学习的 prompt token。这样训练更轻量，适合数据量较小或只想快速适配任务的场景。

多 GPU 训练不是简单地“更快”这么单一。它还会影响 global batch size、日志保存方式、checkpoint 写入位置和显存分布。

## 课时安排

| 时间 | 内容 |
| --- | --- |
| 0-10 分钟 | 讲解 VPT 和冻结视觉主干 |
| 10-25 分钟 | 阅读 prompt token 配置和参数量 |
| 25-38 分钟 | 单 GPU 运行训练 |
| 38-50 分钟 | 多 GPU 运行训练 |
| 50-60 分钟 | 对比 VPT、LoRA、组件微调 |

## 文件认知

- `config.yaml`：保存数据目录、prompt token 数、训练参数和输出目录。
- `visual_prompt_tuning.py`：VPT 训练脚本。
- `outputs/vpt_epoch_*.pt`：保存的 prompt token 权重。
- `outputs/vpt_report.json`：训练报告。
- `todo.md`：课前准备事项。

## 实验详细步骤

1. 进入目录：

```bash
cd practice_08_vpt_multi_gpu
```

2. 阅读配置：

```bash
sed -n '1,180p' config.yaml
```

3. 默认方式运行：

```bash
python visual_prompt_tuning.py --config config.yaml
```

4. 查看输出：

```bash
ls outputs
cat outputs/vpt_report.json
```

5. 使用 `torchrun` 启动多 GPU：

```bash
torchrun --nproc_per_node=2 visual_prompt_tuning.py --config config.yaml
```

6. 使用 Accelerate 启动：

```bash
accelerate launch visual_prompt_tuning.py --config config.yaml
```

7. 修改 `num_prompt_tokens` 为 `8`、`16`、`32`，观察参数量变化。

## 观察记录

| 配置 | GPU 数 | num_prompt_tokens | trainable_params | 速度 | 显存 |
| --- | --- | --- | --- | --- | --- |
| 默认 |  |  |  |  |  |
| torchrun |  |  |  |  |  |
| accelerate |  |  |  |  |  |

## 课后练习

- 对比 VPT 与 LoRA 的可训练参数量。
- 改变 batch size，观察显存变化。
- 思考什么时候适合使用 VPT。

