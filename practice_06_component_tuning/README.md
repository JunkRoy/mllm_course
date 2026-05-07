# 实操 6：组件选择式微调与训练参数对比

课时：1 小时

本节默认模型、数据和环境已经就绪。学习重点是理解视觉语言模型内部组件，并比较不同“解冻策略”的可训练参数量。

## 学习目标

- 理解 Vision Tower、Projector、LLM 三个组件的作用。
- 理解 Projector-only、Projector + LLM、Vision Tower 局部解冻。
- 学会查看总参数量、可训练参数量和可训练比例。
- 学会从参数量、显存、速度和效果角度比较微调策略。

## 背景知识

视觉语言模型通常由三部分构成：

- Vision Tower：看图，把图片转换成视觉特征。
- Projector：把视觉特征转换为语言模型能理解的表示。
- LLM：理解指令并生成回答。

训练时不一定要更新全部参数。只训练 Projector 最轻量，但能力有限；训练 Projector + LLM 更灵活，但成本更高；局部解冻 Vision Tower 适合视觉领域差异很大的任务。

## 课时安排

| 时间 | 内容 |
| --- | --- |
| 0-10 分钟 | 讲解 VLM 三个核心组件 |
| 10-25 分钟 | 对比三种组件选择式微调策略 |
| 25-40 分钟 | 修改 `tuning_mode` 并生成参数报告 |
| 40-52 分钟 | 对比组件微调与 LoRA |
| 52-60 分钟 | 查看 checkpoint 和参数统计 |

## 文件认知

- `config.yaml`：保存模型路径、训练模式和基础训练参数。
- `component_tuning.py`：冻结/解冻指定模块并统计参数。
- `outputs/parameter_report.json`：参数统计报告。
- `outputs/checkpoint/`：保存当前配置下的模型目录。
- `todo.md`：课前准备事项。

## 实验详细步骤

1. 进入目录：

```bash
cd practice_06_component_tuning
```

2. 阅读配置：

```bash
sed -n '1,160p' config.yaml
```

3. 运行默认模式：

```bash
python component_tuning.py --config config.yaml
```

4. 查看参数报告：

```bash
cat outputs/parameter_report.json
```

5. 依次修改 `tuning_mode`：

- `projector_only`
- `projector_llm`
- `vision_partial`

6. 每修改一次就重新运行脚本，并记录 `trainable_ratio`。

## 观察记录

| tuning_mode | total | trainable | trainable_ratio | 适用场景 |
| --- | --- | --- | --- | --- |
| projector_only |  |  |  |  |
| projector_llm |  |  |  |  |
| vision_partial |  |  |  |  |

## 课后练习

- 对比 LoRA 和 projector-only 的参数量。
- 思考业务数据少时应选择哪种策略。
- 思考视觉域差异大时是否需要解冻 Vision Tower。

