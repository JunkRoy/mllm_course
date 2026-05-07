# 实操 5：基础视觉模型训练与视觉语言模型 LoRA 微调

课时：1.5 小时

本节默认模型、数据和训练环境已经就绪。学习重点是对比“基础视觉模型训练”和“视觉语言模型 LoRA 微调”的数据格式、训练目标和输出结果。

## 学习目标

- 理解基础视觉训练和 VLM LoRA 微调的区别。
- 看懂分类数据目录和 `image + instruction + answer` JSONL 数据。
- 学会修改 epoch、batch size、learning rate 和 LoRA 参数。
- 学会查看 checkpoint 和 adapter 输出。

## 背景知识

基础视觉模型通常学习一个明确视觉任务，例如分类、检测、分割或 OCR。VLM LoRA 微调则是让模型更会按照业务指令回答问题。

LoRA 的核心思想是少量参数训练。基础模型保持不变，只训练一组轻量 adapter。这样训练成本更低，也更方便保存和迁移。

两类训练的区别：

| 对比项 | 基础视觉训练 | VLM LoRA 微调 |
| --- | --- | --- |
| 输入 | 图片和任务标签 | 图片、指令、答案 |
| 输出 | 类别、框、mask 或文字 | 自然语言或结构化回答 |
| 产物 | checkpoint | adapter |
| 目标 | 学视觉任务 | 学回答方式和业务知识 |

## 课时安排

| 时间 | 内容 |
| --- | --- |
| 0-15 分钟 | 对比基础视觉训练和 VLM LoRA 微调 |
| 15-30 分钟 | 讲解两类数据格式 |
| 30-50 分钟 | 运行基础视觉训练并查看 checkpoint |
| 50-75 分钟 | 运行 LoRA 微调并查看 adapter |
| 75-90 分钟 | 对比参数配置、训练输出和适用场景 |

## 文件认知

- `config.yaml`：保存模型路径、数据路径、训练参数和 LoRA 参数。
- `train_vision_task.py`：基础视觉训练入口。
- `finetune_vlm_lora.py`：VLM LoRA 微调入口。
- `outputs/vision/`：基础视觉训练输出。
- `outputs/vlm_lora/`：LoRA 微调输出。
- `todo.md`：课前准备事项。

## 实验详细步骤

1. 进入目录：

```bash
cd practice_05_train_lora
```

2. 阅读训练配置：

```bash
sed -n '1,220p' config.yaml
```

3. 查看基础视觉数据结构：

```bash
find ../data/vision_dataset -maxdepth 2 -type f | head
```

4. 运行基础视觉训练：

```bash
python train_vision_task.py --config config.yaml
```

5. 查看 checkpoint：

```bash
find outputs/vision -type f
```

6. 查看 VLM LoRA 数据：

```bash
head ../data/vlm_lora/train.jsonl
```

7. 运行 LoRA 微调：

```bash
python finetune_vlm_lora.py --config config.yaml
```

8. 查看 adapter 输出：

```bash
find outputs/vlm_lora -type f
```

9. 修改 `config.yaml` 中的参数并记录差异：

- `epochs`
- `batch_size`
- `learning_rate`
- `lora.r`

## 观察记录

| 实验 | 参数修改 | 输出文件 | 观察 |
| --- | --- | --- | --- |
| 基础训练 |  |  |  |
| LoRA 微调 |  |  |  |
| LoRA r 变化 |  |  |  |

## 课后练习

- 为同一张图片写三条不同 instruction。
- 比较 `r=8` 和 `r=16` 的训练产物。
- 思考什么时候用基础视觉训练，什么时候用 VLM 微调。

