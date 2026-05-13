# 实操 5：Qwen3-VL LoRA 训练、推理与评估

本节使用 2 张 5090 对 `Qwen3-VL-8B-Instruct` 做 LoRA 微调，并用 `qwen_vl_val.jsonl` 做推理和评估。

## 1. 本节文件

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | 训练、推理、评估的统一配置 |
| `finetune_vlm_lora.py` | LoRA 微调入口 |
| `infer_vlm_lora.py` | 加载 LoRA adapter 做单图或 JSONL 批量推理 |
| `evaluate_vlm_lora.py` | 在 val/test JSONL 上计算 precision、recall、F1 |
| `vlm_lora_utils.py` | 推理和评估共用工具 |
| `train_vision_task.py` | 基础视觉分类训练示例 |
| `requirements.txt` | 本节依赖 |

## 2. 当前服务器路径

当前 `config.yaml` 使用这些真实路径：

```yaml
vlm_model_path: /root/autodl-tmp/Qwen3-VL-8B-Instruct/
vlm_train_jsonl: /autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_train.jsonl
vlm_val_jsonl: /autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_val.jsonl
vlm_test_jsonl: /autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_test.jsonl
vlm_data_root: /autodl-fs/data/dataset/merged_helmet_voc/
lora_adapter_path: outputs/vlm_lora/adapter
```

运行前确认：

```bash
ls /root/autodl-tmp/Qwen3-VL-8B-Instruct/
ls /autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_train.jsonl
ls /autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_val.jsonl
```

如果还没有 JSONL，请先完成 `practice_04_dataset_prelabel_convert`。

## 3. 安装依赖

```bash
pip install -r practice_05_train_lora/requirements.txt
```

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.is_bf16_supported())"
```

## 4. 关键配置解释

训练相关：

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  bf16: true
  gradient_checkpointing: true
```

2 张卡时总 batch size：

```text
2 * 1 * 8 = 16
```

图片 token 控制：

```yaml
model:
  min_pixels: 3136
  max_pixels: 1003520
```

如果报显存不足，或图片视觉 token 太多，优先把 `max_pixels` 调小，例如：

```yaml
max_pixels: 501760
```

不要直接截断多模态序列，否则容易报：

```text
Image features and image tokens do not match
```

## 5. 训练前检查数据

只检查 JSONL 和图片，不加载模型：

```bash
python practice_05_train_lora/finetune_vlm_lora.py \
  --config practice_05_train_lora/config.yaml \
  --check-data
```

测试前 1 条样本能否被 processor 编码：

```bash
python practice_05_train_lora/finetune_vlm_lora.py \
  --config practice_05_train_lora/config.yaml \
  --debug-collate 1
```

逐条扫描全部样本，定位坏样本：

```bash
python practice_05_train_lora/finetune_vlm_lora.py \
  --config practice_05_train_lora/config.yaml \
  --debug-scan -1
```

## 6. 开始 2 卡训练

长期训练建议用 `screen`，避免 SSH 断开导致训练被杀：

```bash
screen -S qwen_lora
```

进入 screen 后运行：

```bash
cd /home/workspace/mllm_course
torchrun --nproc_per_node=2 practice_05_train_lora/finetune_vlm_lora.py \
  --config practice_05_train_lora/config.yaml
```

离开 screen 但不停止训练：

```text
Ctrl + A，然后按 D
```

重新进入：

```bash
screen -r qwen_lora
```

训练输出：

```text
outputs/vlm_lora/checkpoint-*
outputs/vlm_lora/adapter/
```

`adapter/` 就是后续推理和评估要加载的 LoRA。

## 7. 从 checkpoint 恢复训练

查看 checkpoint：

```bash
ls outputs/vlm_lora
```

如果有：

```text
checkpoint-100
```

把 `config.yaml` 改成：

```yaml
resume_from_checkpoint: outputs/vlm_lora/checkpoint-100
```

然后重新运行训练命令。

## 8. 单图推理

`config.yaml` 中已经配置了：

```yaml
inference:
  input_image: /autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_images/000000.jpg
  output_path: outputs/vlm_lora/inference_result.json
```

运行：

```bash
python practice_05_train_lora/infer_vlm_lora.py \
  --config practice_05_train_lora/config.yaml
```

如果 config 中配置了 `inference.jsonl_path`，脚本会优先做批量预测。想临时单图推理，可以显式指定图片：

```bash
python practice_05_train_lora/infer_vlm_lora.py \
  --config practice_05_train_lora/config.yaml \
  --image /autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_images/000001.jpg \
  --prompt "请检测图片中的安全帽和未佩戴安全帽的头部。请只输出 JSON 数组。"
```

## 9. 使用 qwen_vl_val.jsonl 批量预测

当前配置：

```yaml
inference:
  jsonl_path: /autodl-fs/data/dataset/merged_helmet_voc/qwen_vl_val.jsonl
  batch_output_path: outputs/vlm_lora/val_predictions.jsonl
```

直接运行：

```bash
python practice_05_train_lora/infer_vlm_lora.py \
  --config practice_05_train_lora/config.yaml
```

只预测前 20 条：

```bash
python practice_05_train_lora/infer_vlm_lora.py \
  --config practice_05_train_lora/config.yaml \
  --max-samples 20
```

输出：

```text
outputs/vlm_lora/val_predictions.jsonl
```

## 10. 计算 val/test 指标

评估验证集：

```bash
python practice_05_train_lora/evaluate_vlm_lora.py \
  --config practice_05_train_lora/config.yaml \
  --split val
```

评估测试集：

```bash
python practice_05_train_lora/evaluate_vlm_lora.py \
  --config practice_05_train_lora/config.yaml \
  --split test
```

只评估前 20 条，先确认流程：

```bash
python practice_05_train_lora/evaluate_vlm_lora.py \
  --config practice_05_train_lora/config.yaml \
  --split val \
  --max-samples 20
```

输出：

```text
outputs/vlm_lora/eval_predictions.jsonl
outputs/vlm_lora/eval_metrics.json
```

指标含义：

| 指标 | 含义 |
| --- | --- |
| `precision` | 预测出来的框里有多少是对的 |
| `recall` | 标注框里有多少被找到了 |
| `f1` | precision 和 recall 的综合 |
| `iou_threshold` | bbox 匹配阈值，当前默认 `0.5` |

## 11. 常见问题

问题：训练被 `SIGHUP` 杀掉。

解决：用 `screen` 或 `nohup` 后台运行。AutoDL 长任务不要直接挂在 SSH 前台。

问题：系统盘空间小。

解决：大模型和数据放 `/root/autodl-tmp` 或 `/autodl-fs/data`。如果 checkpoint 很大，建议把 `output_dir` 改到数据盘。

问题：报 `Image features and image tokens do not match`。

解决：不要截断多模态序列；当前脚本不会手动截断。若仍然报错，降低 `model.max_pixels`。

问题：报 `Unsupported image file`。

解决：回到 practice_04，用 `--normalize-images` 重新生成 JSONL 和图片。

问题：adapter 路径不存在。

解决：先完成训练，确认 `outputs/vlm_lora/adapter/` 存在；或把 `lora_adapter_path` 改成真实 adapter 目录。

## 12. 推荐顺序

1. 用 practice_04 生成 `qwen_vl_train.jsonl` 和 `qwen_vl_val.jsonl`。
2. `--check-data` 检查训练数据。
3. `--debug-collate 1` 确认 processor 能编码。
4. 用 `screen` 启动 2 卡训练。
5. 训练完成后跑 `infer_vlm_lora.py`。
6. 最后跑 `evaluate_vlm_lora.py` 看指标。
