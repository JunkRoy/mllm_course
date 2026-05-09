# 实操 3：Qwen-Image、Qwen-VL / InternVL 模型调用

课时：1 小时

本节将图像生成与视觉理解拆分为两个独立脚本：
- `qwen_image_generate.py`：仅负责 Qwen-Image 图像生成/编辑。
- `vl_vqa.py`：仅负责 Qwen-VL / InternVL 的 VQA（视觉问答）。

## 学习目标

- 学会用 Qwen-Image 进行文生图和图像编辑。
- 学会用 Qwen-VL / InternVL 对图片进行问答与 OCR。
- 学会通过配置文件管理模型路径、输入输出路径和推理参数。

## 文件说明

- `config.yaml`：统一配置文件，包含 `qwen_image` 和 `vlm` 两个配置块。
- `qwen_image_generate.py`：Qwen-Image 调用脚本（生成与编辑）。
- `vl_vqa.py`：视觉语言模型调用脚本（VQA）。
- `outputs/generated_*.png`：文生图结果。
- `outputs/edited_image.png`：图像编辑结果。
- `outputs/vqa_results.json`：VQA 结果。

## 配置说明（重点）

在 `config.yaml` 中进行配置：

### 1) Qwen-Image 配置（`qwen_image`）

- `model_path`：Qwen-Image 模型目录。
- `output_dir`：图片输出目录。
- `generation_prompts`：文生图提示词列表（可配置多条）。
- `edit_image_path`：待编辑图片路径；设为 `null` 表示跳过编辑。
- `edit_prompt`：图像编辑提示词。

### 2) VLM 配置（`vlm`）

- `model_path`：Qwen-VL / InternVL 模型目录。
- `image_path`：VQA 输入图片路径。
- `output_dir`：VQA 输出目录。
- `max_new_tokens`：回答最大生成长度。
- `questions`：问题列表（可配置多条）。

## 运行脚本

进入目录：

```bash
cd practice_03_qwen_image_vl
```

### 仅运行 Qwen-Image（文生图 + 可选编辑）

```bash
python qwen_image_generate.py --config config.yaml
```

### 仅运行 Qwen-VL / InternVL（VQA）

```bash
python vl_vqa.py --config config.yaml
```

## 可配置项建议

- 图像生成方向：
  - 调整 `generation_prompts`，控制场景、主体、文字、风格。
  - 切换 `edit_prompt` 比较编辑效果。
- VQA 方向：
  - 调整 `questions`，覆盖描述、OCR、结构化 JSON 输出。
  - 调整 `max_new_tokens` 平衡回答完整度和速度。

## 结果查看

```bash
ls outputs
cat outputs/vqa_results.json
```
