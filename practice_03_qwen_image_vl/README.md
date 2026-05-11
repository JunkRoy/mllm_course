# 实操 3：Qwen-Image 图像生成与 Qwen-VL 图片问答

本章节把多模态模型调用拆成两个任务：

1. 使用 Qwen-Image 根据文字生成图片，或对已有图片进行编辑。
2. 使用 Qwen-VL / InternVL 对图片进行问答、描述和 OCR。

这两个任务分别由两个脚本完成，方便你单独学习和调试。

## 1. 本节目标

完成本节后，你应该能做到：

- 理解图像生成模型和视觉语言模型的区别。
- 使用配置文件管理模型路径、输入图片、输出目录和提示词。
- 运行文生图脚本并查看生成图片。
- 运行 VQA 脚本并查看 JSON 问答结果。

## 2. 目录结构

从项目根目录查看：

```bash
ls practice_03_qwen_image_vl
```

主要文件：

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | 统一配置文件，包含 `qwen_image` 和 `vlm` 两个配置块 |
| `qwen_image_generate.py` | 图像生成和可选图片编辑脚本 |
| `vl_vqa.py` | 图片问答脚本 |
| `requirements.txt` | 本节依赖 |
| `outputs/generated_*.png` | 文生图结果 |
| `outputs/edited_image.png` | 图片编辑结果 |
| `outputs/vqa_results.json` | 图片问答结果 |

## 3. 安装依赖

```bash
pip install -r practice_03_qwen_image_vl/requirements.txt
```

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. 查看配置文件

```bash
sed -n '1,240p' practice_03_qwen_image_vl/config.yaml
```

配置文件分成两个部分。

### 4.1 Qwen-Image 配置

| 字段 | 含义 |
| --- | --- |
| `model_path` | Qwen-Image 本地模型目录 |
| `output_dir` | 生成图片保存目录 |
| `generation_prompts` | 文生图提示词列表 |
| `edit_image_path` | 可选的待编辑图片路径，设为 `null` 可跳过编辑 |
| `edit_prompt` | 图片编辑提示词 |

### 4.2 VLM 配置

| 字段 | 含义 |
| --- | --- |
| `model_path` | Qwen-VL / InternVL 本地模型目录 |
| `image_path` | 图片问答输入图片 |
| `output_dir` | 问答结果输出目录 |
| `max_new_tokens` | 最大生成长度 |
| `questions` | 要问模型的问题列表 |

## 5. 运行 Qwen-Image 图像生成

从项目根目录执行：

```bash
python practice_03_qwen_image_vl/qwen_image_generate.py \
  --config practice_03_qwen_image_vl/config.yaml
```

运行完成后查看：

```bash
ls practice_03_qwen_image_vl/outputs
```

如果配置了多条 `generation_prompts`，会得到多张：

```text
generated_000.png
generated_001.png
...
```

如果 `edit_image_path` 不为空，并且当前模型管线支持图片编辑，还会生成：

```text
edited_image.png
```

## 6. 运行 Qwen-VL / InternVL 图片问答

```bash
python practice_03_qwen_image_vl/vl_vqa.py \
  --config practice_03_qwen_image_vl/config.yaml
```

查看问答结果：

```bash
cat practice_03_qwen_image_vl/outputs/vqa_results.json
```

结果中每条记录包含：

- `question`：你问模型的问题。
- `answer`：模型生成的回答。

## 7. 修改提示词做实验

你可以修改 `config.yaml` 中的 `generation_prompts`，例如：

```yaml
generation_prompts:
  - 一张工业巡检场景图片，包含安全帽、管道和仪表盘，写实风格。
```

也可以修改 `questions`，例如：

```yaml
questions:
  - 请描述图片中的主要对象。
  - 请识别图片中的文字，并以 JSON 返回。
  - 图片中是否存在安全风险？请说明理由。
```

修改后重新运行对应脚本即可。

## 8. 常见问题

问题：模型路径不存在。

解决：检查 `model_path`，确保模型权重已经放到服务器上。

问题：显存不足。

解决：尝试使用更小模型、减少生成分辨率，或换显存更大的 GPU。

问题：图片编辑没有效果。

说明：不是所有 diffusion pipeline 都支持 `image` 输入。脚本会尽量兼容，但最终取决于模型本身能力。

问题：VQA 输出不符合 JSON。

解决：把问题写得更明确，例如“只输出 JSON，不要输出解释文字”。

## 9. 学习任务

请完成下面练习：

1. 写两条不同风格的文生图 prompt，并比较生成结果。
2. 把 `edit_image_path` 设为 `null`，确认图片编辑步骤被跳过。
3. 给同一张图片提出三个问题：描述、OCR、风险分析。
4. 修改 `max_new_tokens`，观察回答是否更完整。
