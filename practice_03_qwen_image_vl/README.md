# 实操 3：Qwen 图像生成、图像编辑与图片问答

本节包含三类任务：

1. 本地 Qwen-Image 文生图/图像编辑。
2. 通过阿里云百炼 DashScope API 调用图像编辑模型。
3. 使用 VLM 做图片问答。

## 1. 本节文件

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | 本地生成、API 编辑、VLM 问答的统一配置 |
| `qwen_image_generate.py` | 本地 Qwen-Image / Qwen-Image-Edit 推理 |
| `qwen_image_edit_api.py` | 调用 DashScope API 做图像编辑 |
| `vl_vqa.py` | 使用视觉语言模型做图片问答 |
| `requirements.txt` | 本节依赖 |

## 2. 安装依赖

```bash
pip install -r practice_03_qwen_image_vl/requirements.txt
```

如果要调用阿里云 API，还需要设置 API Key：

```bash
export DASHSCOPE_API_KEY="你的 API Key"
```

不要把真实 API Key 写进代码或提交到仓库。

## 3. 配置说明

当前 `config.yaml` 分三块：

| 配置块 | 作用 |
| --- | --- |
| `qwen_image.local` | 本地 Qwen-Image 生成/编辑 |
| `qwen_image.api` | DashScope API 图像编辑 |
| `vlm` | 图片问答 |

API 编辑当前输入图片路径是：

```yaml
qwen_image:
  api:
    input_images:
      - /root/autodl-tmp/dataset/images/img-1.png
```

运行前确认：

```bash
ls /root/autodl-tmp/dataset/images/img-1.png
```

## 4. 本地 Qwen-Image 推理

如果你已经准备好了本地 Qwen-Image 模型，并正确设置了：

```yaml
qwen_image:
  local:
    model_path: ../models/qwen-image
```

运行：

```bash
python practice_03_qwen_image_vl/qwen_image_generate.py \
  --config practice_03_qwen_image_vl/config.yaml
```

只运行文生图：

```bash
python practice_03_qwen_image_vl/qwen_image_generate.py \
  --config practice_03_qwen_image_vl/config.yaml \
  --task generate
```

只运行图像编辑：

```bash
python practice_03_qwen_image_vl/qwen_image_generate.py \
  --config practice_03_qwen_image_vl/config.yaml \
  --task edit
```

输出目录由 `qwen_image.local.output_dir` 控制，当前是：

```text
outputs/
```

## 5. DashScope API 图像编辑

确认配置：

```yaml
qwen_image:
  api:
    model: qwen-image-2.0-pro
    input_images:
      - /root/autodl-tmp/dataset/images/img-1.png
    prompt: 将图片编辑成更清晰的工业检测场景，保留原始主体，补充柔和灯光。
```

运行：

```bash
python practice_03_qwen_image_vl/qwen_image_edit_api.py \
  --config practice_03_qwen_image_vl/config.yaml \
  --save-response
```

脚本会：

1. 读取本地图片。
2. 转成 API 可接收的 data URL。
3. 调用 DashScope。
4. 下载返回图片到 `outputs/api_edited_*.png`。
5. 保存完整响应到 `outputs/api_response.json`。

## 6. 图片问答

`vl_vqa.py` 使用 `vlm` 配置块：

```yaml
vlm:
  model_path: ../models/vlm
  image_path: ../dataset/images/vqa_demo.jpg
  questions:
    - 请描述图片内容。
    - 请识别图片中的文字，并按 JSON 返回。
```

如果你要使用服务器上的 Qwen3-VL，请把 `model_path` 和 `image_path` 改成真实路径。

运行：

```bash
python practice_03_qwen_image_vl/vl_vqa.py \
  --config practice_03_qwen_image_vl/config.yaml
```

输出：

```text
outputs/vqa_results.json
```

## 7. 常见问题

问题：API 报没有 Key。

解决：确认当前终端执行过 `export DASHSCOPE_API_KEY="..."`。

问题：API 没有返回图片。

解决：加 `--save-response`，查看 `outputs/api_response.json` 中的错误信息。

问题：本地模型路径不存在。

解决：检查 `qwen_image.local.model_path` 或 `vlm.model_path` 是否是服务器上的真实模型目录。

问题：图片路径不存在。

解决：检查 `input_images`、`edit_image_path` 或 `vlm.image_path`。

## 8. 建议练习

1. 改 API 的 `prompt`，观察编辑效果。
2. 改 `qwen_image.local.generation_prompts`，尝试不同文生图描述。
3. 改 `vlm.questions`，让模型只输出 JSON。
