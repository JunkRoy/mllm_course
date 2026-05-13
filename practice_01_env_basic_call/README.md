# 实操 1：环境检查与 Qwen3-VL 基础调用

本节目标是先跑通一个最小闭环：读取一张图片，向 Qwen3-VL 提问，把回答保存成 JSON。

## 1. 本节文件

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | 配置模型路径、图片路径、提示词和输出目录 |
| `run_basic_inference.py` | 基础图片问答脚本 |
| `requirements.txt` | 本节依赖 |
| `outputs/result.json` | 推理结果，脚本运行后生成 |

## 2. 当前配置

当前 `config.yaml` 使用服务器上的真实路径：

```yaml
model_path: /root/autodl-tmp/Qwen3-VL-8B-Instruct/
image_path: /root/autodl-tmp/dataset/images/img-1.png
prompt: 请描述图片中的主要内容，并给出可见对象列表。
output_dir: outputs
max_new_tokens: 256
device: auto
```

运行前请确认：

```bash
ls /root/autodl-tmp/Qwen3-VL-8B-Instruct/
ls /root/autodl-tmp/dataset/images/img-1.png
```

## 3. 安装依赖

在项目根目录执行：

```bash
pip install -r practice_01_env_basic_call/requirements.txt
```

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 4. 运行

从项目根目录运行：

```bash
python practice_01_env_basic_call/run_basic_inference.py \
  --config practice_01_env_basic_call/config.yaml
```

脚本会：

1. 加载 `model_path` 中的本地 Qwen3-VL。
2. 打开 `image_path` 指定的图片。
3. 使用 `prompt` 提问。
4. 把回答和 GPU 信息写入 `outputs/result.json`。

## 5. 查看结果

```bash
cat outputs/result.json
```

结果中重点看：

| 字段 | 含义 |
| --- | --- |
| `gpu` | CUDA、GPU 数量和显存状态 |
| `image_path` | 本次输入图片 |
| `prompt` | 本次问题 |
| `answer` | 模型回答 |

## 6. 修改实验

想换问题，只改 `config.yaml`：

```yaml
prompt: 请用 JSON 数组列出图片中的主要物体。
```

想换图片，只改：

```yaml
image_path: /root/autodl-tmp/dataset/images/你的图片.png
```

然后重新运行第 4 步。

## 7. 常见问题

问题：模型路径不存在。

解决：检查 `/root/autodl-tmp/Qwen3-VL-8B-Instruct/` 是否存在，里面应有模型权重和配置文件。

问题：图片路径不存在。

解决：检查 `image_path`，路径必须指向真实图片。

问题：显存不足。

解决：先用更小模型或更小图片验证流程；本节只是基础调用，不建议一开始就换很大的输入。

问题：SSH 断开。

解决：本节推理很短，一般不需要后台运行；后面长时间训练请用 `screen` 或 `nohup`。
