# 实操 1：视觉大模型环境检查与基础调用

本章节面向第一次运行多模态模型脚本的同学。目标不是马上训练模型，而是先完成一个最小闭环：

1. 检查 Ubuntu 环境和 Python 依赖。
2. 阅读 `config.yaml`，理解模型路径、图片路径和提示词。
3. 运行一次图片问答推理。
4. 查看 `outputs/result.json`，确认模型回答和运行信息已经保存。

## 1. 学习目标

完成本节后，你应该能做到：

- 看懂一个视觉语言模型推理脚本的基本结构。
- 知道 `config.yaml` 中每个字段的作用。
- 能用一张图片和一个问题调用模型。
- 能判断输出文件是否正常生成。
- 能通过修改 prompt 观察模型回答的变化。

## 2. 目录结构

请先进入项目根目录：

```bash
cd mllm_course
```

查看本节文件：

```bash
ls practice_01_env_basic_call
```

主要文件说明：

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | 保存模型路径、输入图片、提示词、输出目录和生成参数 |
| `run_basic_inference.py` | 基础视觉问答推理脚本 |
| `requirements.txt` | 本节需要的 Python 依赖 |
| `outputs/result.json` | 脚本运行后的结果文件 |

## 3. 环境准备

建议在 Ubuntu 中使用虚拟环境。下面命令假设你已经创建并激活了 Python 环境。

安装依赖：

```bash
pip install -r practice_01_env_basic_call/requirements.txt
```

检查 PyTorch 是否能导入：

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

如果服务器有 NVIDIA GPU，可以查看显卡状态：

```bash
nvidia-smi
```

## 4. 检查配置文件

查看配置：

```bash
sed -n '1,160p' practice_01_env_basic_call/config.yaml
```

你需要重点关注：

| 字段 | 含义 |
| --- | --- |
| `model_path` | 本地模型目录，例如 `../public_pretrain_models/Qwen3-VL-2B-Thinking` |
| `image_path` | 输入图片路径，默认从同级 `dataset` 目录读取 |
| `prompt` | 你希望模型回答的问题 |
| `output_dir` | 输出目录 |
| `max_new_tokens` | 最多生成多少 token |
| `device` | 模型加载设备，通常为 `auto` |

## 5. 运行基础推理

从项目根目录运行：

```bash
python practice_01_env_basic_call/run_basic_inference.py \
  --config practice_01_env_basic_call/config.yaml
```

也可以进入本节目录后运行：

```bash
cd practice_01_env_basic_call
python run_basic_inference.py --config config.yaml
```

正常情况下，终端会打印一段 JSON，并且生成：

```text
practice_01_env_basic_call/outputs/result.json
```

## 6. 查看输出结果

查看结果文件：

```bash
cat practice_01_env_basic_call/outputs/result.json
```

重点检查：

- `gpu`：是否检测到 CUDA 和 GPU 信息。
- `image_path`：本次输入图片路径。
- `prompt`：本次输入问题。
- `answer`：模型生成的回答。

## 7. 修改提示词再运行

打开 `config.yaml`，把 `prompt` 改成更具体的问题，例如：

```yaml
prompt: 请列出图片中的主要对象，并用 JSON 数组返回。
```

再次运行：

```bash
python practice_01_env_basic_call/run_basic_inference.py \
  --config practice_01_env_basic_call/config.yaml
```

观察 `answer` 是否更接近你要求的格式。

## 8. 常见问题

问题：模型路径不存在。

解决：检查 `config.yaml` 中的 `model_path`，确认服务器上已经放好了模型权重。

问题：图片路径不存在。

解决：检查 `image_path`。本课程推荐把公共数据放在项目根目录同级的 `dataset` 目录中。

问题：CUDA 不可用。

解决：先运行 `nvidia-smi`，再检查 PyTorch 是否安装了 CUDA 版本。

问题：输出目录不存在。

说明：脚本会自动创建 `output_dir`，通常不需要手动创建。

## 9. 学习任务

请完成下面练习：

1. 修改 `prompt`，让模型只输出 JSON。
2. 修改 `max_new_tokens`，观察回答长度变化。
3. 更换一张图片，比较模型回答是否变化。
4. 在 `result.json` 中找到模型回答字段。

完成这些任务后，你就跑通了视觉语言模型推理的最小流程。
