# 实操 10：模型服务封装、推理加速与多 GPU 部署

课时：1.5 小时

本节默认模型、图片、服务环境和 GPU 环境已经就绪。学习重点是把模型调用封装成服务接口，并理解推理加速、模型合并、量化和多 GPU 部署的基本思路。

## 学习目标

- 理解脚本推理和服务推理的区别。
- 学会阅读模型运行时封装类。
- 学会使用 FastAPI 接收图片和文本并返回 JSON。
- 学会运行 benchmark，记录响应时间。
- 理解模型合并、量化、单 GPU 和多 GPU 推理的差异。

## 背景知识

前面实操大多是脚本式调用，适合实验和验证。真实应用通常需要服务化，把模型封装成 HTTP 接口，供前端、业务系统或其他程序调用。

服务化流程一般包括：

1. 启动服务。
2. 接收图片和文本参数。
3. 调用模型推理。
4. 解析模型输出。
5. 返回 JSON。

推理加速关注三个指标：响应时间、显存占用和输出质量。量化可能降低显存，但也可能影响输出效果，需要实验对比。

## 课时安排

| 时间 | 内容 |
| --- | --- |
| 0-15 分钟 | 讲解脚本推理和服务推理 |
| 15-35 分钟 | 阅读 `model_runtime.py` 的封装逻辑 |
| 35-55 分钟 | 启动 FastAPI 并调用 `/vqa`、`/pipeline` |
| 55-70 分钟 | 运行 benchmark 并记录响应时间 |
| 70-82 分钟 | 讲解模型合并、量化和部署配置 |
| 82-90 分钟 | 串联分割、OCR、VQA 和结构化输出 |

## 文件认知

- `config.yaml`：保存模型路径、精度、设备映射、测试轮数和样例图片。
- `model_runtime.py`：模型加载、推理和端到端流程封装。
- `service_app.py`：FastAPI 服务入口。
- `benchmark_inference.py`：推理耗时测试脚本。
- `merge_quantize_model.py`：模型合并和量化说明入口。
- `outputs/benchmark_report.json`：性能测试结果。
- `todo.md`：课前准备事项。

## 实验详细步骤

1. 进入目录：

```bash
cd practice_10_model_service_deploy
```

2. 阅读配置：

```bash
sed -n '1,220p' config.yaml
```

3. 启动服务：

```bash
uvicorn service_app:app --host 0.0.0.0 --port 8000
```

4. 调用 VQA 接口：

```bash
curl -X POST http://127.0.0.1:8000/vqa \
  -F "image=@../data/images/service_demo.jpg" \
  -F "prompt=请描述图片内容，并输出 JSON。"
```

5. 调用端到端接口：

```bash
curl -X POST http://127.0.0.1:8000/pipeline \
  -F "image=@../data/images/service_demo.jpg" \
  -F "prompt=请输出分割、OCR、VQA 的结构化结果。"
```

6. 运行 benchmark：

```bash
python benchmark_inference.py --config config.yaml
```

7. 查看测试结果：

```bash
cat outputs/benchmark_report.json
```

8. 执行模型合并流程：

```bash
python merge_quantize_model.py --config config.yaml --adapter_path ../outputs/vlm_lora/adapter --merged_output outputs/merged_model
```

## 观察记录

| 配置 | GPU 数 | 精度 | 平均响应时间 | 显存占用 | 输出质量 |
| --- | --- | --- | --- | --- | --- |
| 单 GPU |  |  |  |  |  |
| 多 GPU |  |  |  |  |  |
| 量化 |  |  |  |  |  |

## 课后练习

- 增加 `/health` 接口。
- 增加单独的 `/ocr` 接口。
- 将真实分割、OCR、VQA 模型调用补充到 `run_end_to_end`。

