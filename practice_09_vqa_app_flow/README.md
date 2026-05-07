# 实操 9：多模态 VQA 应用流程搭建

课时：1 小时

本节默认模型、图片和环境已经就绪。学习重点是搭建一个最小多轮 VQA 流程，并通过 Prompt 控制回答格式、回答粒度和结构化字段。

## 学习目标

- 理解图片问答应用的基本流程。
- 学会组织多轮问题。
- 学会让模型完成图片描述、OCR 理解、目标识别和 JSON 输出。
- 学会通过 Prompt 约束模型回答。

## 背景知识

VQA 是 Visual Question Answering，也就是“看图回答问题”。它适合工业巡检、文档理解、现场照片分析等场景。

Prompt 对 VQA 非常重要。一个模糊问题可能得到散乱回答，一个清晰问题可以得到更稳定的结构化输出。例如可以要求模型输出：

```json
{
  "summary": "",
  "objects": [],
  "texts": [],
  "evidence": ""
}
```

## 课时安排

| 时间 | 内容 |
| --- | --- |
| 0-10 分钟 | 讲解 VQA 应用流程 |
| 10-25 分钟 | 演示图片描述、OCR、目标识别 |
| 25-40 分钟 | 演示 Prompt 控制 JSON 输出 |
| 40-52 分钟 | 修改多轮问题并运行 |
| 52-60 分钟 | 查看结果并总结 Prompt 写法 |

## 文件认知

- `config.yaml`：保存模型路径、图片路径、系统提示词和多轮问题。
- `vqa_pipeline.py`：多轮 VQA 流程脚本。
- `outputs/vqa_turns.json`：保存每轮问题和回答。
- `todo.md`：课前准备事项。

## 实验详细步骤

1. 进入目录：

```bash
cd practice_09_vqa_app_flow
```

2. 阅读配置：

```bash
sed -n '1,200p' config.yaml
```

3. 运行 VQA 流程：

```bash
python vqa_pipeline.py --config config.yaml
```

4. 查看输出：

```bash
cat outputs/vqa_turns.json
```

5. 修改 `questions`：

- 图片中有哪些主要对象？
- 图片中是否包含文字？请提取。
- 请按 JSON 输出对象、文字、风险点和依据。

6. 修改 `system_prompt`，加入：

- 必须基于图片可见内容回答。
- 不确定时输出 `unknown`。
- 输出字段固定为 `summary`、`objects`、`texts`、`evidence`。

7. 再次运行并比较输出。

## 观察记录

| Prompt 修改 | 输出是否结构化 | 是否包含依据 | 问题 |
| --- | --- | --- | --- |
| 默认 |  |  |  |
| 增加 JSON 字段 |  |  |  |
| 增加 unknown 规则 |  |  |  |

## 课后练习

- 针对工业图片输出缺陷判断。
- 针对文档截图输出标题和表格字段。
- 针对现场图片输出风险点和处理建议。

