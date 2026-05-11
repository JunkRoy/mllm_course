# 实操 4：检测数据集整理、可视化检查与 Qwen-VL 训练数据转换

本章节面向刚接触多模态训练数据处理的同学。你不需要一开始就理解所有代码细节，只要按照步骤执行，就可以完成一条完整的数据处理流程：

1. 整合两个安全帽相关数据集。
2. 统一 XML 标注里的类别名称和检测框格式。
3. 去除比较相似的重复图片。
4. 把检测框和类别画到图片上，人工检查标注质量。
5. 把训练集转换成 Qwen-VL 大模型微调常用的 JSONL prompt 数据。

本实验默认数据目录为：

```text
E:\培训相关\case\2026-中海油-多模态\dataset
```

默认 Python 环境为：

```text
E:\.venv\Scripts\python.exe
```

## 1. 本节你会学到什么

完成本节后，你应该能理解下面几件事：

- 原始图片和 XML 标注文件为什么要一一对应。
- `head`、`helmet`、`hat` 这类类别名为什么需要统一。
- 为什么训练前要划分 `train`、`val`、`test`。
- 如何用可视化图片检查检测框是否正确。
- 如何把传统目标检测数据转换成面向 Qwen-VL 的对话式训练样本。

## 2. 当前目录中的文件

请先进入本章节目录：

```powershell
cd E:\PythonWorkspace\GitWorkspace\mllm_course
```

然后查看目录：

```powershell
dir practice_04_dataset_prelabel_convert
```

你会看到几个重要文件：

| 文件 | 作用 |
| --- | --- |
| `merge_helmet_datasets.py` | 整合两个 VOC 格式数据集，统一类别，并去除近似重复图片 |
| `visualize_voc_boxes.py` | 读取合并后的 XML，把类别名和检测框画到图片上 |
| `voc_to_qwen_vl_jsonl.py` | 读取合并后的训练集，生成 Qwen-VL 微调用 JSONL |
| `requirements.txt` | 本节脚本依赖 |
| `build_and_convert_dataset.py` | 课程原始示例脚本，演示通用数据格式转换 |
| `config.yaml` | 课程原始示例配置文件 |

本次安全帽数据处理主要使用前三个脚本。

## 3. 原始数据应该长什么样

你的数据根目录是：

```text
E:\培训相关\case\2026-中海油-多模态\dataset
```

该目录下应该至少包含：

```text
dataset
├─ Safety Helmet Detection_datasets_
│  ├─ images
│  └─ annotations
├─ VOC2028
│  ├─ JPEGImages
│  ├─ Annotations
│  └─ ImageSets
└─ merged_helmet_voc
```

其中：

- `Safety Helmet Detection_datasets_` 中的标注类别主要是 `head` 和 `helmet`。
- `VOC2028` 中的标注类别主要是 `hat`。
- 本实验会把 `hat` 统一成 `helmet`。
- 合并结果默认输出到 `merged_helmet_voc`。

## 4. 检查 Python 环境

本实验建议固定使用你已经准备好的虚拟环境：

```powershell
& E:\.venv\Scripts\python.exe --version
```

检查依赖是否已经安装：

```powershell
& E:\.venv\Scripts\python.exe -c "import PIL; import tqdm; print('依赖检查通过')"
```

如果提示找不到 `PIL` 或 `tqdm`，可以安装依赖：

```powershell
& E:\.venv\Scripts\python.exe -m pip install -r practice_04_dataset_prelabel_convert\requirements.txt
```

## 5. 第一步：整合两个数据集

脚本：

```text
practice_04_dataset_prelabel_convert\merge_helmet_datasets.py
```

这个脚本会做以下事情：

- 读取 `Safety Helmet Detection_datasets_` 和 `VOC2028`。
- 解析每个 XML 中的 `object/name` 和 `bndbox`。
- 只保留 `head` 和 `helmet` 两类。
- 把 `hat` 改成 `helmet`。
- 跳过无效框、缺图片、缺标注的异常样本。
- 使用图片感知哈希去掉近似重复图片。
- 生成新的 VOC 数据集。

执行命令：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\merge_helmet_datasets.py --overwrite
```

执行完成后，默认输出目录是：

```text
E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc
```

输出目录结构类似：

```text
merged_helmet_voc
├─ Annotations
│  ├─ 000000.xml
│  ├─ 000001.xml
│  └─ ...
├─ JPEGImages
│  ├─ 000000.jpg
│  ├─ 000001.png
│  └─ ...
├─ ImageSets
│  └─ Main
│     ├─ train.txt
│     ├─ val.txt
│     ├─ test.txt
│     └─ trainval.txt
├─ classes.txt
└─ summary.json
```

## 6. 数据合并参数说明

如果你只是第一次跑，使用上一节的命令即可。下面这些参数适合你想调整结果时使用。

指定输入数据目录：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\merge_helmet_datasets.py --dataset-root "E:\培训相关\case\2026-中海油-多模态\dataset" --overwrite
```

指定输出目录：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\merge_helmet_datasets.py --output-dir "E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc_v2" --overwrite
```

关闭近似重复图片去重：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\merge_helmet_datasets.py --dedupe-threshold -1 --overwrite
```

调整去重强度：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\merge_helmet_datasets.py --dedupe-threshold 6 --overwrite
```

说明：

- `--dedupe-threshold` 越大，认为“相似”的图片越多，删除越激进。
- 默认值是 `4`，适合先做一次温和去重。
- 如果担心误删图片，可以先用 `-1` 关闭去重。

## 7. 第二步：检查合并结果

先检查输出目录是否存在：

```powershell
dir "E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc"
```

检查类别文件：

```powershell
type "E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\classes.txt"
```

正常应该看到：

```text
head
helmet
```

检查处理摘要：

```powershell
type "E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\summary.json"
```

重点看这些字段：

- `input_records`：合并前读到多少条有效数据。
- `output_records`：合并和去重后保留多少条数据。
- `duplicates_removed`：去掉了多少张近似重复图片。
- `output_stats`：最终每个类别的标注框数量。

## 8. 第三步：把 XML 标注画到图片上

脚本：

```text
practice_04_dataset_prelabel_convert\visualize_voc_boxes.py
```

它会读取：

```text
merged_helmet_voc\Annotations
merged_helmet_voc\JPEGImages
```

然后把 XML 里的类别和 box 画到图片上，默认保存到：

```text
merged_helmet_voc\visualizations
```

建议先只画 100 张，速度快，也方便人工查看：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\visualize_voc_boxes.py --max-images 100
```

如果确认没问题，再画全部图片：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\visualize_voc_boxes.py
```

指定输出可视化目录：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\visualize_voc_boxes.py --visual-dir "E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\preview"
```

检查图片：

```powershell
dir "E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\visualizations"
```

你需要人工观察：

- `helmet` 框是否框住安全帽。
- `head` 框是否框住未戴安全帽的头部。
- 是否有框明显偏移。
- 是否有类别明显标错。
- 是否有大量重复或质量很差的图片。

## 9. 第四步：转换成 Qwen-VL 训练 JSONL

脚本：

```text
practice_04_dataset_prelabel_convert\voc_to_qwen_vl_jsonl.py
```

它会读取合并数据集中的训练划分：

```text
merged_helmet_voc\ImageSets\Main\train.txt
```

然后生成：

```text
merged_helmet_voc\qwen_vl_train.jsonl
```

执行命令：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\voc_to_qwen_vl_jsonl.py --split train
```

转换验证集：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\voc_to_qwen_vl_jsonl.py --split val
```

转换训练加验证集：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\voc_to_qwen_vl_jsonl.py --split trainval
```

指定 JSONL 输出位置：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\voc_to_qwen_vl_jsonl.py --split train --jsonl-path "E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\qwen_train.jsonl"
```

## 10. Qwen-VL JSONL 长什么样

每一行是一条训练样本。格式大致如下：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "JPEGImages/000001.jpg"
        },
        {
          "type": "text",
          "text": "请检测图片中的安全帽和未佩戴安全帽的头部。请只输出 JSON 数组，每个元素包含 label 和 bbox，bbox 格式为 [xmin, ymin, xmax, ymax]。"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "[{\"label\": \"helmet\", \"bbox\": [10, 20, 80, 100]}]"
        }
      ]
    }
  ]
}
```

这里的含义是：

- 用户输入一张图片。
- 用户要求模型检测安全帽和未戴安全帽的头部。
- 助手答案是 JSON 数组。
- 每个目标包含 `label` 和 `bbox`。
- `bbox` 使用像素坐标 `[xmin, ymin, xmax, ymax]`。

## 11. 使用绝对图片路径

默认 JSONL 中图片路径是相对 `merged_helmet_voc` 的路径，例如：

```text
JPEGImages/000001.jpg
```

如果你的训练框架要求绝对路径，可以加：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\voc_to_qwen_vl_jsonl.py --split train --absolute-image-path
```

这样图片路径会变成类似：

```text
E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\JPEGImages\000001.jpg
```

## 12. 修改提示词

默认提示词是：

```text
请检测图片中的安全帽和未佩戴安全帽的头部。请只输出 JSON 数组，每个元素包含 label 和 bbox，bbox 格式为 [xmin, ymin, xmax, ymax]。
```

如果你想换成自己的提示词，可以这样：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\voc_to_qwen_vl_jsonl.py --split train --prompt "请找出图片中的 helmet 和 head，输出 JSON 数组，字段包括 label 和 bbox。"
```

建议训练时提示词保持稳定，不要同一批数据里频繁改变输出格式要求。

## 13. 推荐完整流程

如果你是第一次操作，建议按照下面顺序执行。

第一步，检查环境：

```powershell
& E:\.venv\Scripts\python.exe -c "import PIL; import tqdm; print('依赖检查通过')"
```

第二步，合并数据集：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\merge_helmet_datasets.py --overwrite
```

第三步，先可视化 100 张检查效果：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\visualize_voc_boxes.py --max-images 100
```

第四步，如果前 100 张看起来正常，再可视化全部：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\visualize_voc_boxes.py
```

第五步，生成 Qwen-VL 训练数据：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\voc_to_qwen_vl_jsonl.py --split train
```

第六步，生成 Qwen-VL 验证数据：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\voc_to_qwen_vl_jsonl.py --split val
```

## 14. 常见问题

问题：提示 `This script needs Pillow`。

解决：

```powershell
& E:\.venv\Scripts\python.exe -m pip install pillow
```

问题：提示找不到 `train.txt`。

原因通常是还没有先运行 `merge_helmet_datasets.py`，或者输出目录不是默认目录。

解决：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\merge_helmet_datasets.py --overwrite
```

问题：可视化图片没有框。

可能原因：

- XML 中没有有效 `object`。
- 类别被过滤掉了。
- 图片和 XML 文件名没有对应上。

建议先检查对应 XML：

```powershell
type "E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\Annotations\000000.xml"
```

问题：去重后图片变少很多。

原因可能是 `--dedupe-threshold` 设置过大。

可以降低阈值：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\merge_helmet_datasets.py --dedupe-threshold 2 --overwrite
```

或者关闭去重：

```powershell
& E:\.venv\Scripts\python.exe practice_04_dataset_prelabel_convert\merge_helmet_datasets.py --dedupe-threshold -1 --overwrite
```

## 15. 学习任务

完成实验后，请你尝试回答：

1. 为什么 `hat` 要映射成 `helmet`？
2. `train.txt`、`val.txt`、`test.txt` 分别有什么作用？
3. 为什么要先画 100 张图片做人工检查？
4. Qwen-VL JSONL 中，`user` 和 `assistant` 分别代表什么？
5. 如果训练时模型输出格式不稳定，应该如何改 prompt？

## 16. 最终产物检查清单

完成本节后，请确认这些文件或目录存在：

```text
E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\Annotations
E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\JPEGImages
E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\ImageSets\Main\train.txt
E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\classes.txt
E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\summary.json
E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\visualizations
E:\培训相关\case\2026-中海油-多模态\dataset\merged_helmet_voc\qwen_vl_train.jsonl
```

看到这些产物，说明你已经完成了从传统检测数据集到多模态大模型训练数据的完整转换流程。
