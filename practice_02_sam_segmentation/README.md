# 实操 2：SAM2 / SAM3 图像分割

本节用同一张图片和同一组提示（点/框），分别运行 SAM2 和 SAM3，观察分割 mask 的效果。

> 说明：SAM 原生是「输入图像 + prompt（点/框等）-> 输出目标区域」。本实践现在支持**按目标组合 prompt**，而不是只把点和框拆开独立跑。

## 1. 本节文件

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | 模型路径、输入图片、点提示、框提示和输出目录 |
| `segment_with_sam2.py` | 使用 SAM2 做分割 |
| `segment_with_sam3.py` | 使用 SAM3 做分割 |
| `segment_with_prompts.py` | 通用提示分割示例 |
| `requirements.txt` | 本节依赖 |

## 目标级 Prompt（推荐）

你可以在 `config.yaml` 里配置 `targets`，每个目标可同时包含 box / 前景点 / 背景点（以下用安全帽检测示例）：

```yaml
targets:
  - name: "safety_helmet"
    box: [180, 60, 340, 220]
    points:
      - [250, 130, 1]   # 安全帽前景点
      - [180, 250, 0]   # 人体背景点
```

脚本会把同一目标的提示联合送入模型（SAM2 原生支持点+框联合；SAM3 保持 box 主导，并将 point 作为 point_as_box 兼容路径保留）。

如果 `targets` 为空，才会回退到旧的 `point_prompts` / `box_prompts`。

## 2. 当前配置

当前 `config.yaml` 使用服务器真实路径：

```yaml
sam2_model_name: /root/autodl-tmp/sam2.1-hiera-large
sam3_model_name: /root/autodl-tmp/sam3
device: cuda
image_path: /root/autodl-tmp/dataset/helmet/images/helmet_demo_01.jpg
output_dir: outputs
confidence_threshold: 0.5
point_prompts:
  - [250, 130, 1]
box_prompts:
  - [180, 60, 340, 220]
```

运行前确认模型和图片存在：

```bash
ls /root/autodl-tmp/sam2.1-hiera-large
ls /root/autodl-tmp/sam3
ls /root/autodl-tmp/dataset/helmet/images/helmet_demo_01.jpg
```

## 3. 安装依赖

```bash
pip install -r practice_02_sam_segmentation/requirements.txt
```

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. 运行 SAM2

从项目根目录执行：

```bash
python practice_02_sam_segmentation/segment_with_sam2.py \
  --config practice_02_sam_segmentation/config.yaml
```

输出目录：

```text
outputs/sam2/
```

通常会包含分割 mask、叠加可视化图和结果 JSON。

## 5. 运行 SAM3

```bash
python practice_02_sam_segmentation/segment_with_sam3.py \
  --config practice_02_sam_segmentation/config.yaml
```

输出目录：

```text
outputs/sam3/
```

## 6. 提示词怎么改

点提示格式：

```yaml
point_prompts:
  - [x, y, label]
```

其中：

| 值 | 含义 |
| --- | --- |
| `x, y` | 图片上的像素坐标 |
| `label=1` | 前景点，告诉模型“这里是目标” |
| `label=0` | 背景点，告诉模型“这里不是目标” |

框提示格式：

```yaml
box_prompts:
  - [x1, y1, x2, y2]
```

坐标是图片像素坐标，左上角为 `(0, 0)`。

## 7. 建议执行顺序

1. 先运行 SAM2，确认图片和提示点可用。
2. 再运行 SAM3，对比同一框提示下的 mask 差异。
3. 修改 `box_prompts`，观察框越准时 mask 是否更稳定。
4. 修改 `confidence_threshold`，观察 mask 面积变化。

## 8. 常见问题

问题：模型路径不存在。

解决：检查 `sam2_model_name`、`sam3_model_name` 是否指向真实模型目录。

问题：图片打不开。

解决：检查 `image_path` 是否存在，且是常见图片格式，如 jpg/png。

问题：没有 CUDA。

解决：把 `device` 改成 `cpu` 可以跑通流程，但速度会慢很多。

问题：结果看起来不准。

解决：先调整 `box_prompts`，框住更完整的目标；再增加前景点和背景点辅助模型。
