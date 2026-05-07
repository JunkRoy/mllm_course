# 实操 2 准备事项

## 模型

- SAM3 本地权重：`../models/sam/sam3.pt`。
- 如果希望自动从 Hugging Face 下载权重，可以在 `config.yaml` 中将 `load_from_hf` 改为 `true`，并确保已经完成 Hugging Face 登录和 SAM3 权重访问授权。

## 数据

- 输入图片：`../data/images/seg_demo.jpg`。
- 建议额外准备小目标、遮挡、边界模糊三类图片。

## 环境依赖

- Python 运行环境。
- PyTorch、TorchVision、OpenCV、NumPy、Pillow、PyYAML。
- Meta SAM3 官方包：`https://github.com/facebookresearch/sam3`。

## 课前检查

```bash
ls ../models/sam/sam3.pt
ls ../data/images/seg_demo.jpg
sed -n '1,120p' requirements.txt
```
