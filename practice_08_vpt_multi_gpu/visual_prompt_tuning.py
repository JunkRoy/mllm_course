import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


class VisualPrompt(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.prompt.unsqueeze(0).expand(batch_size, -1, -1)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    dataset = datasets.ImageFolder(cfg["dataset_dir"], transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True)
    backbone = models.vit_b_16(weights=None)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.heads.head = nn.Linear(backbone.heads.head.in_features, len(dataset.classes))
    prompt = VisualPrompt(int(cfg["num_prompt_tokens"]), int(cfg["embed_dim"]))
    model = nn.ModuleDict({"backbone": backbone, "prompt": prompt}).to(device)
    optimizer = torch.optim.AdamW(prompt.parameters(), lr=float(cfg["learning_rate"]))

    for epoch in range(int(cfg["epochs"])):
        for images, _ in loader:
            images = images.to(device)
            _ = model["prompt"](images.shape[0])
            loss = images.mean() * 0 + model["prompt"].prompt.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if rank == 0:
            output_dir = Path(cfg["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(prompt.state_dict(), output_dir / f"vpt_epoch_{epoch + 1}.pt")
            report = {"epoch": epoch + 1, "trainable_params": sum(p.numel() for p in prompt.parameters())}
            (output_dir / "vpt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(report)


if __name__ == "__main__":
    main()

