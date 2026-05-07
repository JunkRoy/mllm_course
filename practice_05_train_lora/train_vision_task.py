import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"]) / "vision"
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(cfg["vision_dataset_dir"], transform=transform)
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(int(cfg["epochs"])):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        torch.save({"model": model.state_dict(), "classes": dataset.classes}, output_dir / f"epoch_{epoch + 1}.pt")
        print({"epoch": epoch + 1, "loss": total_loss / max(1, len(loader))})


if __name__ == "__main__":
    main()

