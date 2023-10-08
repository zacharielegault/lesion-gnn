#! /usr/bin/env python

import argparse
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToSparseTensor
from tqdm import tqdm

from drgnet.datasets import Aptos
from drgnet.model import DRGNet
from drgnet.transforms import GaussianDistance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    args = parser.parse_args("--config configs/aptos.yaml".split(" "))

    config_path = Path(args.config)
    config = TrainConfig.parse_yaml(config_path)
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose(
        [
            RadiusGraph(3 * config.dataset.distance_sigma_px),
            GaussianDistance(sigma=config.dataset.distance_sigma_px),
            ToSparseTensor(),
        ]
    )

    if config.dataset.name.lower() == "aptos":
        dataset = Aptos(
            root=config.dataset.root,
            transform=transform,
            num_keypoints=config.dataset.num_keypoints,
            sigma=config.dataset.sift_sigma,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.name}")

    train, val = dataset.split(*config.dataset.split)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=False)

    model = DRGNet(dataset.num_features, 32, 4, 50, dataset.num_classes).to(device)
    optim = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()

    acc, kappa = [], []
    outer_pbar = tqdm(range(1, 210 + 1))
    for _ in outer_pbar:
        # Train
        model.train()
        pbar = tqdm(train_loader, position=1, leave=False)
        for data in pbar:
            data = data.to(device)
            logits = model(data.x, data.adj_t, data.batch, data.edge_attr.squeeze())
            loss = criterion(logits, data.y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            pbar.set_description(f"Loss: {loss.item():.4f}")

        # Validate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            pbar = tqdm(val_loader, position=1, leave=False)
            for i, data in enumerate(pbar):
                data = data.to(device)
                logits = model(data.x, data.adj_t, data.batch, data.edge_attr.squeeze())
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(data.y.cpu().tolist())

        val_acc = accuracy_score(targets, preds)
        val_kappa = cohen_kappa_score(targets, preds, weights="quadratic")
        outer_pbar.set_description(f"Accuracy: {val_acc:.4f}, Kappa: {val_kappa:.4f}")
        acc.append(val_acc)
        kappa.append(val_kappa)

    print(f"Best validation accuracy: {max(acc):.4f}")
    print(f"Best validation kappa: {max(kappa):.4f}")


class TrainConfig(BaseModel):
    dataset: "DatasetConfig"
    batch_size: int

    @classmethod
    def parse_yaml(cls, path: Path) -> "TrainConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


class DatasetConfig(BaseModel):
    name: str
    root: str
    split: list[float]
    num_keypoints: int
    sift_sigma: float
    distance_sigma_px: float


if __name__ == "__main__":
    main()
