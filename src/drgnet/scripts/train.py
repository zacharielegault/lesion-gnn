#! /usr/bin/env python

import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import EarlyStopping
from pydantic import BaseModel
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToSparseTensor

from drgnet.datasets import Aptos
from drgnet.model import DRGNetLightning
from drgnet.transforms import GaussianDistance


def main():
    L.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    args = parser.parse_args("--config configs/aptos.yaml".split(" "))

    config_path = Path(args.config)
    config = TrainConfig.parse_yaml(config_path)
    print(config)

    # Dataset
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

    # Model
    model = DRGNetLightning(dataset.num_features, 32, 4, 50, dataset.num_classes)

    # Training
    trainer = L.Trainer(
        devices=[0],
        max_epochs=210,
        callbacks=[EarlyStopping(monitor="val_kappa", patience=20, mode="max")],
    )
    trainer.fit(model, train_loader, val_loader)


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
