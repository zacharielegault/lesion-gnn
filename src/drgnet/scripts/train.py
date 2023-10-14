#! /usr/bin/env python

import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from pydantic import BaseModel
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToSparseTensor

from drgnet.datasets import Aptos
from drgnet.model import DRGNetLightning
from drgnet.transforms import GaussianDistance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    args = parser.parse_args("--config configs/aptos.yaml".split(" "))

    config_path = Path(args.config)
    config = Config.parse_yaml(config_path)
    print(config)

    L.seed_everything(config.seed)

    # Dataset
    transform = Compose(
        [
            RadiusGraph(3 * config.dataset.distance_sigma_px),
            GaussianDistance(sigma=config.dataset.distance_sigma_px),
        ]
    )
    if not config.model.compile:
        transform.transforms.append(ToSparseTensor())

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
    model = DRGNetLightning(
        input_features=dataset.num_features,
        gnn_hidden_dim=config.model.gnn_hidden_dim,
        num_layers=config.model.num_layers,
        sortpool_k=config.model.sortpool_k,
        num_classes=dataset.num_classes,
        conv_hidden_dims=config.model.conv_hidden_dims,
        compile=config.model.compile,
    )

    # Training
    trainer = L.Trainer(
        devices=[0],
        max_epochs=config.max_epochs,
        callbacks=[ModelCheckpoint(monitor="val_kappa", mode="max")],
    )
    trainer.fit(model, train_loader, val_loader)


class Config(BaseModel):
    dataset: "DatasetConfig"
    model: "ModelConfig"
    batch_size: int
    max_epochs: int
    seed: int

    @classmethod
    def parse_yaml(cls, path: Path) -> "Config":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


class DatasetConfig(BaseModel):
    name: str
    root: str
    split: tuple[float, float]
    num_keypoints: int
    sift_sigma: float
    distance_sigma_px: float


class ModelConfig(BaseModel):
    gnn_hidden_dim: int
    num_layers: int
    sortpool_k: int
    conv_hidden_dims: tuple[int, int]
    compile: bool


if __name__ == "__main__":
    main()
