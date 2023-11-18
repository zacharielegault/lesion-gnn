#! /usr/bin/env python

import argparse
from pathlib import Path
from typing import Any, Optional

import lightning as L
import lightning.pytorch as pl
import numpy as np
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pydantic import BaseModel
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToSparseTensor

import wandb
from drgnet.datasets import DDR, Aptos, LESIONSArgs, SIFTArgs
from drgnet.model import DRGNetLightning
from drgnet.transforms import GaussianDistance


class ConfusionMatrixCallback(L.Callback):
    def __init__(
        self, labels=["0-No DR", "1-Moderate", "2-Mild", "3-Advanced", "4-Proliferative", "5-Poor Quality"]
    ) -> None:
        super().__init__()
        self.test_predictions = []
        self.test_groundtruth = []
        self.current_dataset = None
        self.labels = labels

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: DRGNetLightning) -> None:
        self.test_predictions = []
        self.test_groundtruth = []

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: DRGNetLightning,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.current_dataset = trainer.test_dataloaders.dataset.dataset_name
        self.test_predictions.append(outputs[0].cpu().numpy())
        self.test_groundtruth.append(outputs[1].cpu().numpy())

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: DRGNetLightning) -> None:
        cm = wandb.plot.confusion_matrix(
            y_true=np.concatenate(self.test_groundtruth),
            preds=np.concatenate(self.test_predictions),
            class_names=self.labels,
        )
        wandb.log({f"Confusion_Matrix_{self.current_dataset}": cm})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    # args = parser.parse_args("--config DRG-NET/configs/aptos_lesions.yaml".split(" "))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = Config.parse_yaml(config_path)
    print(config)

    L.seed_everything(config.seed)
    logger = WandbLogger(
        project=config.project_name,
        settings=wandb.Settings(code_dir="."),
        entity="liv4d-polytechnique",
        tags=[config.tag],
        config=config.model_dump(),
    )
    run_name = logger.experiment.name

    # Dataset
    transform = Compose(
        [
            RadiusGraph(3 * config.dataset.distance_sigma_px, loop=True),
            GaussianDistance(sigma=config.dataset.distance_sigma_px),
        ]
    )
    if not config.model.compile:
        transform.transforms.append(ToSparseTensor())

    if config.tag.lower() == "sift":
        kwargs: SIFTArgs = dict(num_keypoints=config.dataset.num_keypoints, sigma=config.dataset.sift_sigma)
    elif config.tag.lower() == "lesions":
        kwargs: LESIONSArgs = dict(
            which_features=config.dataset.which_features, feature_layer=config.dataset.feature_layer
        )

    train_dataset = DDR(root=config.dataset.root_ddr, transform=transform, mode=config.tag, variant="train", **kwargs)
    valid_dataset = DDR(root=config.dataset.root_ddr, transform=transform, mode=config.tag, variant="valid", **kwargs)
    test_dataset_ddr = DDR(root=config.dataset.root_ddr, transform=transform, mode=config.tag, variant="test", **kwargs)

    test_dataset_aptos = Aptos(root=config.dataset.root_aptos, transform=transform, mode=config.tag, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    test_loader_ddr = DataLoader(test_dataset_ddr, batch_size=config.batch_size, shuffle=False)
    test_loader_aptos = DataLoader(test_dataset_aptos, batch_size=config.batch_size, shuffle=False)

    # Model
    model = DRGNetLightning(
        input_features=train_dataset.num_features,
        gnn_hidden_dim=config.model.gnn_hidden_dim,
        num_layers=config.model.num_layers,
        sortpool_k=config.model.sortpool_k,
        num_classes=train_dataset.num_classes,
        conv_hidden_dims=config.model.conv_hidden_dims,
        compile=config.model.compile,
    )

    # Training
    trainer = L.Trainer(
        devices=[0],
        max_epochs=config.max_epochs,
        logger=logger,
        check_val_every_n_epoch=10,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"checkpoints/{run_name}/", monitor="val_kappa", mode="max", save_last=True, save_top_k=1
            ),
            ConfusionMatrixCallback(),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader_aptos, ckpt_path="best")
    trainer.test(model, test_loader_ddr, ckpt_path="best")


class Config(BaseModel):
    dataset: "DatasetConfig"
    model: "ModelConfig"
    batch_size: int
    max_epochs: int
    seed: int
    project_name: str
    tag: str

    @classmethod
    def parse_yaml(cls, path: Path) -> "Config":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


class DatasetConfig(BaseModel):
    name: str
    root_aptos: str
    root_ddr: str
    split: tuple[float, float]
    num_keypoints: Optional[int] = None
    sift_sigma: Optional[float] = None
    distance_sigma_px: float
    which_features: Optional[str] = None
    feature_layer: Optional[int] = None
    features_reduction: Optional[str] = "mean"


class ModelConfig(BaseModel):
    gnn_hidden_dim: int
    num_layers: int
    sortpool_k: int
    conv_hidden_dims: tuple[int, int]
    compile: bool


if __name__ == "__main__":
    main()
