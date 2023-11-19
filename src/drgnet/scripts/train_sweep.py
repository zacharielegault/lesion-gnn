#! /usr/bin/env python

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToSparseTensor

import wandb
from drgnet.callbacks import ConfusionMatrixCallback
from drgnet.datasets import DDR, Aptos, LESIONSArgs, SIFTArgs
from drgnet.model import DRGNetLightning
from drgnet.transforms import GaussianDistance
from drgnet.utils import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--gnn_hidden_dim", type=int, default=64, help="GNN hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers.")
    parser.add_argument("--sortpool_k", type=int, default=10, help="Sortpooling k.")
    parser.add_argument("--distance_sigma_px", type=float, default=15, help="Distance sigma in pixels.")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer.")

    args = parser.parse_args()
    config_path = Path(args.config)
    config = Config.parse_yaml(config_path)

    config.model.lr = args.lr
    config.model.weight_decay = args.weight_decay
    config.model.gnn_hidden_dim = args.gnn_hidden_dim
    config.model.num_layers = args.num_layers
    config.model.sortpool_k = args.sortpool_k
    config.dataset.distance_sigma_px = args.distance_sigma_px
    config.model.optimizer_algo = args.optimizer

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader_ddr = DataLoader(test_dataset_ddr, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader_aptos = DataLoader(test_dataset_aptos, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Model
    model = DRGNetLightning(
        input_features=train_dataset.num_features,
        gnn_hidden_dim=config.model.gnn_hidden_dim,
        num_layers=config.model.num_layers,
        sortpool_k=config.model.sortpool_k,
        num_classes=train_dataset.num_classes,
        conv_hidden_dims=config.model.conv_hidden_dims,
        compile=config.model.compile,
        lr=config.model.lr,
        weight_decay=config.model.weight_decay,
        optimizer_algo=config.model.optimizer_algo,
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


if __name__ == "__main__":
    main()
