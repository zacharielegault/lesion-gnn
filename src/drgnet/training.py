import dataclasses

import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToSparseTensor

from drgnet.callbacks import ConfusionMatrixCallback
from drgnet.datasets import DDR, Aptos
from drgnet.models import get_model
from drgnet.transforms import GaussianDistance
from drgnet.utils.config import Config


def train(config: Config):
    print(config)
    L.seed_everything(config.seed)

    # Dataset
    transform = Compose(
        [
            RadiusGraph(3 * config.dataset.distance_sigma_px, loop=True),
            GaussianDistance(sigma=config.dataset.distance_sigma_px),
        ]
    )
    if not config.model.compile:
        transform.transforms.append(ToSparseTensor())

    if config.dataset.root_ddr is None:
        dataset = Aptos(root=config.dataset.root_aptos, transform=transform, pre_transform_kwargs=config.dataset.nodes)
        train_dataset, valid_dataset = dataset.split(config.dataset.train_split, shuffle=True)
        test_dataset_ddr = test_dataset_aptos = None
    else:
        train_dataset = DDR(
            root=config.dataset.root_ddr,
            transform=transform,
            variant="train",
            pre_transform_kwargs=config.dataset.nodes,
        )
        train_dataset = train_dataset.index_select([i for i, d in enumerate(train_dataset) if d.y < 5])

        valid_dataset = DDR(
            root=config.dataset.root_ddr,
            transform=transform,
            variant="valid",
            pre_transform_kwargs=config.dataset.nodes,
        )
        valid_dataset = valid_dataset.index_select([i for i, d in enumerate(valid_dataset) if d.y < 5])

        test_dataset_ddr = DDR(
            root=config.dataset.root_ddr, transform=transform, variant="test", pre_transform_kwargs=config.dataset.nodes
        )
        test_dataset_ddr = test_dataset_ddr.index_select([i for i, d in enumerate(test_dataset_ddr) if d.y < 5])

        test_dataset_aptos = Aptos(
            root=config.dataset.root_aptos, transform=transform, pre_transform_kwargs=config.dataset.nodes
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    if test_dataset_aptos and test_dataset_ddr:
        test_loader_ddr = DataLoader(test_dataset_ddr, batch_size=config.batch_size, shuffle=False, num_workers=4)
        test_loader_aptos = DataLoader(test_dataset_aptos, batch_size=config.batch_size, shuffle=False, num_workers=4)

    config.model.optimizer.class_weights = train_dataset.get_class_weights(mode="inverse_frequency")

    # Model
    config.model.num_classes.value = train_dataset.num_classes
    config.model.input_features.value = train_dataset.num_features
    model = get_model(config.model)

    logged_args = dataclasses.asdict(config)
    logged_args["input_features"] = train_dataset.num_features

    logger = WandbLogger(
        project=config.project_name,
        settings=wandb.Settings(code_dir="."),
        entity="liv4d-polytechnique",
        tags=[config.tag],
        config=logged_args,
    )
    run_name = logger.experiment.name

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
    if test_dataset_ddr and test_dataset_aptos:
        trainer.test(model, test_loader_aptos, ckpt_path="best")
        trainer.test(model, test_loader_ddr, ckpt_path="best")
