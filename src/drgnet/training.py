import dataclasses

import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from drgnet.callbacks import ConfusionMatrixCallback
from drgnet.datasets.datamodule import DataModule
from drgnet.models import get_model
from drgnet.utils.config import Config


def train(config: Config):
    print(config)
    L.seed_everything(config.seed)

    datamodule = DataModule(config.dataset, compile=config.model.compile)
    datamodule.setup("fit")  # Setup train (and val) dataset before setting placeholders

    # Set placeholders before instantiating the model
    config.model.optimizer.class_weights.value = datamodule.train_datasets.get_class_weights(
        mode=config.model.optimizer.class_weights_mode
    )
    config.model.num_classes.value = datamodule.train_datasets.num_classes
    config.model.input_features.value = datamodule.train_datasets.num_features
    model = get_model(config.model)

    logged_args = dataclasses.asdict(config)
    logged_args["input_features"] = datamodule.train_datasets.num_features
    logger = WandbLogger(
        project=config.project_name,
        settings=wandb.Settings(code_dir="."),
        entity="liv4d-polytechnique",
        tags=[config.tag],
        config=logged_args,
    )
    run_name = logger.experiment.name

    trainer = L.Trainer(
        devices=[0],
        max_epochs=config.max_epochs,
        logger=logger,
        check_val_every_n_epoch=10,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"checkpoints/{run_name}/",
                monitor=config.monitored_metric,
                mode=config.monitor_mode,
                save_last=True,
                save_top_k=1,
            ),
            ConfusionMatrixCallback(),
        ],
    )
    trainer.fit(model, datamodule=datamodule)

    datamodule.setup("test")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
