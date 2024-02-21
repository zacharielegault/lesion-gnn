import dataclasses

import lightning as L
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from lesion_gnn.callbacks import ConfusionMatrixCallback
from lesion_gnn.datasets.datamodule import DataModule
from lesion_gnn.models import get_model
from lesion_gnn.utils.config import Config


def train(config: Config) -> dict[str, float]:
    VALIDATION_INTERVAL = 10

    L.seed_everything(config.seed)

    datamodule = DataModule(config.dataset, compile=config.model.compile)
    datamodule.setup("fit")  # Setup train (and val) dataset before setting placeholders

    # Set placeholders before instantiating the model
    config.model.optimizer.class_weights.value = datamodule.train_datasets.get_class_weights(
        mode=config.model.optimizer.class_weights_mode
    )
    config.model.num_classes.value = datamodule.train_datasets.num_classes
    config.model.input_features.value = datamodule.train_datasets.num_features

    print(config)

    model = get_model(config.model)

    logged_args = dataclasses.asdict(config)
    logged_args["input_features"] = datamodule.train_datasets.num_features
    logger = WandbLogger(
        project=config.project_name,
        settings=wandb.Settings(code_dir="."),
        entity="liv4d-polytechnique",
        tags=config.tags,
        config=logged_args,
    )
    run_name = logger.experiment.name

    callbacks = [
        ModelCheckpoint(
            dirpath=f"checkpoints/{run_name}/",
            monitor=config.monitored_metric,
            mode=config.monitor_mode,
            save_last=True,
            save_top_k=1,
        ),
        ConfusionMatrixCallback(),
    ]
    if config.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=config.monitored_metric,
                mode=config.monitor_mode,
                patience=config.early_stopping_patience // VALIDATION_INTERVAL,
            )
        )

    trainer = L.Trainer(
        devices=[0],
        max_epochs=config.max_epochs,
        logger=logger,
        check_val_every_n_epoch=VALIDATION_INTERVAL,
        log_every_n_steps=len(datamodule.train_dataloader()),
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

    datamodule.setup("test")
    out: list[dict[str, float]] = trainer.test(model, datamodule=datamodule, ckpt_path="best")
    out = {k: v for d in out for k, v in d.items()}  # Flatten list of dicts
    return out
