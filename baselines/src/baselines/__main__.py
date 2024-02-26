import dataclasses
import os

import lightning as L
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from baselines.timm import TimmConfig, TimmModel
from fundus_datamodules import DDRClassificationDataModule
from lesion_gnn.callbacks import BatchSizeFinder, ConfusionMatrixCallback
from lesion_gnn.models.base import LossType, OptimizerAlgo, OptimizerConfig
from lesion_gnn.utils import ClassWeights

MAX_EPOCHS = 100
VALIDATION_INTERVAL = 10
SEED = 1234
PROJECT_NAME = "DDR-classification"
TAGS = ["BASELINE"]
MONITORED_METRIC, MONITOR_MODE = "val_kappa", "max"
EARLY_STOPPING_PATIENCE = 20

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
TRAINING_DATA_AUG = True

L.seed_everything(SEED)

datamodule = DDRClassificationDataModule(
    root="data/DDR-dataset",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    persistent_workers=True,
    training_data_aug=TRAINING_DATA_AUG,
)
datamodule.setup("fit")  # Setup train (and val) dataset before setting placeholders

# Set placeholders before instantiating the model
model_config = TimmConfig(
    optimizer=OptimizerConfig(
        lr=1e-3,
        lr_scheduler=None,
        weight_decay=1e-4,
        algo=OptimizerAlgo.ADAMW,
        loss_type=LossType.CE,
        class_weights_mode=ClassWeights.UNIFORM,
    ),
    name="resnet18",
)
model_config.optimizer.class_weights.value = datamodule.train.get_class_weights(
    mode=model_config.optimizer.class_weights_mode
)
model_config.num_classes.value = datamodule.train.num_classes

model = TimmModel(model_config)

logger = WandbLogger(
    project=PROJECT_NAME,
    settings=wandb.Settings(code_dir="."),
    entity="liv4d-polytechnique",
    tags=TAGS,
    config={
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "training_data_aug": TRAINING_DATA_AUG,
        "model": dataclasses.asdict(model_config),
    },
)
run_name = logger.experiment.name

callbacks = [
    ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}/",
        monitor=MONITORED_METRIC,
        mode=MONITOR_MODE,
        save_last=True,
        save_top_k=1,
    ),
    ConfusionMatrixCallback(),
    BatchSizeFinder(),
    EarlyStopping(
        monitor=MONITORED_METRIC,
        mode=MONITOR_MODE,
        patience=EARLY_STOPPING_PATIENCE // VALIDATION_INTERVAL,
    ),
]

trainer = L.Trainer(
    devices=[0],
    max_epochs=MAX_EPOCHS,
    logger=logger,
    check_val_every_n_epoch=VALIDATION_INTERVAL,
    log_every_n_steps=len(datamodule.train_dataloader()),
    callbacks=callbacks,
)
trainer.fit(model, datamodule=datamodule)

datamodule.setup("test")
trainer.test(model, datamodule=datamodule, ckpt_path="best")
