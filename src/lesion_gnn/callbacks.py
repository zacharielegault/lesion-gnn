from typing import Any

import lightning as L
import numpy as np
import wandb
from lightning.pytorch.utilities.types import STEP_OUTPUT

from lesion_gnn.models.base import BaseLightningModule


class ConfusionMatrixCallback(L.Callback):
    def __init__(
        self, labels=["0-No DR", "1-Moderate", "2-Mild", "3-Advanced", "4-Proliferative", "5-Poor Quality"]
    ) -> None:
        super().__init__()
        self.test_predictions = []
        self.test_groundtruth = []
        self.current_dataset = None
        self.labels = labels

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: BaseLightningModule) -> None:
        self.test_predictions = []
        self.test_groundtruth = []

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: BaseLightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.current_dataset = list(trainer.datamodule.test_datasets.keys())[dataloader_idx]
        self.test_predictions.append(outputs[0].cpu().numpy())
        self.test_groundtruth.append(outputs[1].cpu().numpy())

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: BaseLightningModule) -> None:
        cm = wandb.plot.confusion_matrix(
            y_true=np.concatenate(self.test_groundtruth),
            preds=np.concatenate(self.test_predictions),
            class_names=self.labels,
        )
        wandb.log({f"Confusion_Matrix_{self.current_dataset}": cm})
