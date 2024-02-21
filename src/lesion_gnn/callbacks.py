from typing import Any

import lightning as L
import numpy as np
import wandb
from lightning.pytorch.callbacks import BatchSizeFinder as _BatchSizeFinder
from lightning.pytorch.utilities.exceptions import MisconfigurationException
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


class BatchSizeFinder(_BatchSizeFinder):
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str | None = None) -> None:
        try:
            super().setup(trainer, pl_module, stage)
        except MisconfigurationException as e:
            # FIXME: This is kind of a hack, but the BatchSizeFinder cannot be used with multiple val/test/predict
            # dataloaders
            if "The Batch size finder cannot be used with multiple" in str(e):
                return
            raise e

    def on_validation_start(self, *args, **kwargs) -> None:
        return

    def on_test_start(self, *args, **kwargs) -> None:
        return

    def on_predict_start(self, *args, **kwargs) -> None:
        return
