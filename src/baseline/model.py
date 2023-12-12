from typing import Literal

import timm
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from drgnet.base_model import BaseModel


class BaselineModel(BaseModel):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet18",
        lr: float = 0.001,
        weight_decay: float = 0.01,
        optimizer_algo: Literal["adam", "adamw", "sgd"] = "adamw",
        loss_type: Literal["MSE", "CE", "SmoothL1"] = "CE",
        weights: Tensor | None = None,
    ) -> None:
        super().__init__(num_classes, lr, weight_decay, optimizer_algo, loss_type, weights)

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["diagnosis"])
        self.log("train_loss", loss, batch_size=batch["image"].shape[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["diagnosis"])
        logits = self.logits_to_preds(logits)

        self.log("val_loss", loss, batch_size=batch["image"].shape[0], on_step=False, on_epoch=True)

        self.log_dict(
            self.multiclass_metrics(logits, batch.y), batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )
        self.log_dict(
            self.referable_metrics(logits, batch.y), batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )

    def test_step(self, batch, batch_idx: int) -> None:
        logits = self(batch["image"])
        logits = self.logits_to_preds(logits)
        dataset_name = self.trainer.test_dataloaders.dataset.dataset_name
        if dataset_name == "Aptos":
            multiclass_metric = self.aptos_multiclass_metrics
            referable_metric = self.aptos_referable_metrics
        elif dataset_name == "DDR_test":
            multiclass_metric = self.ddr_multiclass_metrics
            referable_metric = self.ddr_referable_metrics

        self.log_dict(multiclass_metric(logits, batch["diagnosis"]), on_step=False, on_epoch=True)
        self.log_dict(referable_metric(logits, batch["diagnosis"]), on_step=False, on_epoch=True)
        if not self.is_regression:
            return torch.argmax(logits, dim=1), batch["diagnosis"]
        else:
            return logits, batch.y
