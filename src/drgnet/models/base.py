from enum import Enum

import lightning as L
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, CohenKappa, F1Score, Precision, Recall

from drgnet.metrics import (
    ReferableDRAccuracy,
    ReferableDRAUROC,
    ReferableDRAveragePrecision,
    ReferableDRF1,
    ReferableDRPrecision,
    ReferableDRRecall,
)


class OptimizerAlgo(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class LossType(str, Enum):
    MSE = "MSE"
    CE = "CE"
    SMOOTH_L1 = "SmoothL1"


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Nece

    lr: float = 0.001
    weight_decay: float = 0.01
    optimizer_algo: OptimizerAlgo = OptimizerAlgo.ADAMW
    loss_type: LossType = LossType.CE
    class_weights: torch.Tensor | None = None


class BaseModelConfig(BaseModel):
    """Default config for all models. When subclassing BaseLightningModule, a subclass of this config should be created
    as well.
    """

    num_classes: int
    optimizer: OptimizerConfig


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        optimizer_algo: OptimizerAlgo = OptimizerAlgo.ADAMW,
        loss_type: LossType = LossType.CE,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        match loss_type:
            case LossType.MSE:
                self.criterion = nn.MSELoss()
            case LossType.SMOOTH_L1:
                self.criterion = nn.SmoothL1Loss()
            case LossType.CE:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            case other:
                raise ValueError(f"Invalid loss type: {other}")

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_algo = optimizer_algo

        self.setup_metrics()

    def forward(self, data: Data) -> Tensor:
        """Forward pass of the model.

        If the model is probabilistic, the output is logits. If the model does regression, the output is a single value
        per graph.
        """
        raise NotImplementedError("Forward method must be implemented")

    @property
    def is_regression(self) -> bool:
        return self.loss_type in ["MSE", "SmoothL1"]

    @property
    def is_probabilistic(self) -> bool:
        return not self.is_regression

    def setup_metrics(self) -> None:
        self.multiclass_metrics = MetricCollection(
            {
                "micro_acc": Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
                "kappa": CohenKappa(task="multiclass", num_classes=self.num_classes, weights="quadratic"),
                "macro_f1": F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
                "macro_precision": Precision(task="multiclass", num_classes=self.num_classes, average="macro"),
                "macro_recall": Recall(task="multiclass", num_classes=self.num_classes, average="macro"),
            },
            prefix="val_",
        )
        self.referable_metrics = MetricCollection(
            {
                "acc": ReferableDRAccuracy(),
                "f1": ReferableDRF1(),
                "precision": ReferableDRPrecision(),
                "recall": ReferableDRRecall(),
            },
            prefix="val_referable_",
        )

        if self.is_probabilistic:
            self.referable_metrics.add_metrics({"auroc": ReferableDRAUROC(), "auprc": ReferableDRAveragePrecision()})

        self.aptos_multiclass_metrics = self.multiclass_metrics.clone(prefix="test_aptos_")
        self.ddr_multiclass_metrics = self.multiclass_metrics.clone(prefix="test_ddr_")

        self.aptos_referable_metrics = self.referable_metrics.clone(prefix="test_referable_aptos_")
        self.ddr_referable_metrics = self.referable_metrics.clone(prefix="test_referable_ddr_")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        match self.optimizer_algo:
            case OptimizerAlgo.ADAM:
                return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            case OptimizerAlgo.ADAMW:
                return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            case OptimizerAlgo.SGD:
                return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def logits_to_preds(self, logits: Tensor) -> Tensor:
        if not self.is_regression:
            return logits  # FIXME: Shouldn't we argmax here?
        else:
            return torch.round(logits).long()

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        self.log("train_loss", loss, batch_size=batch.num_graphs, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        logits = self.logits_to_preds(logits)

        self.log("val_loss", loss, batch_size=batch.num_graphs)

        self.log_dict(
            self.multiclass_metrics(logits, batch.y), batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )
        self.log_dict(
            self.referable_metrics(logits, batch.y), batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )

    def test_step(self, batch: Data, batch_idx: int) -> None:
        logits = self(batch)
        logits = self.logits_to_preds(logits)
        dataset_name = self.trainer.test_dataloaders.dataset.dataset_name
        if dataset_name == "Aptos":
            multiclass_metric = self.aptos_multiclass_metrics
            referable_metric = self.aptos_referable_metrics
        elif dataset_name == "DDR_test":
            multiclass_metric = self.ddr_multiclass_metrics
            referable_metric = self.ddr_referable_metrics

        self.log_dict(multiclass_metric(logits, batch.y), batch_size=batch.num_graphs, on_step=False, on_epoch=True)
        self.log_dict(referable_metric(logits, batch.y), batch_size=batch.num_graphs, on_step=False, on_epoch=True)
        if not self.is_regression:
            return torch.argmax(logits, dim=1), batch.y
        else:
            return logits, batch.y
