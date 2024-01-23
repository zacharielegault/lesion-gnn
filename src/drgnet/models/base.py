import dataclasses
from enum import Enum
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
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
from drgnet.utils.placeholder import Placeholder


class OptimizerAlgo(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class LossType(str, Enum):
    MSE = "MSE"
    CE = "CE"
    SMOOTH_L1 = "SmoothL1"


@dataclasses.dataclass
class OptimizerConfig:
    lr: float = 0.001
    schedule_warmup_epochs: int | None = 10  # Set to None to disable warmup
    schedule_min_lr: float | None = 0.0  # Set to None to disable annealing
    weight_decay: float = 0.01
    algo: OptimizerAlgo = OptimizerAlgo.ADAMW
    loss_type: LossType = LossType.CE
    class_weights: torch.Tensor | None = None


@dataclasses.dataclass
class BaseModelConfig:
    """Default config for all models. When subclassing BaseLightningModule, a subclass of this config should be created
    as well.
    """

    num_classes: Placeholder[int]
    optimizer: OptimizerConfig
    name: str


class BaseLightningModule(L.LightningModule):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.warmup_epochs = config.optimizer.schedule_warmup_epochs
        self.min_lr = config.optimizer.schedule_min_lr or config.optimizer.lr  # If None, disable annealing
        self.loss_type = config.optimizer.loss_type
        self.num_classes = config.num_classes.value
        match config.optimizer.loss_type:
            case LossType.MSE:
                self.criterion = nn.MSELoss()
            case LossType.SMOOTH_L1:
                self.criterion = nn.SmoothL1Loss()
            case LossType.CE:
                self.criterion = nn.CrossEntropyLoss(weight=config.optimizer.class_weights)
            case other:
                raise ValueError(f"Invalid loss type: {other}")

        self.lr = config.optimizer.lr
        self.weight_decay = config.optimizer.weight_decay
        self.optimizer_algo = config.optimizer.algo

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

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        match self.optimizer_algo:
            case OptimizerAlgo.ADAM:
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            case OptimizerAlgo.ADAMW:
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            case OptimizerAlgo.SGD:
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.warmup_epochs is None or self.warmup_epochs == 0:
            # If no warmup, use cosine annealing. If config.optimizer.schedule_min_lr was None, use self.lr as min_lr
            # which means no annealing.
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.trainer.max_epochs, eta_min=self.min_lr
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs=self.warmup_epochs,
                        max_epochs=self.trainer.max_epochs,
                        eta_min=self.min_lr,
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }

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
