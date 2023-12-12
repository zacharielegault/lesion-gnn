from typing import Literal

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
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


class BaseModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        optimizer_algo: Literal["adam", "adamw", "sgd"] = "adamw",
        loss_type: Literal["MSE", "CE", "SmoothL1"] = "CE",
        weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        match loss_type:
            case "MSE":
                self.criterion = nn.MSELoss()
            case "SmoothL1":
                self.criterion = nn.SmoothL1Loss()
            case "CE":
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            case other:
                raise ValueError(f"Invalid loss type: {other}")

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_algo = optimizer_algo

        self.setup_metric(probabilistic_predictions=loss_type == "CE")

    @property
    def is_regression(self) -> bool:
        return self.loss_type in ["MSE", "SmoothL1"]

    def setup_metric(self, probabilistic_predictions: bool = True) -> None:
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

        if probabilistic_predictions:
            self.referable_metrics.add_metrics({"auroc": ReferableDRAUROC(), "auprc": ReferableDRAveragePrecision()})

        self.aptos_multiclass_metrics = self.multiclass_metrics.clone(prefix="test_aptos_")
        self.ddr_multiclass_metrics = self.multiclass_metrics.clone(prefix="test_ddr_")

        self.aptos_referable_metrics = self.referable_metrics.clone(prefix="test_referable_aptos_")
        self.ddr_referable_metrics = self.referable_metrics.clone(prefix="test_referable_ddr_")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        match self.optimizer_algo:
            case "adam":
                return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            case "adamw":
                return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            case "sgd":
                return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def logits_to_preds(self, logits: Tensor) -> Tensor:
        if not self.is_regression:
            return logits
        else:
            return torch.round(logits).long()
