import dataclasses
from typing import TYPE_CHECKING, Any

import lightning as L
import timm
import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch import LongTensor, Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, CohenKappa, F1Score, Precision, Recall

from lesion_gnn.metrics import (
    ReferableDRAccuracy,
    ReferableDRAUROC,
    ReferableDRAveragePrecision,
    ReferableDRF1,
    ReferableDRPrecision,
    ReferableDRRecall,
)
from lesion_gnn.models.base import BaseModelConfig, LossType, OptimizerAlgo

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler


@dataclasses.dataclass(kw_only=True)
class TimmConfig(BaseModelConfig):
    pass


class TimmModel(L.LightningModule):
    def __init__(self, config: TimmConfig) -> None:
        super().__init__()
        self.lr_scheduler_config = config.optimizer.lr_scheduler
        self.loss_type = config.optimizer.loss_type
        self.num_classes = config.num_classes.value
        match config.loss_type:
            case LossType.MSE:
                self.criterion = nn.MSELoss()
            case LossType.SMOOTH_L1:
                self.criterion = nn.SmoothL1Loss()
            case LossType.CE:
                self.criterion = nn.CrossEntropyLoss()
            case other:
                raise ValueError(f"Invalid loss type: {other}")

        self.lr = config.optimizer.lr
        self.weight_decay = config.optimizer.weight_decay
        self.optimizer_algo = config.optimizer.algo

        self.setup_metrics()
        self.model = timm.create_model(config.name, pretrained=True, num_classes=config.num_classes.value)

    def setup_metrics(self) -> None:
        self._multiclass_metrics_template = MetricCollection(
            {
                "micro_acc": Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
                "kappa": CohenKappa(task="multiclass", num_classes=self.num_classes, weights="quadratic"),
                "macro_f1": F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
                "macro_precision": Precision(task="multiclass", num_classes=self.num_classes, average="macro"),
                "macro_recall": Recall(task="multiclass", num_classes=self.num_classes, average="macro"),
            }
        )
        self._referable_metrics_template = MetricCollection(
            {
                "acc": ReferableDRAccuracy(),
                "f1": ReferableDRF1(),
                "precision": ReferableDRPrecision(),
                "recall": ReferableDRRecall(),
            }
        )

        if self.is_probabilistic:
            self._referable_metrics_template.add_metrics(
                {
                    "auroc": ReferableDRAUROC(),
                    "auprc": ReferableDRAveragePrecision(),
                }
            )

        self._multiclass_metrics: dict[str, MetricCollection] = dict()
        self._referable_metrics: dict[str, MetricCollection] = dict()

    @property
    def is_regression(self) -> bool:
        return self.loss_type in [LossType.MSE, LossType.SMOOTH_L1]

    @property
    def is_probabilistic(self) -> bool:
        return not self.is_regression

    def multiclass_metrics(self, stage: str, dataset_name: str) -> MetricCollection:
        prefix = f"{stage}_{dataset_name}_"
        if prefix not in self._multiclass_metrics:
            self._multiclass_metrics[prefix] = self._multiclass_metrics_template.clone(prefix=prefix)
        return self._multiclass_metrics[prefix]

    def referable_metrics(self, stage: str, dataset_name: str) -> MetricCollection:
        prefix = f"{stage}_{dataset_name}_"
        if prefix not in self._referable_metrics:
            self._referable_metrics[prefix] = self._referable_metrics_template.clone(prefix=prefix)
        return self._referable_metrics[prefix]

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        match self.optimizer_algo:
            case OptimizerAlgo.ADAM:
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            case OptimizerAlgo.ADAMW:
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            case OptimizerAlgo.SGD:
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler_config is None:
            return optimizer

        if self.lr_scheduler_config.name == "LinearWarmupCosineAnnealingLR":
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, **self.lr_scheduler_config.kwargs)
        else:
            scheduler_cls: type[LRScheduler] = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config.name)
            scheduler = scheduler_cls(optimizer, **self.lr_scheduler_config.kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.lr_scheduler_config.monitor,
                "interval": self.lr_scheduler_config.interval,
                "frequency": self.lr_scheduler_config.frequency,
            },
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, LongTensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, LongTensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.round().long() if self.is_regression else logits

        multiclass_metrics = self._multiclass_metrics_template.clone(prefix="val_")
        referable_metrics = self._referable_metrics_template.clone(prefix="val_")

        logging_kwargs = dict(on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log("val_loss", loss, **logging_kwargs)
        self.log_dict(multiclass_metrics(preds, y), **logging_kwargs)
        self.log_dict(referable_metrics(preds, y), **logging_kwargs)

    def test_step(self, batch: tuple[Tensor, LongTensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.round().long() if self.is_regression else logits

        multiclass_metrics = self._multiclass_metrics_template.clone(prefix="test_")
        referable_metrics = self._referable_metrics_template.clone(prefix="test_")

        logging_kwargs = dict(on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log("test_loss", loss, **logging_kwargs)
        self.log_dict(multiclass_metrics(preds, y), **logging_kwargs)
        self.log_dict(referable_metrics(preds, y), **logging_kwargs)
