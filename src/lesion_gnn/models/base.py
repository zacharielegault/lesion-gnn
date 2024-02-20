import dataclasses
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import lightning as L
import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch import Tensor
from torch_geometric.data import Data
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
from lesion_gnn.utils import ClassWeights
from lesion_gnn.utils.placeholder import Placeholder

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler


class OptimizerAlgo(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class LossType(str, Enum):
    MSE = "MSE"
    CE = "CE"
    SMOOTH_L1 = "SmoothL1"


@dataclasses.dataclass(kw_only=True)
class LRSchedulerConfig:
    """Configuration for a learning rate scheduler.

    Args:
        name: Name of the lr scheduler class
        kwargs: Keyword arguments to pass to the lr scheduler, after the optimizer itself
        monitor: Metric to monitor for schedulers that require it (e.g. ReduceLROnPlateau)
        interval: Whether to step the lr scheduler after each epoch or each step
        frequency: Step the lr scheduler every `frequency` intervals
    """

    name: str
    kwargs: dict[str, Any]
    monitor: str = "val_loss"
    interval: Literal["epoch", "step"] = "epoch"
    frequency: int = 1


@dataclasses.dataclass(kw_only=True)
class OptimizerConfig:
    lr: float = 0.001
    lr_scheduler: LRSchedulerConfig | None = None
    weight_decay: float = 0.01
    algo: OptimizerAlgo = OptimizerAlgo.ADAMW
    loss_type: LossType = LossType.CE
    class_weights_mode: ClassWeights = ClassWeights.UNIFORM
    class_weights: Placeholder[torch.Tensor] = dataclasses.field(default_factory=Placeholder, init=False)


@dataclasses.dataclass(kw_only=True)
class BaseModelConfig:
    """Default config for all models. When subclassing BaseLightningModule, a subclass of this config should be created
    as well.
    """

    num_classes: Placeholder[int] = dataclasses.field(default_factory=Placeholder, init=False)
    optimizer: OptimizerConfig
    name: str


class BaseLightningModule(L.LightningModule):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.lr_scheduler_config = config.optimizer.lr_scheduler
        self.loss_type = config.optimizer.loss_type
        self.num_classes = config.num_classes.value
        match config.optimizer.loss_type:
            case LossType.MSE:
                self.criterion = nn.MSELoss()
            case LossType.SMOOTH_L1:
                self.criterion = nn.SmoothL1Loss()
            case LossType.CE:
                self.criterion = nn.CrossEntropyLoss(weight=config.optimizer.class_weights.value)
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

    def logits_to_preds(self, logits: Tensor) -> Tensor:
        if not self.is_regression:
            return logits  # FIXME: Shouldn't we argmax here?
        else:
            return torch.round(logits).long()

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        logits = self(batch)
        y = batch.y.float() if self.is_regression else batch.y
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, batch_size=batch.num_graphs, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> None:
        logits = self(batch)
        y = batch.y.float() if self.is_regression else batch.y
        loss = self.criterion(logits, y)
        logits = self.logits_to_preds(logits)

        dataset_name = list(self.trainer.datamodule.val_datasets.keys())[dataloader_idx]
        multiclass_metrics = self.multiclass_metrics("val", dataset_name)
        referable_metrics = self.referable_metrics("val", dataset_name)

        logging_kwargs = dict(batch_size=batch.num_graphs, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"val_{dataset_name}_loss", loss, **logging_kwargs)
        self.log_dict(multiclass_metrics(logits, batch.y), **logging_kwargs)
        self.log_dict(referable_metrics(logits, batch.y), **logging_kwargs)

    def test_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> None:
        logits = self(batch)
        logits = self.logits_to_preds(logits)

        dataset_name = list(self.trainer.datamodule.test_datasets.keys())[dataloader_idx]
        multiclass_metrics = self.multiclass_metrics("test", dataset_name)
        referable_metrics = self.referable_metrics("test", dataset_name)

        logging_kwargs = dict(batch_size=batch.num_graphs, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log_dict(multiclass_metrics(logits, batch.y), **logging_kwargs)
        self.log_dict(referable_metrics(logits, batch.y), **logging_kwargs)

        if not self.is_regression:
            return torch.argmax(logits, dim=1), batch.y
        else:
            return logits, batch.y
