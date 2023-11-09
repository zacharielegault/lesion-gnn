import torch
import torchmetrics
from torch import Tensor
from torchmetrics import Metric


class ReferableDRMetric(Metric):
    """Classification metric base class for (binary) referable DR classification.

    A grade of 0 or 1 is considered non-referable, while a grade of 2, 3, or 4 is considered referable.
    """

    def __init__(self) -> None:
        super().__init__()

        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")

    def update(self, logits: Tensor, target: Tensor) -> None:
        multiclass_probs = torch.softmax(logits, dim=-1)
        binary_probs = multiclass_probs[:, 2:].sum(dim=-1)
        binary_target = target >= 2

        self.y_pred.append(binary_probs)
        self.y_true.append(binary_target)


class ReferableDRAccuracy(ReferableDRMetric):
    """Accuracy for (binary) referable DR classification."""

    def compute(self) -> Tensor:
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        return torchmetrics.functional.accuracy(y_pred, y_true, task="binary")


class ReferableDRPrecision(ReferableDRMetric):
    """Precision for (binary) referable DR classification."""

    def compute(self) -> Tensor:
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        return torchmetrics.functional.precision(y_pred, y_true, task="binary")


class ReferableDRRecall(ReferableDRMetric):
    """Recall for (binary) referable DR classification."""

    def compute(self) -> Tensor:
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        return torchmetrics.functional.recall(y_pred, y_true, task="binary")


class ReferableDRF1(ReferableDRMetric):
    """F1 score for (binary) referable DR classification."""

    def compute(self) -> Tensor:
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        return torchmetrics.functional.f1_score(y_pred, y_true, task="binary")


class ReferableDRAUROC(ReferableDRMetric):
    """AUROC for (binary) referable DR classification."""

    def compute(self) -> Tensor:
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        return torchmetrics.functional.auroc(y_pred, y_true, task="binary")


class ReferableDRAveragePrecision(ReferableDRMetric):
    """AUPRC for (binary) referable DR classification."""

    def compute(self) -> Tensor:
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        return torchmetrics.functional.average_precision(y_pred, y_true, task="binary")
