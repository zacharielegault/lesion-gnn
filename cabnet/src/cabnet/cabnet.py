import lightning as L
import timm
import torch
import torch_geometric
import wandb
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import Accuracy, CohenKappa, ConfusionMatrix, F1Score, MetricCollection, Precision, Recall

from lesion_gnn.metrics import (
    ReferableDRAccuracy,
    ReferableDRAUROC,
    ReferableDRAveragePrecision,
    ReferableDRF1,
    ReferableDRPrecision,
    ReferableDRRecall,
)


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.conv(self.gap(x))


class SpatialAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * x.mean(dim=1, keepdim=True).sigmoid()


class GlobalAttentioonBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.channel_attn = ChannelAttentionBlock(in_channels)
        self.spatial_attn = SpatialAttentionBlock()

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial_attn(self.channel_attn(x))


class CategoryAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, k: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.conv = nn.Conv2d(in_channels, num_classes * k, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_classes * k)
        self.relu = nn.ReLU()
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape
        inputs = x
        conv = self.relu(self.bn(self.conv(x)))
        intra_class_avg = conv.reshape(B, self.num_classes, self.k, H, W).mean(dim=2)  # B x num_classes x H x W
        s = self.gmp(conv).reshape(B, self.num_classes, self.k, 1).mean(dim=-2, keepdim=True)  # B x num_classes x 1 x 1
        m = (intra_class_avg * s).mean(dim=1, keepdim=True)  # B x 1 x H x W
        semantic = inputs * m  # B x C x H x W
        return semantic


class CABNet(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        k: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        optimizer: str = "adamw",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(-1,))
        num_features = self.backbone.feature_info.channels(-1)
        self.global_attention = GlobalAttentioonBlock(in_channels=num_features)
        self.category_attention = CategoryAttentionBlock(in_channels=num_features, num_classes=num_classes, k=k)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_classes)

        self.metrics = nn.ModuleDict(
            {
                "template": MetricCollection(
                    {
                        "confusion_matrix": ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                        "micro_acc": Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
                        "kappa": CohenKappa(task="multiclass", num_classes=self.num_classes, weights="quadratic"),
                        "macro_f1": F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
                        "macro_precision": Precision(task="multiclass", num_classes=self.num_classes, average="macro"),
                        "macro_recall": Recall(task="multiclass", num_classes=self.num_classes, average="macro"),
                        "ref_acc": ReferableDRAccuracy(),
                        "ref_f1": ReferableDRF1(),
                        "ref_precision": ReferableDRPrecision(),
                        "ref_recall": ReferableDRRecall(),
                        "ref_auroc": ReferableDRAUROC(),
                        "ref_auprc": ReferableDRAveragePrecision(),
                    }
                )
            }
        )

    def get_metrics(self, stage: str) -> MetricCollection:
        prefix = f"{stage}_"
        if prefix not in self.metrics:
            self.metrics[prefix] = self.metrics["template"].clone(prefix=prefix)
        return self.metrics[prefix]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        try:
            # Try to resolve the optimizer from torch_geometric, which looks for the optimizer in torch.optim
            optimizer = torch_geometric.nn.resolver.optimizer_resolver(
                self.optimizer, self.parameters(), lr=self.lr, weight_decay=self.weight_decay, **self.optimizer_kwargs
            )
            return optimizer
        except ValueError:
            pass

        try:
            # Otherwise, try to resolve the optimizer from timm, which has its own optimizers
            optimizer = timm.optim.create_optimizer_v2(
                self.parameters(), self.optimizer, lr=self.lr, weight_decay=self.weight_decay, **self.optimizer_kwargs
            )
            return optimizer
        except AssertionError:  # timm.optim.create_optimizer_v2 raises an AssertionError if the optimizer is not found
            print(self.optimizer, "not found in timm.optim.create_optimizer_v2")
            pass

        # If all else fails, raise an error
        raise ValueError(f"Could not resolve optimizer '{self.optimizer}'")

    def forward(self, img: Tensor) -> Tensor:
        (x,) = self.backbone(img)
        x = self.global_attention(x)
        x = self.category_attention(x)
        x = self.gap(x).squeeze()
        logits = self.fc(x)
        return logits

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        img, target = batch
        logits = self(img)
        loss = F.cross_entropy(logits, target)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        img, target = batch
        logits = self(img)
        metrics = self.get_metrics("val")
        metrics.update(logits, target)

    def on_validation_epoch_end(self) -> None:
        metrics = self.get_metrics("val")
        metrics_dict = metrics.compute()
        confmat = metrics_dict.pop("val_confusion_matrix")  # Log the confusion matrix separately
        self.log_dict(metrics_dict, on_step=False, on_epoch=True)

        y_true, y_pred = recover_preds_from_confmat(confmat)
        cm = wandb.plot.confusion_matrix(
            y_true=y_true.numpy(), preds=y_pred.numpy(), class_names=[str(i) for i in range(self.num_classes)]
        )
        wandb.log({"val_confusion_matrix": cm})

        metrics.reset()

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        img, target = batch
        logits = self(img)
        metrics = self.get_metrics("test")
        metrics.update(logits, target)

    def on_test_epoch_end(self) -> None:
        metrics = self.get_metrics("test")
        metrics_dict = metrics.compute()
        confmat = metrics_dict.pop("test_confusion_matrix")  # Log the confusion matrix separately
        self.log_dict(metrics_dict, on_step=False, on_epoch=True)

        y_true, y_pred = recover_preds_from_confmat(confmat)
        cm = wandb.plot.confusion_matrix(
            y_true=y_true.numpy(), preds=y_pred.numpy(), class_names=[str(i) for i in range(self.num_classes)]
        )
        wandb.log({"test_confusion_matrix": cm})

        metrics.reset()


def recover_preds_from_confmat(confmat: Tensor) -> tuple[Tensor, Tensor]:
    """Recover the predictions and targets from a confusion matrix

    Args:
        confmat (torch.Tensor): Confusion matrix

    Returns:
        tuple: (y_true, y_pred) where y_true and y_pred are the true and predicted labels
    """
    y_true = []
    y_pred = []

    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            y_true.extend([i] * int(confmat[i, j]))
            y_pred.extend([j] * int(confmat[i, j]))

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    return y_true, y_pred
