import argparse

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from fundus_lesions_toolkit.models.segmentation import get_model
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, CohenKappa, F1Score, Precision, Recall

from fundus_datamodules.ddr import DDRClassificationDataModule
from lesion_gnn.metrics import (
    ReferableDRAccuracy,
    ReferableDRAUROC,
    ReferableDRAveragePrecision,
    ReferableDRF1,
    ReferableDRPrecision,
    ReferableDRRecall,
)


class Net(L.LightningModule):
    def __init__(self, features_layer: int, lr: float = 1e-4) -> None:
        """
        features_layer=0 is the image itself, 1 is the first encoder layer, 2 is the second encoder layer, and so on.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.features_layer = features_layer
        self.encoder = get_model("unet", "resnest50d", "All", compile=False).encoder
        num_features = self.encoder.model.feature_info.channels()[self.features_layer - 1]
        self.fc = nn.Linear(num_features, 5)

        self.encoder.requires_grad_(False)

        self.metrics = MetricCollection(
            {
                "micro_acc": Accuracy(task="multiclass", num_classes=5, average="micro"),
                "kappa": CohenKappa(task="multiclass", num_classes=5, weights="quadratic"),
                "macro_f1": F1Score(task="multiclass", num_classes=5, average="macro"),
                "macro_precision": Precision(task="multiclass", num_classes=5, average="macro"),
                "macro_recall": Recall(task="multiclass", num_classes=5, average="macro"),
                "ref_acc": ReferableDRAccuracy(),
                "ref_f1": ReferableDRF1(),
                "ref_precision": ReferableDRPrecision(),
                "ref_recall": ReferableDRRecall(),
                "ref_auroc": ReferableDRAUROC(),
                "ref_auprc": ReferableDRAveragePrecision(),
            }
        )

    def forward(self, img):
        features = self._get_features(img)
        gap = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        return self.fc(gap)

    @torch.inference_mode()
    def _get_features(self, img):
        return self.encoder(img)[self.features_layer]

    def training_step(self, batch, batch_idx):
        img, y = batch
        logits = self(img)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, y = batch
        logits = self(img)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.metrics.update(logits, y)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.metrics.compute(), on_step=False, on_epoch=True)
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        img, y = batch
        logits = self(img)
        self.metrics.update(logits, y)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.metrics.compute(), on_step=False, on_epoch=True)
        self.metrics.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and test the model")
    parser.add_argument("--features_layer", type=int, help="Layer to extract features from")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test", action="store_true", help="Just test the model")
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint. Required for testing")
    args = parser.parse_args()

    L.seed_everything(args.seed)

    assert args.train != args.test, "Either train or test must be selected"
    if args.test:
        assert args.checkpoint is not None, "Checkpoint path must be provided for testing"

    dm = DDRClassificationDataModule(
        root="data/DDR-dataset",
        ignore_ungradable=True,
        img_size=(1536, 1536),
        batch_size=4,
        num_workers=6,
        persistent_workers=True,
        training_data_aug=False,
    )

    if args.train:
        model = Net(args.features_layer)
        logger = WandbLogger(
            project="segmentation-features", config={"seed": args.seed, "features_layer": args.features_layer}
        )
        run_name = logger.experiment.name
        callbacks = [
            ModelCheckpoint(dirpath=f"checkpoints/{run_name}/", monitor="kappa", mode="max", save_last=True),
            EarlyStopping(monitor="kappa", mode="max", patience=20),
        ]
        trainer = L.Trainer(
            devices=[0], max_epochs=100, logger=logger, callbacks=callbacks, benchmark=True, accumulate_grad_batches=32
        )
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    elif args.test:
        model = Net.load_from_checkpoint(args.checkpoint)
        trainer = L.Trainer(devices=[0], benchmark=True)
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
