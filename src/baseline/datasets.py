from pathlib import Path

import albumentations as A
import nntools.dataset as D
import pandas as pd
from albumentations.pytorch import ToTensorV2
from lightning.data import LightningDataModule
from torch_geometric.data import DataLoader


class FundusDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage == "fit":
            df = pd.read_csv(
                Path(self.config.dataset.ddr_folder) / "train.txt",
                header=None,
                delimiter=" ",
                names=["filename", "diagnosis"],
            )
            self.train_dataset = D.ClassificationDataset(
                img_root=Path(self.config.dataset.ddr_folder) / "train",
                shape=self.config.dataset.image_size,
                label_dataframe=df,
                gt_column="diagnosis",
                file_column="filename",
                keep_size_ratio=True,
                auto_pad=True,
            )
            self.train_dataset.composer = self.train_composer
        if stage == "valid":
            df = pd.read_csv(
                Path(self.config.dataset.ddr_folder) / "valid.txt",
                header=None,
                delimiter=" ",
                names=["filename", "diagnosis"],
            )
            self.valid_dataset = D.ClassificationDataset(
                img_root=Path(self.config.dataset.ddr_folder) / "valid",
                shape=self.config.dataset.image_size,
                label_dataframe=df,
                gt_column="diagnosis",
                file_column="filename",
                keep_size_ratio=True,
                auto_pad=True,
            )
            self.valid_dataset.composer = self.eval_composer
        if stage == "test":
            df = pd.read_csv(
                Path(self.config.dataset.ddr_folder) / "test.txt",
                header=None,
                delimiter=" ",
                names=["filename", "diagnosis"],
            )
            self.test_dataset = D.ClassificationDataset(
                img_root=Path(self.config.dataset.ddr_folder) / "test",
                shape=self.config.dataset.image_size,
                label_dataframe=df,
                gt_column="diagnosis",
                file_column="filename",
                keep_size_ratio=True,
                auto_pad=True,
            )

            df_aptos = pd.read_csv(Path(self.config.dataset.aptos_folder) / "train.csv")

            self.test_dataset_aptos = D.ClassificationDataset(
                img_root=Path(self.config.dataset.aptos_folder) / "train/",
                shape=self.config.dataset.image_size,
                label_dataframe=df_aptos,
                gt_column="diagnosis",
                file_column="id_code",
                keep_size_ratio=True,
                auto_pad=True,
            )
            self.test_dataset.composer = self.eval_composer
            self.test_dataset_aptos.composer = self.eval_composer

    @property
    def data_augs(self):
        return A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.15), A.ShiftScaleRotate(p=0.15)])

    @property
    def normalize_ops(self):
        return A.Compose([A.Normalize(), ToTensorV2()])

    @property
    def train_composer(self):
        composer = D.Composition()
        composer.add(self.data_augs, self.normalize_ops)
        return composer

    @property
    def eval_composer(self):
        composer = D.Composition()
        composer.add(self.normalize_ops)
        return composer

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    def test_dataloader(self):
        return [
            DataLoader(
                self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True
            ),
            DataLoader(
                self.test_dataset_aptos,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            ),
        ]
