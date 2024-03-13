import os
import warnings
from enum import Enum
from importlib import resources
from typing import Literal

import albumentations as A
import cv2
import pandas as pd
from torch import Tensor

from ..base import FundusClassificationDataset, FundusDataModule

__all__ = [
    "AptosVariant",
    "AptosClassificationDataset",
    "AptosClassificationDataModule",
]


class AptosVariant(str, Enum):
    # The official APTOS competition only provides a train set (with image and labels)
    # and a public test set (with images only). train/valid/test splits are taken from
    # the RETFound paper (https://github.com/rmaphoh/RETFound_MAE).
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    PUBLIC_TEST = "public_test"


class AptosClassificationDataset(FundusClassificationDataset):
    def __init__(
        self,
        root: str | os.PathLike,
        *,
        variant: Literal["train", "valid", "test", "public_test"] | AptosVariant,
        transform: A.BasicTransform | A.BaseCompose | None = None,
    ) -> None:
        self.variant = AptosVariant(variant)
        self.transform = transform

        if self.variant == AptosVariant.PUBLIC_TEST:
            warnings.warn("APTOS2019 public test set does not have labels. Labels will be set to -1.")
            self.image_root = os.path.join(root, "test_images")
            self.labels = pd.read_csv(os.path.join(root, "test.csv"))
            self.labels["diagnosis"] = -1
        else:
            self.image_root = os.path.join(root, "train_images")
            self.labels = pd.read_csv(os.path.join(root, "train.csv"))
            names = resources.files(__package__).joinpath(f"{self.variant.value}.txt").read_text().splitlines()
            self.labels = self.labels[self.labels["id_code"].isin(names)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        image_name = self.labels.iloc[idx, 0]
        image = cv2.imread(os.path.join(self.image_root, f"{image_name}.png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels.iloc[idx, 1]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.labels["diagnosis"].unique())


class AptosClassificationDataModule(FundusDataModule):
    def __init__(
        self,
        root: str | os.PathLike,
        *,
        img_size: tuple[int, int] = (512, 512),
        batch_size: int = 32,
        num_workers: int = 0,
        persistent_workers: bool = True,
        training_data_aug: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            training_data_aug=training_data_aug,
        )

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        if stage == "fit":
            self.train = AptosClassificationDataset(
                self.root,
                variant=AptosVariant.TRAIN,
                transform=self.get_transforms(data_aug=self.training_data_aug),
            )
            self.val = AptosClassificationDataset(
                self.root, variant=AptosVariant.VALID, transform=self.get_transforms()
            )

        if stage == "validate":
            self.val = AptosClassificationDataset(
                self.root, variant=AptosVariant.VALID, transform=self.get_transforms()
            )

        if stage == "test":
            self.test = AptosClassificationDataset(
                self.root, variant=AptosVariant.TEST, transform=self.get_transforms()
            )

        if stage == "predict":
            self.predict = AptosClassificationDataset(
                self.root, variant=AptosVariant.PUBLIC_TEST, transform=self.get_transforms()
            )
