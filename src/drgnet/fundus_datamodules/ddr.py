import os
from enum import Enum
from typing import Any, Literal

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from torch import Tensor

from .base import FundusClassificationDataset, FundusDataModule, FundusSegmentationDataset


class DDRVariant(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class DDRClassificationDataset(FundusClassificationDataset):
    def __init__(
        self,
        root: str | bytes | os.PathLike,
        *,
        variant: Literal["train", "valid", "test"] | DDRVariant,
        ignore_ungradable: bool = True,
        transform: A.BasicTransform | None = None,
    ) -> None:
        self.variant = DDRVariant(variant)
        self.transform = transform

        self.image_root = os.path.join(root, "DR_grading", self.variant.value)
        self.labels = pd.read_csv(
            os.path.join(root, "DR_grading", f"{self.variant.value}.txt"),
            sep=" ",
            names=["image", "label"],
        )
        self.labels["image"] = self.labels["image"].str.split(".").str[0]

        if ignore_ungradable:
            self.labels = self.labels[self.labels["label"] != 5]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        image_name = self.labels.iloc[idx, 0]
        image = cv2.imread(os.path.join(self.image_root, image_name + ".jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels.iloc[idx, 1]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.labels["label"].unique())


class DDRSegmentationDataset(FundusSegmentationDataset):
    def __init__(
        self,
        root: str | bytes | os.PathLike,
        *,
        variant: Literal["train", "valid", "test"] | DDRVariant,
        ignore_ungradable: bool = True,
        return_label: bool = False,
        transform: A.BasicTransform | None = None,
    ) -> None:
        self.variant = DDRVariant(variant)
        self.return_label = return_label
        self.transform = transform

        self.image_root = os.path.join(root, "lesion_segmentation", self.variant.value, "image")
        if self.variant == DDRVariant.VALID:  # Validation set has different folder structure
            self.labels_root = os.path.join(root, "lesion_segmentation", self.variant.value, "segmentation label")
        else:
            self.labels_root = os.path.join(root, "lesion_segmentation", self.variant.value, "label")

        self.labels = pd.read_csv(
            os.path.join(root, "DR_grading", f"{self.variant.value}.txt"),
            sep=" ",
            names=["image", "label"],
        )
        self.labels["image"] = self.labels["image"].str.split(".").str[0]
        self.labels = self.labels[
            self.labels["image"].isin([os.path.splitext(f)[0] for f in os.listdir(self.image_root)])
        ]

        if ignore_ungradable:
            self.labels = self.labels[self.labels["label"] != 5]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, int]:
        image_name = self.labels.iloc[idx, 0]

        image = cv2.imread(os.path.join(self.image_root, image_name + ".jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ex = cv2.imread(os.path.join(self.labels_root, "EX", image_name + ".tif"), cv2.IMREAD_GRAYSCALE)
        he = cv2.imread(os.path.join(self.labels_root, "HE", image_name + ".tif"), cv2.IMREAD_GRAYSCALE)
        ma = cv2.imread(os.path.join(self.labels_root, "MA", image_name + ".tif"), cv2.IMREAD_GRAYSCALE)
        se = cv2.imread(os.path.join(self.labels_root, "SE", image_name + ".tif"), cv2.IMREAD_GRAYSCALE)
        mask = np.where(
            (ex > 0) | (he > 0) | (ma > 0) | (se > 0),
            np.argmax([ex, he, ma, se], axis=0) + 1,
            0,
        )

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if self.return_label:
            label = self.labels.iloc[idx, 1]
            return image, mask, label
        else:
            return image, mask


class DDRDataModule(FundusDataModule):
    dataset_cls: type[DDRClassificationDataset | DDRSegmentationDataset]

    dataset_kwargs: dict[str, Any]

    def setup(self, stage: Literal["fit", "validate", "test"]) -> None:
        if stage == "fit":
            self.train = self.dataset_cls(
                root=self.root,
                variant=DDRVariant.TRAIN,
                transform=self.get_transforms(data_aug=self.training_data_aug),
                **self.dataset_kwargs,
            )
            self.val = DDRSegmentationDataset(
                root=self.root,
                variant=DDRVariant.VALID,
                transform=self.get_transforms(),
                **self.dataset_kwargs,
            )

        if stage == "validate":
            self.val = DDRSegmentationDataset(
                root=self.root,
                variant=DDRVariant.VALID,
                transform=self.get_transforms(),
                **self.dataset_kwargs,
            )

        if stage == "test":
            self.test = DDRSegmentationDataset(
                root=self.root,
                variant=DDRVariant.TEST,
                transform=self.get_transforms(),
                **self.dataset_kwargs,
            )


class DDRSegmentationDataModule(DDRDataModule):
    dataset_cls = DDRSegmentationDataset

    def __init__(
        self,
        root: str | bytes | os.PathLike,
        *,
        ignore_ungradable: bool = True,
        return_label: bool = False,
        img_size: tuple[int, int] = (512, 512),
        batch_size: int = 32,
        num_workers: int = 0,
        persistent_workers: bool = False,
        training_data_aug: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            training_data_aug=training_data_aug,
        )

        self.dataset_kwargs = {
            "ignore_ungradable": ignore_ungradable,
            "return_label": return_label,
        }


class DDRClassificationDataModule(DDRDataModule):
    dataset_cls = DDRClassificationDataset

    def __init__(
        self,
        root: str | bytes | os.PathLike,
        *,
        ignore_ungradable: bool = True,
        img_size: tuple[int, int] = (512, 512),
        batch_size: int = 32,
        num_workers: int = 0,
        persistent_workers: bool = False,
        training_data_aug: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            training_data_aug=training_data_aug,
        )
        self.dataset_kwargs = {"ignore_ungradable": ignore_ungradable}
