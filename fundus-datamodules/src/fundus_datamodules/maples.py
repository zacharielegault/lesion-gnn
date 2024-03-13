import os
from enum import Enum
from typing import Any, Literal

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from torch import Tensor

from .base import FundusClassificationDataset, FundusDataModule, FundusSegmentationDataset


class MaplesVariant(str, Enum):
    TRAIN = "train"
    TEST = "test"


class AnatomicalStructure(str, Enum):
    BRIGHT_UNCERTAIN = "bright_uncertains"
    COTTON_WOOL_SPOTS = "cottonWoolSpots"
    DRUSENS = "drusens"
    EXUDATES = "exudates"
    HEMORRHAGES = "hemorrhages"
    MACULA = "macula"
    MICROANEURYSMS = "microaneurysms"
    OPTIC_CUP = "optic_cup"
    OPTIC_DISK = "optic_disc"
    RED_UNCERTAIN = "red_uncertains"
    VESSELS = "vessels"


class MaplesDisease(str, Enum):
    DIABETIC_RETINOPATHY = "DR"
    MACULAR_EDEMA = "ME"


class MaplesClassificationDataset(FundusClassificationDataset):
    def __init__(
        self,
        root: str | os.PathLike,
        *,
        variant: Literal["train", "test"] | MaplesVariant,
        disease: Literal["DR", "ME"] | MaplesDisease = MaplesDisease.DIABETIC_RETINOPATHY,
        transform: A.BasicTransform | A.BaseCompose | None = None,
    ) -> None:
        self.variant = MaplesVariant(variant)
        self.disease = MaplesDisease(disease)
        self.transform = transform

        self.images_root = os.path.join(root, self.variant.value, "fundus")
        self.labels = pd.read_excel(os.path.join(root, "diagnosis.xls"), sheet_name="Summary")
        self.labels = self.labels[
            self.labels["name"].isin(os.path.splitext(f)[0] for f in os.listdir(self.images_root))
        ]
        match self.disease:
            case MaplesDisease.DIABETIC_RETINOPATHY:
                self.labels["DR"] = self.labels["DR"].map({"R0": 0, "R1": 1, "R2": 2, "R3": 3, "R4A": 4})
                self.labels = self.labels.drop(columns=["ME"])
            case MaplesDisease.MACULAR_EDEMA:
                self.labels["ME"] = self.labels["ME"].map({"M0": 0, "M1": 1, "M2": 2})
                self.labels = self.labels.drop(columns=["DR"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        name = self.labels.iloc[idx, 0]
        image = cv2.imread(os.path.join(self.images_root, name + ".png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels.iloc[idx, 1]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.labels["DR"].unique())


class MaplesSegmentationDataset(FundusSegmentationDataset):
    def __init__(
        self,
        root: str | os.PathLike,
        *,
        variant: Literal["train", "test"] | MaplesVariant,
        return_label: bool = False,
        transform: A.BasicTransform | A.BaseCompose | None = None,
    ) -> None:
        self.variant = MaplesVariant(variant)
        self.return_label = return_label
        self.transform = transform

        self.images_root = os.path.join(root, self.variant.value)

        self.labels = pd.read_excel(os.path.join(root, "diagnosis.xls"), sheet_name="Summary")
        self.labels = self.labels[
            self.labels["name"].isin(
                os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.images_root, "bright_uncertains"))
            )
        ]
        self.labels["DR"] = self.labels["DR"].map({"R0": 0, "R1": 1, "R2": 2, "R3": 3, "R4A": 4})
        self.labels = self.labels.drop(columns=["ME"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, int]:
        name = self.labels.iloc[idx, 0]
        image = cv2.imread(os.path.join(self.images_root, "fundus", name + ".png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ex = cv2.imread(os.path.join(self.images_root, "exudates", name + ".png"), cv2.IMREAD_GRAYSCALE)
        he = cv2.imread(os.path.join(self.images_root, "hemorrhages", name + ".png"), cv2.IMREAD_GRAYSCALE)
        ma = cv2.imread(os.path.join(self.images_root, "microaneurysms", name + ".png"), cv2.IMREAD_GRAYSCALE)
        se = cv2.imread(os.path.join(self.images_root, "cottonWoolSpots", name + ".png"), cv2.IMREAD_GRAYSCALE)
        mask = np.where((ex > 0) | (he > 0) | (ma > 0) | (se > 0), np.argmax([ex, he, ma, se], axis=0) + 1, 0)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if self.return_label:
            label = self.labels.iloc[idx, 1]
            return image, mask, label
        else:
            return image, mask


class MaplesDataModule(FundusDataModule):
    dataset_cls: type[MaplesSegmentationDataset | MaplesClassificationDataset]

    dataset_kwargs: dict[str, Any]

    def setup(self, stage: Literal["fit", "validate", "test"]):
        if stage == "fit":
            self.train = self.dataset_cls(
                self.root,
                variant=MaplesVariant.TRAIN,
                transform=self.get_transforms(data_aug=self.training_data_aug),
                **self.dataset_kwargs,
            )
            self.val = self.dataset_cls(
                self.root,
                variant=MaplesVariant.TEST,
                transform=self.get_transforms(),
                **self.dataset_kwargs,
            )

        if stage == "validate":
            self.val = self.dataset_cls(
                self.root,
                variant=MaplesVariant.TEST,
                transform=self.get_transforms(),
                **self.dataset_kwargs,
            )

        if stage == "test":
            self.test = self.dataset_cls(
                self.root,
                variant=MaplesVariant.TEST,
                transform=self.get_transforms(),
                **self.dataset_kwargs,
            )


class MaplesClassificationDataModule(MaplesDataModule):
    dataset_cls = MaplesClassificationDataset

    def __init__(
        self,
        root: str | os.PathLike,
        img_size: tuple[int, int],
        batch_size: int,
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

        self.dataset_kwargs = {}


class MaplesSegmentationDataModule(MaplesDataModule):
    dataset_cls = MaplesSegmentationDataset

    def __init__(
        self,
        root: str | os.PathLike,
        *,
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

        self.dataset_kwargs = {"return_label": return_label}
