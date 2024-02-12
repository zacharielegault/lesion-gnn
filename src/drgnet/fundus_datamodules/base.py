import os
from typing import Literal

import albumentations as A
import cv2
import lightning as L
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .utils import FundusAutocrop


class FundusDataset(Dataset):
    root: str | bytes | os.PathLike
    transform: A.BasicTransform | None

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"


class FundusClassificationDataset(FundusDataset):
    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        raise NotImplementedError


class FundusSegmentationDataset(FundusDataset):
    return_label: bool

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, int]:
        raise NotImplementedError


class FundusDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        root: str | bytes | os.PathLike,
        img_size: tuple[int, int],
        batch_size: int,
        num_workers: int = 0,
        persistent_workers: bool = False,
        normalization_mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
        normalization_std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
        training_data_aug: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.training_data_aug = training_data_aug

        self.train: FundusDataset | None
        self.val: FundusDataset | None
        self.test: FundusDataset | None

    def get_transforms(self, data_aug: bool = False) -> A.Compose:
        transforms = [
            FundusAutocrop(always_apply=True),
            A.LongestMaxSize(max_size=max(self.img_size), always_apply=True),
            A.PadIfNeeded(
                min_height=self.img_size[0],
                min_width=self.img_size[1],
                always_apply=True,
                border_mode=cv2.BORDER_CONSTANT,
            ),
        ]

        if data_aug:
            transforms += [
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.25),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, scale_limit=0, rotate_limit=10, p=0.25),
                A.HueSaturationValue(val_shift_limit=10, hue_shift_limit=10, sat_shift_limit=10),
                A.Blur(blur_limit=3, p=0.1),
            ]

        transforms += [
            A.Normalize(mean=self.normalization_mean, std=self.normalization_std, always_apply=True),
            ToTensorV2(always_apply=True),
        ]

        return A.Compose(transforms)

    def setup(self, stage: Literal["fit", "validate", "test"]):
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
        )
