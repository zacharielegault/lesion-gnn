import dataclasses
import itertools
from typing import Iterable, Literal

import lightning as L
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, ToSparseTensor

from lesion_gnn.transforms import TransformConfig, get_transform
from lesion_gnn.utils import ClassWeights

from .aptos import Aptos, AptosConfig
from .base import BaseDataset
from .ddr import DDR, DDRConfig


def get_dataset(config: AptosConfig | DDRConfig) -> Aptos | DDR:
    if isinstance(config, AptosConfig):
        return Aptos(config)
    elif isinstance(config, DDRConfig):
        return DDR(config)
    else:
        raise ValueError(f"Invalid dataset config: {config}")


@dataclasses.dataclass(kw_only=True)
class DataConfig:
    train_datasets: list[AptosConfig | DDRConfig]
    val_datasets: list[AptosConfig | DDRConfig]
    test_datasets: list[AptosConfig | DDRConfig]
    transforms: list[TransformConfig]
    batch_size: int
    num_workers: int


class DataModule(L.LightningDataModule):
    def __init__(self, config: DataConfig, compile: bool = False):
        super().__init__()
        self.batch_size = config.batch_size
        self.config = config

        transform = Compose([get_transform(config) for config in self.config.transforms])
        if not compile:
            transform.transforms.append(ToSparseTensor())

        for dataset in itertools.chain(self.config.train_datasets, self.config.val_datasets, self.config.test_datasets):
            dataset.transform = transform

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        if stage == "fit":
            self.train_datasets = ConcatDataset([get_dataset(config) for config in self.config.train_datasets])

        if stage in ("fit", "validate"):
            self.val_datasets = {config.name: get_dataset(config) for config in self.config.val_datasets}

        if stage == "test":
            self.test_datasets = {config.name: get_dataset(config) for config in self.config.test_datasets}

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return {
            name: DataLoader(dataset, batch_size=self.batch_size, num_workers=self.config.num_workers)
            for name, dataset in self.val_datasets.items()
        }

    def test_dataloader(self) -> DataLoader:
        return {
            name: DataLoader(dataset, batch_size=self.batch_size, num_workers=self.config.num_workers)
            for name, dataset in self.test_datasets.items()
        }


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Iterable[BaseDataset]) -> None:
        if len(set(dataset.num_features for dataset in datasets)) != 1:
            raise ValueError("All datasets should have the same number of features.")

        if len(set(dataset.num_classes for dataset in datasets)) != 1:
            raise ValueError("All datasets should have the same number of classes.")

        super().__init__(datasets)

    @property
    def num_features(self) -> int:
        return self.datasets[0].num_features

    @property
    def num_classes(self) -> int:
        return self.datasets[0].num_classes

    def get_class_weights(self, mode: ClassWeights = ClassWeights.INVERSE_FREQUENCY) -> torch.Tensor:
        class_counts = sum(dataset.classes_counts for dataset in self.datasets)

        match mode:
            case ClassWeights.UNIFORM:
                return torch.ones_like(class_counts).float()
            case ClassWeights.INVERSE:
                return 1 / class_counts
            case ClassWeights.QUADRATIC_INVERSE:
                return 1 / class_counts**2
            case ClassWeights.INVERSE_FREQUENCY:
                n_samples = class_counts.sum()
                n_classes = len(class_counts)
                return n_samples / (n_classes * class_counts)
            case _:
                raise ValueError(f"Invalid mode: {mode}")
