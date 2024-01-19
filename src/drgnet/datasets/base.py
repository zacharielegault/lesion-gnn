import os.path as osp
from enum import Enum
from typing import Any, Callable

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import _get_flattened_data_list
from tqdm import tqdm

from .nodes.lesions import LesionsArgs, LesionsExtractor
from .nodes.sift import SiftArgs, SiftExtractor


class ClassWeights(str, Enum):
    UNIFORM = "uniform"
    INVERSE = "inverse"
    QUADRATIC_INVERSE = "quadratic_inverse"
    INVERSE_FREQUENCY = "inverse_frequency"


class BaseDataset(InMemoryDataset):
    def __init__(
        self,
        *,
        root: str,
        pre_transform_kwargs: SiftArgs | LesionsArgs,
        transform: Callable[..., Any] | None = None,
        log: bool = True,
        num_workers: int = 0,
    ):
        assert num_workers >= 0
        self.num_workers = num_workers

        self.pre_transform_kwargs = pre_transform_kwargs
        if isinstance(pre_transform_kwargs, SiftArgs):
            self.mode = "SIFT"
            pre_transform = SiftExtractor(**pre_transform_kwargs.to_dict())
        elif isinstance(pre_transform_kwargs, LesionsArgs):
            self.mode = "LESIONS"
            pre_transform = LesionsExtractor(**pre_transform_kwargs.to_dict())
        else:
            raise ValueError(f"Invalid pre_transform_kwargs: {pre_transform_kwargs}")

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=None,
            log=log,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list[str]:
        """A list of files in the `processed_dir` which needs to be found in order to skip the processing."""
        return ["data.pt"]

    @property
    def processed_dir(self) -> str:
        path = osp.join(self.root, "processed", f"{self.dataset_name}", self.mode)
        if self.mode == "SIFT":
            return osp.join(path, f"{self.pre_transform_kwargs.num_keypoints}")
        elif self.mode == "LESIONS":
            return osp.join(
                path,
                f"{self.pre_transform_kwargs.which_features}_{self.pre_transform_kwargs.feature_layer}",
            )
        else:
            return super().processed_dir

    @property
    def dataset_name(self) -> str:
        raise NotImplementedError("Please implement this property in your subclass.")

    @property
    def _diagnosis(self) -> Any:
        raise NotImplementedError("Please implement this property in your subclass.")

    @property
    def classes_counts(self) -> torch.Tensor:
        data_list = _get_flattened_data_list([data for data in self])
        y = torch.cat([data.y for data in data_list if "y" in data], dim=0)
        _, counts = torch.unique(y, return_counts=True)
        return counts

    def get_class_weights(self, mode: ClassWeights = ClassWeights.INVERSE_FREQUENCY) -> torch.Tensor:
        counts = self.classes_counts
        if mode == "uniform":
            return torch.ones_like(counts)
        elif mode == "inverse":
            return 1 / counts
        elif mode == "quadratic_inverse":
            return 1 / counts**2
        elif mode == "inverse_frequency":
            n_samples = counts.sum()
            n_classes = len(counts)
            return n_samples / (n_classes * counts)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def process(self) -> None:
        """Process raw data and save it into the `processed_dir`."""

        graphs = []

        if self.num_workers == 0:
            for path, label in tqdm(self._path_and_label_generator(), total=len(self._diagnosis)):
                if label > 4:
                    continue
                data = self.pre_transform(path, label)
                graphs.append(data)
        else:
            raise NotImplementedError

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
