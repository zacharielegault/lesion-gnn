import dataclasses
import os.path as osp
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import _get_flattened_data_list
from tqdm import tqdm

from lesion_gnn.utils import ClassWeights

from .nodes.lesions import LesionsExtractor, LesionsNodesConfig
from .nodes.sift import SiftExtractor, SiftNodesConfig


@dataclasses.dataclass(kw_only=True)
class BaseDatasetConfig:
    name: str
    root: str
    nodes: LesionsNodesConfig | SiftNodesConfig
    transform: Callable[..., Any] | None = None
    log: bool = True
    num_workers: int = 0


class BaseDataset(InMemoryDataset):
    def __init__(self, config: BaseDatasetConfig):
        assert config.num_workers >= 0
        self.num_workers = config.num_workers

        self.nodes_config = config.nodes
        if isinstance(self.nodes_config, SiftNodesConfig):
            self.mode = "SIFT"
            pre_transform = SiftExtractor(**dataclasses.asdict(self.nodes_config))
        elif isinstance(self.nodes_config, LesionsNodesConfig):
            self.mode = "LESIONS"
            pre_transform = LesionsExtractor(**dataclasses.asdict(self.nodes_config))
        else:
            raise ValueError(f"Invalid node config: {self.nodes_config}")

        super().__init__(
            root=config.root,
            transform=config.transform,
            pre_transform=pre_transform,
            pre_filter=None,
            log=config.log,
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
            return osp.join(path, f"{self.nodes_config.num_keypoints}")
        elif self.mode == "LESIONS":
            return osp.join(
                path,
                f"{self.nodes_config.which_features}_{self.nodes_config.feature_layer}",
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
        match mode:
            case ClassWeights.UNIFORM:
                return torch.ones_like(counts).float()
            case ClassWeights.INVERSE:
                return 1 / counts
            case ClassWeights.QUADRATIC_INVERSE:
                return 1 / counts**2
            case ClassWeights.INVERSE_FREQUENCY:
                n_samples = counts.sum()
                n_classes = len(counts)
                return n_samples / (n_classes * counts)
            case _:
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

    def _path_and_label_generator(self) -> Iterator[tuple[Path, int]]:
        raise NotImplementedError
