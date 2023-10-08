import warnings
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Any, Callable, Iterator, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class Aptos(InMemoryDataset):
    """APTOS 2019 Blindness Detection Dataset.

    For more information see https://www.kaggle.com/c/aptos2019-blindness-detection

    Expects the following directory structure:
        <root>
        └── raw
            ├── train
            │   └── images
            │       ├── 0a09aa7356c0.png
            │       ├── ...
            │       └── ffec9a18a3ce.png
            └── train.csv
    """

    def __init__(
        self,
        root: str | None = None,
        transform: Callable[..., Any] | None = None,
        log: bool = True,
        num_workers: int = 0,
        num_keypoints: int = 50,
        sigma: float = 1.6,
    ):
        assert num_workers >= 0
        self.num_workers = num_workers

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=_SiftTransform(num_keypoints=num_keypoints, sigma=sigma),
            pre_filter=None,
            log=log,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """A list of files in the `raw_dir` which needs to be found in order to skip the download."""
        return ["train.csv"]

    @cached_property
    def _diagnosis(self) -> pd.DataFrame:
        return pd.read_csv(Path(self.raw_dir) / "train.csv")

    @property
    def processed_file_names(self) -> List[str]:
        """A list of files in the `processed_dir` which needs to be found in order to skip the processing."""
        return ["data.pt"]

    def process(self) -> None:
        """Process raw data and save it into the `processed_dir`."""

        def _path_and_label_generator() -> Iterator[Tuple[Path, int]]:
            for row in self._diagnosis.itertuples():
                path = Path(self.raw_dir) / "train" / "images" / f"{row.id_code}.png"
                label = row.diagnosis
                yield path, label

        graphs = []

        if self.num_workers == 0:
            for path, label in tqdm(_path_and_label_generator(), total=len(self._diagnosis)):
                data = self.pre_transform(path, label)
                graphs.append(data)
        else:
            raise NotImplementedError

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

    def split(self, *splits: float) -> Tuple["Aptos", ...]:
        """Split the dataset into `len(splits)` datasets.

        If `sum(splits) != 1`, the dataset will be split proportionally to the given values.

        Args:
            *splits (float): proportions of the split

        Returns:
            Tuple[Aptos, ...]: a tuple of `Aptos` datasets
        """
        splits = [0, *splits]
        split = np.cumsum(splits)
        split = split / split[-1]
        idx = len(self) * split
        idx = idx.astype(int)

        dataset = self.shuffle()
        return tuple(dataset[start:end] for start, end in pairwise(idx))


class _SiftTransform:
    def __init__(self, num_keypoints: int, sigma: float):
        """Load an image and extract SIFT keypoints.

        Args:
            num_keypoints (int): number of keypoints to extract
            sigma (float): sigma for SIFT's Gaussian filter
        """
        self.num_keypoints = num_keypoints
        self.sigma = sigma

    def __call__(self, img_path: Path, label: int) -> Data:
        """Load an image and extract SIFT keypoints.

        Args:
            img_path (Path): path to the image
            label (int): DR grade

        Returns:
            Data: a PyG `Data` object with the following attributes:
                - `x` (Tensor): SIFT descriptors (num_keypoints, 128)
                - `pos` (Tensor): (x, y) positions of the keypoints (num_keypoints, 2)
                - `score` (Tensor): SIFT scores (num_keypoints,)
                - `y` (Tensor): DR grade (1,)
                - `name` (str): image name
        """
        img = cv2.imread(str(img_path))
        assert img is not None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Turn off the thresholding and keep the top `num_keypoints` keypoints.
        sift = cv2.SIFT_create(nfeatures=self.num_keypoints, contrastThreshold=0, sigma=self.sigma)

        kpts, desc = sift.detectAndCompute(img, None)

        if len(kpts) == 0:
            warnings.warn(f"Image {img_path} has no keypoints.")
        elif len(kpts) < self.num_keypoints:
            warnings.warn(f"Image {img_path} has less than {self.num_keypoints} keypoints.")

        data = Data(
            x=torch.from_numpy(desc),
            pos=torch.from_numpy(np.array([kpt.pt for kpt in kpts])),  # (x, y)
            score=torch.from_numpy(np.array([kpt.response for kpt in kpts])),
            y=torch.tensor([label]),
            name=img_path.stem,
        )

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_keypoints={self.num_keypoints}, sigma={self.sigma})"
