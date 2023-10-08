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
        pre_transform: Callable[..., Any] | None = None,
        pre_filter: Callable[..., Any] | None = None,
        log: bool = True,
        num_workers: int = 0,
        num_keypoints: int = 50,
        sigma: float = 1.6,
    ):
        assert num_workers >= 0
        self.num_workers = num_workers

        self.num_keypoints = num_keypoints
        self.sigma = sigma

        super().__init__(root, transform, pre_transform, pre_filter, log)
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
                yield Path(self.raw_dir) / "train" / "images" / f"{row.id_code}.png", row.diagnosis

        graphs = []

        if self.num_workers == 0:
            for path, label in tqdm(_path_and_label_generator(), total=len(self._diagnosis)):
                data = _load_sift(
                    img_path=path,
                    label=label,
                    num_keypoints=self.num_keypoints,
                    sigma=self.sigma,
                )
                graphs.append(data)
        else:
            raise NotImplementedError

        torch.save((self.data, self.slices), self.processed_paths[0])

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


def _load_sift(img_path: Path, label: int, num_keypoints: int, sigma: float) -> Data:
    """Load an image and extract SIFT keypoints.

    Args:
        img_path (Path): path to the image
        label (int): DR grade
        num_keypoints (int): number of keypoints to extract
        sigma (float): sigma for SIFT's Gaussian filter

    Returns:
        Data: a PyG `Data` object with the following attributes:
            - `x` (Tensor): SIFT descriptors (num_keypoints, 128)
            - `pos` (Tensor): (x, y) positions of the keypoints (num_keypoints, 2)
            - `y` (Tensor): DR grade (1,)
    """
    img = cv2.imread(str(img_path))
    assert img is not None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Turn off the thresholding and keep the top `num_keypoints` keypoints.
    sift = cv2.SIFT_create(nfeatures=num_keypoints, contrastThreshold=0, edgeThreshold=1e6, sigma=sigma)

    kpts, desc = sift.detectAndCompute(img, None)

    if len(kpts) == 0:
        warnings.warn(f"Image {img_path} has no keypoints.")
    elif len(kpts) < num_keypoints:
        warnings.warn(f"Image {img_path} has less than {num_keypoints} keypoints.")

    data = Data(
        x=torch.from_numpy(desc),
        pos=torch.from_numpy(np.array([kpt.pt for kpt in kpts])),  # (x, y)
        y=torch.tensor([label]),
    )

    return data
