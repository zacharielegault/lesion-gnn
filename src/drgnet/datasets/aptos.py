import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class Aptos(Dataset):
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
        self._diagnosis

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
        return [f"{code}.pt" for code in self._diagnosis["id_code"]]

    # def download(self) -> None:
    #     """Download raw data into `raw_dir`."""
    #     raise NotImplementedError

    def process(self) -> None:
        """Process raw data and save it into the `processed_dir`."""

        def _path_and_label_generator() -> Tuple[Path, int]:
            for row in self._diagnosis.itertuples():
                yield Path(self.raw_dir) / "train" / "images" / f"{row.id_code}.png", row.diagnosis

        if self.num_workers == 0:
            for path, label in tqdm(_path_and_label_generator(), total=len(self._diagnosis)):
                _load_sift_save(
                    img_path=path,
                    label=label,
                    save_path=Path(self.processed_dir) / f"{path.stem}.pt",
                    num_keypoints=self.num_keypoints,
                    sigma=self.sigma,
                )
        else:
            raise NotImplementedError

    def len(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self._diagnosis)

    def get(self, idx: int) -> Any:
        """Return the data object at index `idx`."""
        # FIXME: manage train/test split
        path = Path(self.processed_dir) / f"{self._diagnosis.iloc[idx].id_code}.pt"
        return torch.load(path)


def _load_sift_save(img_path: Path, label: int, save_path: Path, num_keypoints: int, sigma: float) -> None:
    """Load an image from a path and extract SIFT features from it."""
    data = _load_sift(img_path=img_path, label=label, num_keypoints=num_keypoints, sigma=sigma)
    torch.save(data, save_path)


def _load_sift(img_path: Path, label: int, num_keypoints: int, sigma: float) -> Data:
    img = cv2.imread(str(img_path))
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
