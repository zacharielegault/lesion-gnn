from itertools import pairwise
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

from .base import BaseDataset


class Aptos(BaseDataset):
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

    @property
    def raw_file_names(self) -> List[str]:
        """A list of files in the `raw_dir` which needs to be found in order to skip the download."""
        return ["train.csv"]

    @property
    def _diagnosis(self) -> pd.DataFrame:
        return pd.read_csv(Path(self.raw_dir) / "train.csv")

    @property
    def dataset_name(self) -> str:
        return "Aptos"

    def _path_and_label_generator(self) -> Iterator[Tuple[Path, int]]:
        for row in self._diagnosis.itertuples():
            path = Path(self.raw_dir) / "train" / "images" / f"{row.id_code}.png"
            label = row.diagnosis
            yield path, label

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
