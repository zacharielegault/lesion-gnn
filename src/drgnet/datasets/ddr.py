from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Iterator, List, Literal, Tuple

import pandas as pd

from .base import BaseDataset, LESIONSArgs, SIFTArgs


class DDR(BaseDataset):
    """DDR for DR-Grading.

    For more information see https://github.com/nkicsl/DDR-dataset

    Expects the following directory structure:
        <root>
        └── raw
            ├── train
            │   ├── 007-0004-000.jpg
            │   ├── ...
            |   └── 20170323093305605.jpg
            └── train.txt
            ├── test
            │   ├── 007-0004-000.jpg
            │   ├── ...
            |   └── 20170323093305605.jpg
            └── test.txt
            ├── valid
            │   ├── 007-2489-100.jpg
            │   ├── ...
            |   └── 20170413101424496.jpg
            └── valid.txt
    """

    @property
    def raw_file_names(self) -> List[str]:
        """A list of files in the `raw_dir` which needs to be found in order to skip the download."""
        return [f"{self.variant}.txt"]

    @property
    def _diagnosis(self) -> pd.DataFrame:
        df = pd.read_table(
            Path(self.root) / self.raw_file_names[0], header=None, delimiter=" ", names=["filename", "diagnosis"]
        )
        return df

    @property
    def dataset_name(self) -> str:
        return f"DDR_{self.variant}"

    def __init__(
        self,
        root: str | None = None,
        transform: Callable[..., Any] | None = None,
        log: bool = True,
        num_workers: int = 0,
        mode="SIFT",
        variant: Literal["train", "valid", "test"] = "train",
        **pre_transform_kwargs: SIFTArgs | LESIONSArgs,
    ):
        assert variant in ["train", "valid", "test"]
        self.variant = variant
        super().__init__(root, transform, log, num_workers, mode, **pre_transform_kwargs)

    def _path_and_label_generator(self) -> Iterator[Tuple[Path, int]]:
        for row in self._diagnosis.itertuples():
            path = Path(self.ro) / self.variant / row.filename
            label = row.diagnosis
            yield path, label
