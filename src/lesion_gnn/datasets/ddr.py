import dataclasses
from enum import Enum
from pathlib import Path
from typing import Iterator

import pandas as pd

from .base import BaseDataset, BaseDatasetConfig


class DDRVariant(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@dataclasses.dataclass(kw_only=True)
class DDRConfig(BaseDatasetConfig):
    variant: DDRVariant
    name: str = dataclasses.field(default="DDR", init=False)


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

    def __init__(self, config: DDRConfig):
        assert config.variant in ["train", "valid", "test"]
        self.variant = config.variant
        super().__init__(config)

    @property
    def raw_file_names(self) -> list[str]:
        """A list of files in the `raw_dir` which needs to be found in order to skip the download."""
        return [f"{self.variant.value}.txt"]

    @property
    def _diagnosis(self) -> pd.DataFrame:
        df = pd.read_table(
            Path(self.raw_dir) / self.raw_file_names[0], header=None, delimiter=" ", names=["filename", "diagnosis"]
        )
        return df

    @property
    def dataset_name(self) -> str:
        return f"DDR_{self.variant.value}"

    def _path_and_label_generator(self) -> Iterator[tuple[Path, int]]:
        for row in self._diagnosis.itertuples():
            path = Path(self.raw_dir) / self.variant.value / row.filename
            label = row.diagnosis
            if label > 4:
                continue
            yield path, label
