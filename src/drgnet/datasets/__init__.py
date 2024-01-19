from pydantic import BaseModel

from .aptos import Aptos
from .ddr import DDR
from .nodes.lesions import LesionsArgs
from .nodes.sift import SiftArgs

__all__ = ["Aptos", "DDR", "SiftArgs", "LesionsArgs", "DatasetConfig"]


class DatasetConfig(BaseModel):
    name: str
    root_aptos: str
    root_ddr: str
    split: tuple[float, float]
    num_keypoints: int | None = None
    sift_sigma: float | None = None
    distance_sigma_px: float
    which_features: str | None = None
    feature_layer: int | None = None
    features_reduction: str | None = "mean"
    reinterpolation: tuple[int, int] | None = None
