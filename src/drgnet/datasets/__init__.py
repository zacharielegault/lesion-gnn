from typing import Optional, Tuple

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
    num_keypoints: Optional[int] = None
    sift_sigma: Optional[float] = None
    distance_sigma_px: float
    which_features: Optional[str] = None
    feature_layer: Optional[int] = None
    features_reduction: Optional[str] = "mean"
    reinterpolation: Optional[Tuple[int, int]] = None
