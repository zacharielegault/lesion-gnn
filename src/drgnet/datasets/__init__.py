import dataclasses

from .aptos import Aptos
from .ddr import DDR
from .nodes.lesions import LesionsNodesConfig
from .nodes.sift import SiftNodesConfig

__all__ = ["Aptos", "DDR", "SiftNodesConfig", "LesionsNodesConfig", "DatasetConfig"]


@dataclasses.dataclass
class DatasetConfig:
    name: str
    root_aptos: str
    root_ddr: str
    split: tuple[float, float]
    distance_sigma_px: float
    nodes: LesionsNodesConfig | SiftNodesConfig
