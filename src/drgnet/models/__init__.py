from typing import Union

from .drgnet import DRGNetLightning, DRGNetModelConfig
from .pointnet import PointNetLightning, PointNetModelConfig

# Define union type for all model configs
ModelConfig = Union[DRGNetModelConfig, PointNetModelConfig]


__all__ = ["DRGNetLightning", "PointNetLightning", "ModelConfig"]


def get_model(config: ModelConfig) -> Union[DRGNetLightning, PointNetLightning]:
    """Return a LightningModule for the given config."""
    if isinstance(config, DRGNetModelConfig):
        return DRGNetLightning(config)
    elif isinstance(config, PointNetModelConfig):
        return PointNetLightning(config)
    else:
        raise ValueError(f"Unknown model config type {type(config)}")
