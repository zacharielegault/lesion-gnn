from .drgnet import DRGNetLightning, DRGNetModelConfig
from .gat import GATConfig, GATLightning
from .pointnet import PointNetLightning, PointNetModelConfig
from .set_transformer import SetTransformerLightning, SetTransformerModelConfig

# Define union type for all model configs. Do not use `typing.Union` because of inconsistent behavior when inspecting
# the type of a variable with a union type.
ModelConfig = DRGNetModelConfig | PointNetModelConfig | SetTransformerModelConfig


__all__ = ["DRGNetLightning", "PointNetLightning", "ModelConfig"]


def get_model(config: ModelConfig) -> DRGNetLightning | PointNetLightning:
    """Return a LightningModule for the given config."""
    if isinstance(config, DRGNetModelConfig):
        return DRGNetLightning(config)
    elif isinstance(config, PointNetModelConfig):
        return PointNetLightning(config)
    elif isinstance(config, SetTransformerModelConfig):
        return SetTransformerLightning(config)
    elif isinstance(config, GATConfig):
        return GATLightning(config)
    else:
        raise ValueError(f"Unknown model config type {type(config)}")
