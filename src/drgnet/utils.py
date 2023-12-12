from pathlib import Path
from typing import Optional, Tuple

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    dataset: "DatasetConfig"
    model: "ModelConfig"
    batch_size: int
    max_epochs: int
    seed: int
    project_name: str
    tag: str

    @classmethod
    def parse_yaml(cls, path: Path) -> "Config":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


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


class ModelConfig(BaseModel):
    gnn_hidden_dim: int
    num_layers: int
    sortpool_k: int
    conv_hidden_dims: Tuple[int, int]
    compile: bool
    lr: Optional[float] = 0.001
    weight_decay: Optional[float] = 0.01
    optimizer_algo: Optional[str] = "adamw"
    loss_type: Optional[str] = "CE"
