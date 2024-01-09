import argparse
import typing
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple, Union

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

    @classmethod
    def parse(cls) -> "Config":
        config_parser = ArgumentParser()
        config_parser.add_argument("--config", type=str, help="Path to YAML config file.", metavar="FILE")
        args, argv = config_parser.parse_known_args()

        config_path = Path(args.config)
        config = Config.parse_yaml(config_path)

        parser = _make_parser(config)
        args = parser.parse_args(argv)
        _override_config(config, args)

        return config


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


def _override_config(config: BaseModel, args: argparse.Namespace):
    # Set the config fields to the values passed in as command line arguments. Args for nested fields are passed in
    # as dot-separated strings, e.g. `--dataset.path /path/to/dataset` will set `config.dataset.path` to the given
    # path.
    for field_name, value in vars(args).items():
        if value is None:
            continue

        _set_config_field(config, field_name, value)


def _set_config_field(config: BaseModel, field_name: str, value: typing.Any):
    if "." in field_name:
        field_name, rest = field_name.split(".", 1)
        return _set_config_field(getattr(config, field_name), rest, value)

    setattr(config, field_name, value)


def _make_parser(config: BaseModel, parser: Optional[ArgumentParser] = None, prefix: str = "") -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser()

    if prefix:
        prefix += "."  # Separate nested fields with a dot

    for field_name, field_info in config.model_fields.items():
        if field_info.annotation in (int, float, str, bool):
            parser.add_argument(f"--{prefix}{field_name}", type=field_info.annotation)
        elif typing.get_origin(field_info.annotation) in (tuple, list):
            parser.add_argument(f"--{prefix}{field_name}", type=typing.get_args(field_info.annotation)[0], nargs="+")
        elif typing.get_origin(field_info.annotation) == Union:
            assert len(typing.get_args(field_info.annotation)) == 2
            assert typing.get_args(field_info.annotation)[1] == type(None)
            parser.add_argument(f"--{prefix}{field_name}", type=typing.get_args(field_info.annotation)[0])
        elif issubclass(field_info.annotation, BaseModel):
            _make_parser(field_info.annotation, parser, prefix=field_name)
        else:
            raise ValueError(f"Unsupported type {field_info.annotation}")

    return parser
