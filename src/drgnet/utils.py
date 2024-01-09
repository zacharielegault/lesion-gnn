from __future__ import annotations

import argparse
import typing
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple, Union

import yaml
from pydantic import BaseModel


class BaseModelWithParser(BaseModel):
    @classmethod
    def parse(cls) -> BaseModelWithParser:
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, help="Path to YAML config file.", metavar="FILE")
        parser = _make_parser(cls, parser)
        args = parser.parse_args()

        config = Config.parse_yaml(Path(args.config))
        del args.config  # Remove the config file path from the args because it's not a field in the config model
        _override_config(config, args)

        return config

    @classmethod
    def parse_yaml(cls, path: Path) -> BaseModelWithParser:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


class Config(BaseModelWithParser):
    dataset: DatasetConfig
    model: ModelConfig
    batch_size: int
    max_epochs: int
    seed: int
    project_name: str
    tag: str


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


def _make_parser(config: type(BaseModel), parser: Optional[ArgumentParser] = None, prefix: str = "") -> ArgumentParser:
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
        elif issubclass((t := typing._eval_type(field_info.annotation, globals(), locals())), BaseModel):
            _make_parser(t, parser, prefix=field_name)
        else:
            raise ValueError(f"Unsupported type {field_info.annotation}")

    return parser
