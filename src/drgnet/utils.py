import argparse
import typing
from argparse import ArgumentParser
from pathlib import Path
from types import NoneType
from typing import Optional, Union

import yaml
from pydantic import BaseModel

from drgnet.datasets import DatasetConfig
from drgnet.models import ModelConfig


class BaseModelWithParser(BaseModel):
    @classmethod
    def parse(cls) -> "BaseModelWithParser":
        """Parse the command line arguments and return a config object.

        The `--config` argument is required and must be a path to a YAML config file. All other arguments are optional.
        The config file is parsed first, then the command line arguments are used to override the config file values.
        """
        parser = ArgumentParser()
        parser.add_argument("--config", type=str, help="Path to YAML config file.", metavar="FILE")
        parser = _make_parser(cls, parser)
        args = parser.parse_args()

        config = Config.from_yaml(Path(args.config))
        del args.config  # Remove the config file path from the args because it's not a field in the config model
        _override_config(config, args)

        return config

    @classmethod
    def from_yaml(cls, path: Path) -> "BaseModelWithParser":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


class Config(BaseModelWithParser):
    """Whenever this model is updated, the JSON schema should be updated as well.

    To generate a JSON schema for this model, run:
    >>> import json
    >>> from drgnet.utils import Config
    >>> print(json.dumps(Config.model_json_schema(), indent=4))

    To validate a config file against the schema, run:
    >>> import jsonschema
    >>> from drgnet.utils import Config
    >>> with open("config.yaml", "r") as f:
    ...     config = yaml.safe_load(f)
    >>> jsonschema.validate(config, Config.model_json_schema())
    """

    dataset: DatasetConfig
    model: ModelConfig
    batch_size: int
    max_epochs: int
    seed: int
    project_name: str
    tag: str


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
        if field_info.annotation in (int, float, str, bool) or issubclass(field_info.annotation, str):
            parser.add_argument(f"--{prefix}{field_name}", type=field_info.annotation)
        elif typing.get_origin(field_info.annotation) in (tuple, list):
            parser.add_argument(f"--{prefix}{field_name}", type=typing.get_args(field_info.annotation)[0], nargs="+")
        elif typing.get_origin(field_info.annotation) == Union:
            if (
                len(typing.get_args(field_info.annotation)) == 2
                and typing.get_args(field_info.annotation)[1] == NoneType
            ):
                parser.add_argument(f"--{prefix}{field_name}", type=typing.get_args(field_info.annotation)[0])
            elif all(issubclass(t, BaseModel) for t in typing.get_args(field_info.annotation)):
                for t in typing.get_args(field_info.annotation):
                    _make_parser(t, parser, prefix=field_name)
        elif issubclass((t := typing._eval_type(field_info.annotation, globals(), locals())), BaseModel):
            # Evaluate the type annotation to a concrete type, then check if it's a subclass of BaseModel
            _make_parser(t, parser, prefix=field_name)
        elif issubclass(field_info.annotation, BaseModel):
            _make_parser(field_info.annotation, parser, prefix=field_name)
        else:
            raise ValueError(f"Unsupported type {field_info.annotation}")

    return parser
