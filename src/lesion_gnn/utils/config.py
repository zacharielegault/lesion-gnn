import argparse
import dataclasses
import importlib
import os
import sys
import types
import typing
import warnings
from argparse import ArgumentParser
from types import NoneType

from lesion_gnn.datasets.datamodule import DataConfig
from lesion_gnn.models import ModelConfig
from lesion_gnn.utils.placeholder import Placeholder


@dataclasses.dataclass(kw_only=True)
class Config:
    dataset: DataConfig
    model: ModelConfig
    monitored_metric: str
    monitor_mode: str
    max_epochs: int
    seed: int
    project_name: str
    tag: str


def get_config(file_path: str | bytes | os.PathLike, module_name: str | None = None) -> Config:
    """Load a config file and return a config object.

    The config file must be a Python file that defines a `cfg` variable. The `module_name` argument is optional; if
    provided, the module will be registered in `sys.modules` under the given name. This allows the config file to be
    imported as a module, e.g. `import experiment_config`.

    Args:
        file_path: Path to the config file.
        module_name: Optional name to register the config module under.

    Returns:
        The config object.
    """

    name = module_name or "experiment_config"

    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if module_name is not None:
        sys.modules[name] = module

    return module.cfg


def parse_args() -> Config:
    """Parse the command line arguments and return a config object.

    The `--config` argument is required and must be a path to a Python file that defines a `cfg` variable that
    should be of type `Config`. The config file is parsed first, then the command line arguments are used to
    override the config file values.

    Returns:
        The config object.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to Python config file.", metavar="FILE", required=True)
    args, remaining_argv = parser.parse_known_args()
    config = get_config(args.config)

    if remaining_argv:
        warnings.warn("Overriding config values with command line arguments is disabled")
    # parser = _make_parser(config)
    # args = parser.parse_args(remaining_argv)
    # _override_config(config, args)

    return config


def _override_config(config, args: argparse.Namespace):
    # Set the config fields to the values passed in as command line arguments. Args for nested fields are passed in
    # as dot-separated strings, e.g. `--dataset.path /path/to/dataset` will set `config.dataset.path` to the given
    # path.
    for field_name, value in vars(args).items():
        if value is None:
            continue

        _set_config_field(config, field_name, value)


def _set_config_field(config, field_name: str, value: typing.Any):
    if "." in field_name:
        field_name, rest = field_name.split(".", 1)
        return _set_config_field(getattr(config, field_name), rest, value)

    setattr(config, field_name, value)


def _make_parser(config, parser: ArgumentParser | None = None, prefix: str = "") -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser()

    if prefix:
        prefix += "."  # Separate nested fields with a dot

    for field in dataclasses.fields(config):
        _dispatch(field.type, prefix + field.name, parser)

    return parser


def _dispatch(
    field_type: typing.Type,
    field_name: str,
    parser: ArgumentParser,
):
    if typing.get_origin(field_type) in (tuple, list):
        _dispatch(typing.get_args(field_type)[0], field_name, parser)
        # _add_maybe_existing_arg(parser, field_name, type=typing.get_args(field_type)[0], nargs="+")
    elif isinstance(field_type, types.UnionType) or isinstance(field_type, typing._UnionGenericAlias):
        if len(typing.get_args(field_type)) == 2 and typing.get_args(field_type)[1] == NoneType:
            _add_maybe_existing_arg(parser, field_name, type=typing.get_args(field_type)[0])
        elif all(dataclasses.is_dataclass(t) for t in typing.get_args(field_type)):
            for t in typing.get_args(field_type):
                _make_parser(t, parser, prefix=field_name)
        else:
            raise ValueError(f"Unsupported type {field_type}")
    elif dataclasses.is_dataclass(field_type):
        _make_parser(field_type, parser, prefix=field_name)
    elif isinstance(field_type, typing._GenericAlias):
        if typing.get_origin(field_type) == Placeholder:
            return
        else:
            raise ValueError(f"Unsupported type {field_type}")
    elif issubclass(field_type, (int, float, str, bool)):
        _add_maybe_existing_arg(parser, field_name, type=field_type)
    else:
        raise ValueError(f"Unsupported type {field_type}")


def _add_maybe_existing_arg(parser, dest, **kwargs):
    """Add an argument to the parser if it hasn't already been added."""
    if dest not in (action.dest for action in parser._actions):
        parser.add_argument("--" + dest, **kwargs)
