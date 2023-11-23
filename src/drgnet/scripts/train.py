#! /usr/bin/env python

import argparse
from pathlib import Path

from drgnet.training import train
from drgnet.utils import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = Config.parse_yaml(config_path)
    train(config)


if __name__ == "__main__":
    main()
