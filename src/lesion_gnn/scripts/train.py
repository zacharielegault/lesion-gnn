#! /usr/bin/env python

from lesion_gnn.training import train
from lesion_gnn.utils.config import parse_args


def main():
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
