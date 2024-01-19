#! /usr/bin/env python

from drgnet.training import train
from drgnet.utils.config import parse_args


def main():
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
