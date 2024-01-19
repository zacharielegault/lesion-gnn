#! /usr/bin/env python

from drgnet.training import train
from drgnet.utils import parse_args


def main():
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
