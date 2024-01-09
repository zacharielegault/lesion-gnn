#! /usr/bin/env python

from drgnet.training import train
from drgnet.utils import Config


def main():
    config = Config.parse()
    train(config)


if __name__ == "__main__":
    main()
