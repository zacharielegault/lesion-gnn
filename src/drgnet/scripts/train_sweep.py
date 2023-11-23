#! /usr/bin/env python

import argparse
from pathlib import Path

from drgnet.training import train
from drgnet.utils import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--gnn_hidden_dim", type=int, default=64, help="GNN hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers.")
    parser.add_argument("--sortpool_k", type=int, default=10, help="Sortpooling k.")
    parser.add_argument("--distance_sigma_px", type=float, default=15, help="Distance sigma in pixels.")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer.")

    args = parser.parse_args()
    config_path = Path(args.config)
    config = Config.parse_yaml(config_path)

    config.model.lr = args.lr
    config.model.weight_decay = args.weight_decay
    config.model.gnn_hidden_dim = args.gnn_hidden_dim
    config.model.num_layers = args.num_layers
    config.model.sortpool_k = args.sortpool_k
    config.dataset.distance_sigma_px = args.distance_sigma_px
    config.model.optimizer_algo = args.optimizer

    train(config)

if __name__ == "__main__":
    main()
