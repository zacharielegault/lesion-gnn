import dataclasses
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_sparse import SparseTensor

from lesion_gnn.utils.placeholder import Placeholder

from .base import BaseLightningModule, BaseModelConfig


class GAT(nn.Module):
    def __init__(self, input_features: int, hiddden_channels: list[int], num_classes: int, heads: int, dropout: float):
        super().__init__()
        assert all(d % heads == 0 for d in hiddden_channels)
        self.in_proj = nn.Linear(input_features, hiddden_channels[0])
        self.convs = nn.ModuleList(
            [GATConv(d1, d2 // heads, heads=heads, dropout=dropout) for d1, d2 in pairwise(hiddden_channels)]
        )
        self.out_proj = nn.Linear(hiddden_channels[-1], num_classes)

    def forward(self, x: Tensor, edge_index: LongTensor | SparseTensor, batch: LongTensor) -> Tensor:
        x = self.in_proj(x)
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.out_proj(x)
        return x


@dataclasses.dataclass(kw_only=True)
class GATConfig(BaseModelConfig):
    input_features: Placeholder[int] = dataclasses.field(default_factory=Placeholder, init=False)
    hiddden_channels: list[int]
    heads: int
    dropout: float
    compile: bool
    name: str = dataclasses.field(default="GAT", init=False)


class GATLightning(BaseLightningModule):
    def __init__(self, config: GATConfig):
        super().__init__(config)
        model = GAT(
            input_features=config.input_features.value,
            hiddden_channels=config.hiddden_channels,
            num_classes=1 if self.is_regression else config.num_classes.value,
            heads=config.heads,
            dropout=config.dropout,
        )
        self.model = torch.compile(model, dynamic=True) if config.compile else model

    def forward(self, data: Data) -> Tensor:
        if hasattr(data, "adj_t"):
            edge_index = data.adj_t
        else:
            edge_index = data.edge_index

        logits = self.model(data.x, edge_index, data.batch)

        if self.is_regression:
            logits = torch.clamp(logits.squeeze(1), min=0, max=self.num_classes - 1)

        return logits
