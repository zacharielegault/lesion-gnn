import dataclasses
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MLP, GINConv, global_mean_pool
from torch_sparse import SparseTensor

from lesion_gnn.utils.placeholder import Placeholder

from .base import BaseLightningModule, BaseModelConfig


class GIN(nn.Module):
    def __init__(self, input_features: int, hidden_channels: list[int], num_classes: int, dropout: float):
        super().__init__()
        assert all(d > 0 for d in hidden_channels)
        self.in_proj = nn.Linear(input_features, hidden_channels[0])
        self.convs = nn.ModuleList(
            [GINConv(MLP([d1, d2, d2], act="ELU", dropout=dropout)) for d1, d2 in pairwise(hidden_channels)]
        )
        self.out_proj = nn.Linear(hidden_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: LongTensor | SparseTensor, batch: LongTensor) -> Tensor:
        x = self.in_proj(x)
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.out_proj(x)
        return x


@dataclasses.dataclass(kw_only=True)
class GINConfig(BaseModelConfig):
    input_features: Placeholder[int] = dataclasses.field(default_factory=Placeholder, init=False)
    hidden_channels: list[int]
    dropout: float
    compile: bool
    name: str = dataclasses.field(default="GIN", init=False)


class GINLightning(BaseLightningModule):
    def __init__(self, config: GINConfig):
        super().__init__(config)
        model = GIN(
            input_features=config.input_features.value,
            hidden_channels=config.hidden_channels,
            num_classes=1 if self.is_regression else config.num_classes.value,
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
