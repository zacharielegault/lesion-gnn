from itertools import pairwise

import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MLP, GraphConv, SortAggregation
from torch_sparse import SparseTensor

from .base import BaseLightningModule, BaseModelConfig


class DRGNet(nn.Module):
    def __init__(
        self,
        input_features: int,
        gnn_hidden_dim: int,
        num_layers: int,
        sortpool_k: int,
        num_classes: int,
        conv_hidden_dims: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__()

        # GNN layers
        gnn_dims = [input_features] + [gnn_hidden_dim] * num_layers
        self.graph_convs = nn.ModuleList(
            [GraphConv(in_channels, out_channels) for in_channels, out_channels in pairwise(gnn_dims)]
        )
        self.graph_convs.append(GraphConv(gnn_hidden_dim, 1))
        total_latent_dim = gnn_hidden_dim * num_layers + 1

        # Sort pooling
        self.sort_pool = SortAggregation(sortpool_k)

        # Conv layers
        kernel_size = 5  # Kernel size for the second conv layer
        self.conv1 = nn.Conv1d(1, conv_hidden_dims[0], kernel_size=total_latent_dim, stride=total_latent_dim)
        self.max_pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(conv_hidden_dims[0], conv_hidden_dims[1], kernel_size=kernel_size, stride=1)

        # MLP
        dense_dim = int((sortpool_k - 2) / 2 + 1)  # Convolution math to compute output size
        dense_dim = (dense_dim - kernel_size + 1) * conv_hidden_dims[1]
        self.mlp = MLP([dense_dim, 128, num_classes], dropout=0.5, norm=None, act=F.elu)

    def forward(
        self, x: Tensor, edge_index: Tensor | SparseTensor, batch: Tensor, edge_weight: Tensor | None = None
    ) -> Tensor:
        xs = []
        for graph_conv in self.graph_convs:
            x = F.elu(graph_conv(x, edge_index, edge_weight))
            xs.append(x)
        x_cat = torch.cat(xs, dim=1)  # (num_nodes, hidden_dim * num_layers)

        x = self.sort_pool(x_cat, batch)  # (num_graphs, hidden_dim * num_layers * sortpool_k)
        x = x.unsqueeze(1)  # (num_graphs, 1, (hidden_dim * num_layers + 1) * sortpool_k)

        x = F.elu(self.conv1(x))
        x = self.max_pool(x)
        x = F.elu(self.conv2(x))

        x = x.view(x.size(0), -1)  # (num_graphs, dense_dim)
        logits = self.mlp(x)  # (num_graphs, num_classes)

        return logits


class DRGNetModelConfig(BaseModelConfig):
    name: str = "DRGNet"
    input_features: int
    gnn_hidden_dim: int
    num_layers: int
    sortpool_k: int
    conv_hidden_dims: tuple[int, int] = (16, 32)
    compile: bool = False


class DRGNetLightning(BaseLightningModule):
    def __init__(self, config: DRGNetModelConfig) -> None:
        super().__init__(config)

        model = DRGNet(
            input_features=config.input_features,
            gnn_hidden_dim=config.gnn_hidden_dim,
            num_layers=config.num_layers,
            sortpool_k=config.sortpool_k,
            num_classes=1 if self.is_regression else config.num_classes,
            conv_hidden_dims=config.conv_hidden_dims,
        )
        self.model = torch_geometric.compile(model, dynamic=True) if compile else model

    def forward(self, data: Data) -> Tensor:
        if hasattr(data, "adj_t"):
            edge_index = data.adj_t
        else:
            edge_index = data.edge_index

        logits = self.model(data.x, edge_index, data.batch, data.edge_weight)

        if self.is_regression:
            logits = torch.clamp(logits.squeeze(1), min=0, max=self.num_classes - 1)

        return logits
