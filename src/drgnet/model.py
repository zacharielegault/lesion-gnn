from itertools import pairwise

import lightning as L
import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MLP, GraphConv, SortAggregation
from torch_sparse import SparseTensor
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, CohenKappa, F1Score, Precision, Recall

from drgnet.metrics import (
    ReferableDRAccuracy,
    ReferableDRAUROC,
    ReferableDRAveragePrecision,
    ReferableDRF1,
    ReferableDRPrecision,
    ReferableDRRecall,
)


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


class DRGNetLightning(L.LightningModule):
    def __init__(
        self,
        input_features: int,
        gnn_hidden_dim: int,
        num_layers: int,
        sortpool_k: int,
        num_classes: int,
        conv_hidden_dims: tuple[int, int] = (16, 32),
        compile: bool = False,
    ) -> None:
        super().__init__()
        model = DRGNet(
            input_features=input_features,
            gnn_hidden_dim=gnn_hidden_dim,
            num_layers=num_layers,
            sortpool_k=sortpool_k,
            num_classes=num_classes,
            conv_hidden_dims=conv_hidden_dims,
        )
        self.model = torch_geometric.compile(model, dynamic=True) if compile else model
        self.criterion = nn.CrossEntropyLoss()
        self.multiclass_metrics = MetricCollection(
            {
                "micro_acc": Accuracy(task="multiclass", num_classes=num_classes, average="micro"),
                "kappa": CohenKappa(task="multiclass", num_classes=num_classes, weights="quadratic"),
                "macro_f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                "macro_precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
                "macro_recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
            },
            prefix="val_",
        )

        self.referable_metrics = MetricCollection(
            {
                "acc": ReferableDRAccuracy(),
                "auroc": ReferableDRAUROC(),
                "auprc": ReferableDRAveragePrecision(),
                "f1": ReferableDRF1(),
                "precision": ReferableDRPrecision(),
                "recall": ReferableDRRecall(),
            },
            prefix="val_",
        )

    def forward(
        self, x: Tensor, edge_index: Tensor | SparseTensor, batch: Tensor, edge_weight: Tensor | None = None
    ) -> Tensor:
        return self.model(x, edge_index, batch, edge_weight)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        if hasattr(batch, "adj_t"):
            edge_index = batch.adj_t
        else:
            edge_index = batch.edge_index

        logits = self.model(batch.x, edge_index, batch.batch, batch.edge_weight)
        loss = self.criterion(logits, batch.y)
        self.log("train_loss", loss, batch_size=batch.num_graphs, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        if hasattr(batch, "adj_t"):
            edge_index = batch.adj_t
        else:
            edge_index = batch.edge_index

        logits = self(batch.x, edge_index, batch.batch, batch.edge_weight)
        loss = self.criterion(logits, batch.y)
        self.log("val_loss", loss, batch_size=batch.num_graphs)
        self.log_dict(
            self.multiclass_metrics(logits, batch.y), batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )
        self.log_dict(
            self.referable_metrics(logits, batch.y), batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)
