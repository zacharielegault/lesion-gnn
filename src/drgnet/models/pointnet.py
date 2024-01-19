from dataclasses import dataclass

import torch
import torch_geometric
from torch import LongTensor, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius

from drgnet.utils.placeholder import Placeholder

from .base import BaseLightningModule, BaseModelConfig


class SAModule(torch.nn.Module):
    def __init__(self, ratio: float, radius: float, nn: torch.nn.Module) -> None:
        super().__init__()
        self.ratio = ratio
        self.r = radius
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x: Tensor, pos: Tensor, batch: LongTensor) -> tuple[Tensor, Tensor, LongTensor]:
        pos = pos.to(x.dtype)  # FIXME: This is a hack, needs to be fixed when pre-processing data.
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn: torch.nn.Module) -> None:
        super().__init__()
        self.nn = nn

    def forward(self, x: Tensor, pos: Tensor, batch: LongTensor) -> Tensor:
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        return x


class PointNet(torch.nn.Module):
    def __init__(self, input_features: int, pos_dim: int, num_classes: int):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([input_features + pos_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + pos_dim, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + pos_dim, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, num_classes], dropout=0.5, norm=None)

    def forward(self, x: Tensor, pos: Tensor, batch: LongTensor) -> Tensor:
        x, pos, batch = self.sa1_module(x, pos, batch)
        x, pos, batch = self.sa2_module(x, pos, batch)
        x = self.sa3_module(x, pos, batch)

        return self.mlp(x).log_softmax(dim=-1)


@dataclass
class PointNetModelConfig(BaseModelConfig):
    input_features: Placeholder[int]
    pos_dim: int


class PointNetLightning(BaseLightningModule):
    def __init__(self, config: PointNetModelConfig) -> None:
        super().__init__(config)
        model = PointNet(
            input_features=config.input_features.value,
            pos_dim=config.pos_dim,
            num_classes=1 if self.is_regression else config.num_classes.value,
        )
        self.model = torch_geometric.compile(model, dynamic=True) if compile else model

    def forward(self, data: Data) -> Tensor:
        logits = self.model(data.x, data.pos, data.batch)

        if self.is_regression:
            logits = torch.clamp(logits.squeeze(1), min=0, max=self.num_classes - 1)

        return logits
