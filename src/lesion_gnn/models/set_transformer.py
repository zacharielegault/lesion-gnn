import dataclasses

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.aggr.utils import InducedSetAttentionBlock as ISAB
from torch_geometric.nn.aggr.utils import PoolingByMultiheadAttention as PMA
from torch_geometric.nn.aggr.utils import SetAttentionBlock as SAB
from torch_geometric.utils import to_dense_batch

from lesion_gnn.models.base import BaseLightningModule, BaseModelConfig
from lesion_gnn.utils.placeholder import Placeholder


class SetTransformer(nn.Module):
    def __init__(
        self,
        input_features: int,
        inner_dim: int,
        num_classes: int,
        num_inducing_points: int = 1,
        num_seed_points: int = 1,
        num_encoder_blocks: int = 1,
        num_decoder_blocks: int = 1,
        heads: int = 1,
        concat: bool = True,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.concat = concat

        self.in_proj = nn.Linear(input_features, inner_dim)
        self.encoders = nn.ModuleList(
            [ISAB(inner_dim, num_inducing_points, heads, layer_norm, dropout) for _ in range(num_encoder_blocks)]
        )
        self.pma = PMA(inner_dim, num_seed_points, heads, layer_norm, dropout)
        self.decoders = nn.ModuleList([SAB(inner_dim, heads, layer_norm, dropout) for _ in range(num_decoder_blocks)])
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        for encoder in self.encoders:
            encoder.reset_parameters()
        self.pma.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        x = self.in_proj(x)

        x, mask = to_dense_batch(x, batch)

        for encoder in self.encoders:
            x = encoder(x, mask)

        x = self.pma(x, mask)

        for decoder in self.decoders:
            x = decoder(x)

        x = x.flatten(1, 2) if self.concat else x.mean(dim=1)

        return self.out_proj(x)


@dataclasses.dataclass(kw_only=True)
class SetTransformerModelConfig(BaseModelConfig):
    input_features: Placeholder[int] = dataclasses.field(default_factory=Placeholder, init=False)
    inner_dim: int
    num_inducing_points: int
    num_seed_points: int
    num_encoder_blocks: int
    num_decoder_blocks: int
    heads: int
    layer_norm: bool
    dropout: float
    compile: bool = False
    name: str = dataclasses.field(default="SetTransformer", init=False)


class SetTransformerLightning(BaseLightningModule):
    def __init__(self, config: SetTransformerModelConfig) -> None:
        super().__init__(config)
        model = SetTransformer(
            input_features=config.input_features.value,
            inner_dim=config.inner_dim,
            num_classes=1 if self.is_regression else config.num_classes.value,
            num_inducing_points=config.num_inducing_points,
            num_seed_points=config.num_seed_points,
            num_encoder_blocks=config.num_encoder_blocks,
            num_decoder_blocks=config.num_decoder_blocks,
            heads=config.heads,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )
        self.model = torch.compile(model, dynamic=True) if config.compile else model

    def forward(self, data: Data) -> Tensor:
        logits = self.model(data.x, data.batch)

        if self.is_regression:
            logits = torch.clamp(logits.squeeze(1), min=0, max=self.num_classes - 1)

        return logits
