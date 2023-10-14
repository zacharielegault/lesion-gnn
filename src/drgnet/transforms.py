import math
import warnings
from enum import Enum

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class SaveAs(Enum):
    EDGE_WEIGHT_REPLACE = "edge_weight_replace"
    EDGE_ATTR_CAT = "edge_attr_cat"
    EDGE_ATTR_REPLACE = "edge_attr_replace"


class GaussianDistance(BaseTransform):
    def __init__(
        self,
        sigma: float,
        save_as: SaveAs = SaveAs.EDGE_WEIGHT_REPLACE,
        dtype: torch.dtype = torch.float32,
    ):
        """Gaussian weighted distance transform.

        Args:
            sigma (float): Standard deviation of the Gaussian kernel.
            cat (bool, optional): If set to `False`, all existing edge attributes will be replaced. Defaults to `True`.
        """
        self.sigma = sigma
        self._norm_const = math.sqrt(2 * math.pi * sigma**2)
        self.save_as = save_as
        self.dtype = dtype

    def __call__(self, data: Data) -> Data:
        if data.edge_index.numel() == 0:
            warnings.warn("The graph has no edges, returning the original data object.")
            return data

        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        sq_dist = (pos[row] - pos[col]).pow(2).sum(dim=-1)
        dist = torch.exp(-sq_dist / (2 * self.sigma**2)) / self._norm_const
        dist = dist.to(self.dtype)

        match self.save_as:
            case SaveAs.EDGE_WEIGHT_REPLACE:
                data.edge_weight = dist
            case SaveAs.EDGE_ATTR_CAT:
                dist = dist.view(-1, 1)

                if pseudo is not None:
                    pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                    data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
                else:
                    data.edge_attr = dist
            case SaveAs.EDGE_ATTR_REPLACE:
                dist = dist.view(-1, 1)
                data.edge_attr = dist

        return data

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(sigma={self.sigma})"
