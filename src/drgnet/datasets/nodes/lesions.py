from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import is_available as cuda_is_available
from torch_geometric.data import Data
from torch_geometric.utils import scatter


@dataclass
class LesionsArgs:
    which_features: Literal["decoder", "encoder"]
    feature_layer: int
    features_reduction: Literal["mean", "max"] = "mean"
    reinterpolation: Optional[Tuple[int, int]] = None

    def to_dict(self) -> dict:
        return asdict(self)


# def extract_features_by_cc(cc, features, nlabel, reduce=None):
#     if nlabel == 1:
#         return features.mean((2, 3))
#     cctorch = torch.nn.functional.one_hot(cc, num_classes=nlabel).squeeze().permute(2, 0, 1)
#     cctorch_flat = cctorch.flatten(1, 2)
#     cctorch_sum = cctorch_flat.sum(dim=1, keepdim=True)
#     cctorch_sparse = cctorch_flat.float().to_sparse()

#     features_flat = features.squeeze().flatten(1, 2).transpose(0, 1)
#     features_points = torch.sparse.mm(cctorch_sparse, features_flat) / cctorch_sum
#     return features_points


def extract_features_by_cc(cc, features, nlabel, reduce="mean"):
    if nlabel == 1:
        return features.mean((2, 3))
    features = features.squeeze(0).flatten(1, 2)  # (C, H*W)
    features = features.transpose(0, 1)  # (H*W, C)
    return scatter(features, cc.flatten(), 0, reduce=reduce)


class LesionsExtractor:
    def __init__(
        self,
        which_features: Literal["decoder", "encoder"] = "encoder",
        feature_layer: int = 3,
        features_reduction: Literal["mean", "max"] = "mean",
        compile=True,
        reinterpolation: Optional[Tuple[int, int]] = None,
    ):
        assert which_features in [
            "decoder",
            "encoder",
        ], f"which_features must be either 'decoder' or 'encoder', got {which_features}"
        self.which_features = which_features
        self.feature_layer = feature_layer
        self.device = torch.device("cuda" if cuda_is_available() else "cpu")
        self.features_reduction = features_reduction
        self.reinterpolation = reinterpolation
        if compile:
            # Does not seem to be faster and mode="reduced overhead" crashes after a few iterations
            self.extract_features_by_cc = torch.compile(extract_features_by_cc)
        else:
            self.extract_features_by_cc = extract_features_by_cc

    @torch.no_grad()
    def __call__(self, img_path: Path, label: int) -> Data:
        try:
            from fundus_lesions_toolkit.models import segment
        except ImportError:
            raise ImportError("Please install fundus-lesions-toolkit from the corresponding github repository")

        img = cv2.imread(str(img_path))
        assert img is not None
        img = img[:, :, ::-1]  # BGR to RGB but much faster than cvtColor (x7020)
        labelMap, fmap, decoder_fmap = segment(
            img,
            return_decoder_features=True,
            return_features=True,
            device=self.device,
            features_layer=self.feature_layer,
            reverse_autofit=False,
            compile=True,  # Marginally faster?
        )
        if self.which_features == "decoder":
            features = decoder_fmap

        elif self.which_features == "encoder":
            features = fmap
        # features = features.cpu()
        if self.reinterpolation is not None:
            features = F.interpolate(features, size=self.reinterpolation, mode="bilinear", align_corners=False)

        labelMap = F.adaptive_max_pool2d(labelMap, output_size=features.shape[2:])
        # Using max_pool to avoid losing lesions with interpolation (good practice?)
        labelMap = labelMap.to(features.device)
        Horg, Worg = labelMap.shape[-2:]

        predMapTensor = torch.argmax(labelMap, dim=1, keepdim=True)
        predMap = predMapTensor.squeeze().detach().byte().cpu().numpy()
        H, W = predMap.shape
        nlabel, cc, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
            predMap, connectivity=8, ltype=cv2.CV_16U, ccltype=1
        )
        ## Few remarks on connectedComponentsWithStatsWithAlgorithm ##
        # It does not support ltype=cv2.CV_16 unfortunately (casting needed to convert to tensor)
        # connectivity=8 is faster than connectivity=4 (x3.44)
        # ccltype=1 seems very very slightly faster than ccltype=0 or -1 (N=100 but same image, 0.391 vs 0.381 vs 0.375)

        # centroids are computed on the downsampled image, we rescale them (is it needed?)
        centroids = centroids * Horg / H

        features = torch.cat([features, predMapTensor], dim=1)  # We add the predicted lesion class to the point feature

        cctorch = torch.from_numpy(cc.astype(np.int64)).to(self.device)
        features_points = self.extract_features_by_cc(cctorch, features, nlabel, reduce=self.features_reduction)

        data = Data(
            pos=torch.from_numpy(centroids), y=torch.tensor([label]), name=img_path.stem, x=features_points.cpu()
        )
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(which_features={self.which_features}, feature_layer={self.feature_layer})"
