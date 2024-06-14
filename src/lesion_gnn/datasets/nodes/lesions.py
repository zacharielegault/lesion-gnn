import dataclasses
import functools
from enum import Enum
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
import typing_extensions
from albumentations.pytorch import ToTensorV2
from fundus_lesions_toolkit.models import segment
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.cuda import is_available as cuda_is_available
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from fundus_datamodules.utils import FundusAutocrop


@dataclasses.dataclass(kw_only=True)
class SegmentationEncoderFeatures:
    layer: int


@dataclasses.dataclass(kw_only=True)
class SegmentationDecoderFeatures:
    pass


@dataclasses.dataclass(kw_only=True)
class TimmEncoderFeatures:
    timm_model: str
    layer: int


FeatureSource = SegmentationEncoderFeatures | SegmentationDecoderFeatures | TimmEncoderFeatures
"""
Features can come from
- the segmentation model's encoder, at a given layer
- the segmentation model's decoder, just before the final classification layer
- another encoder (e.g. a resnet, a vit, etc.), at a given layer
This with would be great to model with proper sum type/tagged union, but Python is not great in that regard. We
instead use a union of dataclasses, and match on the type of the object.
>>> import typing_extensions
>>> feature_source = SegmentationEncoderFeatures(layer=4)
>>> match feature_source:
...     case SegmentationEncoderFeatures(layer):
...         print(f"Segmentation encoder at layer {layer}")
...     case SegmentationDecoderFeatures():
...         print("Segmentation decoder")
...     case TimmEncoderFeatures(timm_model, layer):
...         print(f"Timm encoder {timm_model} at layer {layer}")
...     case _:
...         typing_extensions.assert_never(feature_source)
Segmentation encoder at layer 4
"""


class FeaturesReduction(str, Enum):
    MEAN = "mean"
    MAX = "max"


@dataclasses.dataclass(kw_only=True)
class LesionsNodesConfig:
    feature_source: FeatureSource
    features_reduction: FeaturesReduction = FeaturesReduction.MEAN
    reinterpolation: tuple[int, int] | None = None
    compile: bool = True


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
        config: LesionsNodesConfig,
    ):
        self.feature_source = config.feature_source
        self.device = torch.device("cuda" if cuda_is_available() else "cpu")
        self.features_reduction = FeaturesReduction(config.features_reduction)
        self.reinterpolation = config.reinterpolation
        if config.compile:
            # Does not seem to be faster and mode="reduced overhead" crashes after a few iterations
            self.extract_features_by_cc = torch.compile(extract_features_by_cc)
        else:
            self.extract_features_by_cc = extract_features_by_cc

    @torch.no_grad()
    def __call__(self, img_path: Path, label: int) -> Data:
        img_np = cv2.imread(str(img_path))
        if img_np is None:
            raise RuntimeError(f"Could not read image {img_path}")

        img_np = img_np[:, :, ::-1]  # BGR to RGB but much faster than cvtColor (x7020)

        segment_fn = functools.partial(segment, image=img_np, device=self.device, reverse_autofit=False, compile=True)
        match self.feature_source:
            case SegmentationEncoderFeatures(layer=layer):
                label_map, features = segment_fn(return_features=True, features_layer=layer)
            case SegmentationDecoderFeatures():
                label_map, features = segment_fn(return_decoder_features=True)
            case TimmEncoderFeatures(timm_model=timm_model, layer=layer):
                label_map = segment_fn()
                # FIXME: `segment` returns a 3D tensor if both return_features and return_decoder_features are False
                # but a 4D tensor otherwise. Unsqueeze for consistency with the other cases
                label_map = label_map.unsqueeze(0)
                encoder = self._get_timm_encoder(timm_model, layer)
                # FIXME: this is kind of brittle, we should have a better way to know which normalization to use
                transforms = A.Compose(
                    [
                        FundusAutocrop(),
                        A.LongestMaxSize(max_size=512),
                        A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
                        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                        ToTensorV2(),
                    ]
                )
                img_torch = transforms(image=img_np)["image"]
                (features,) = encoder(img_torch.unsqueeze(0).to(self.device))
            case _:
                typing_extensions.assert_never()

        # features = features.cpu()
        if self.reinterpolation is not None:
            features = F.interpolate(features, size=self.reinterpolation, mode="bilinear", align_corners=False)

        label_map = torch.argmax(label_map, dim=1, keepdim=True)
        Horg, Worg = label_map.shape[-2:]
        label_map = F.adaptive_max_pool2d(label_map.float(), output_size=features.shape[2:])
        connected_map = label_map.squeeze().detach().byte().cpu().numpy()

        H, W = label_map.shape[-2:]
        # Using max_pool to avoid losing lesions with interpolation (good practice?)
        label_map = label_map.to(features.device)
        nlabel, cc, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
            connected_map, connectivity=8, ltype=cv2.CV_16U, ccltype=1
        )
        ## Few remarks on connectedComponentsWithStatsWithAlgorithm ##
        # It does not support ltype=cv2.CV_16 unfortunately (casting needed to convert to tensor)
        # connectivity=8 is faster than connectivity=4 (x3.44)
        # ccltype=1 seems very very slightly faster than ccltype=0 or -1 (N=100 but same image, 0.391 vs 0.381 vs 0.375)

        # centroids are computed on the downsampled image, we rescale them (is it needed?)
        centroids = centroids * Horg / H

        features = torch.cat([features, label_map], dim=1)  # We add the predicted lesion class to the point feature

        cctorch = torch.from_numpy(cc.astype(np.int64)).to(self.device)
        features_points = self.extract_features_by_cc(cctorch, features, nlabel, reduce=self.features_reduction)

        data = Data(
            pos=torch.from_numpy(centroids), y=torch.tensor([label]), name=img_path.stem, x=features_points.cpu()
        )
        return data

    @functools.lru_cache(maxsize=1)
    def _get_timm_encoder(self, timm_model: str, layer: int):
        encoder = timm.create_model(timm_model, pretrained=True, features_only=True, out_indices=(layer,))
        encoder = encoder.to(self.device)
        return encoder

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(feature_source={self.feature_source})"
