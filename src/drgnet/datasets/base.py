import os.path as osp
import warnings
from pathlib import Path
from typing import Any, Callable, List, Literal, TypedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import is_available as cuda_is_available
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import _get_flattened_data_list
from torch_geometric.utils import scatter
from tqdm import tqdm


class SIFTArgs(TypedDict):
    num_keypoints: int
    sigma: float


class LESIONSArgs(TypedDict):
    which_features: Literal["decoder", "encoder"]
    feature_layer: int
    features_reduction: Literal["mean", "max"]


class BaseDataset(InMemoryDataset):
    def __init__(
        self,
        root: str | None = None,
        transform: Callable[..., Any] | None = None,
        log: bool = True,
        num_workers: int = 0,
        mode: Literal["SIFT", "LESIONS"] = "SIFT",
        **pre_transform_kwargs: SIFTArgs | LESIONSArgs,
    ):
        assert num_workers >= 0
        assert mode in ["SIFT", "LESIONS"]

        self.pre_transform_kwargs = pre_transform_kwargs
        self.num_workers = num_workers
        self.mode = mode

        pre_transform = (
            _SiftTransform(**pre_transform_kwargs) if self.mode == "SIFT" else _LesionsTransform(**pre_transform_kwargs)
        )

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=None,
            log=log,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> List[str]:
        """A list of files in the `processed_dir` which needs to be found in order to skip the processing."""
        return ["data.pt"]

    @property
    def processed_dir(self) -> str:
        if self.mode == "SIFT":
            return osp.join(
                self.root, f'processed_{self.dataset_name}_{self.mode}_{self.pre_transform_kwargs["num_keypoints"]}'
            )
        elif self.mode == "LESIONS":
            return osp.join(
                self.root,
                f'processed_{self.dataset_name}_{self.mode}_{self.pre_transform_kwargs["which_features"]}_{self.pre_transform_kwargs["feature_layer"]}',
            )
        else:
            return super().processed_dir

    @property
    def dataset_name(self) -> str:
        raise NotImplementedError("Please implement this property in your subclass.")

    @property
    def _diagnosis(self) -> Any:
        raise NotImplementedError("Please implement this property in your subclass.")

    @property
    def classes_counts(self) -> torch.Tensor:
        data_list = _get_flattened_data_list([data for data in self])
        y = torch.cat([data.y for data in data_list if "y" in data], dim=0)
        _, counts = torch.unique(y, return_counts=True)
        return counts

    def get_class_weights(
        self, mode: Literal["uniform", "inverse", "quadratic_inverse", "inverse_frequency"] = "inverse_frequency"
    ) -> torch.Tensor:
        counts = self.classes_counts
        if mode == "uniform":
            return torch.ones_like(counts)
        elif mode == "inverse":
            return 1 / counts
        elif mode == "quadratic_inverse":
            return 1 / counts**2
        elif mode == "inverse_frequency":
            n_samples = counts.sum()
            n_classes = len(counts)
            return n_samples / (n_classes * counts)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def process(self) -> None:
        """Process raw data and save it into the `processed_dir`."""

        graphs = []

        if self.num_workers == 0:
            for path, label in tqdm(self._path_and_label_generator(), total=len(self._diagnosis)):
                if label > 4:
                    continue
                data = self.pre_transform(path, label)
                graphs.append(data)
        else:
            raise NotImplementedError

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class _SiftTransform:
    def __init__(self, num_keypoints: int, sigma: float):
        """Load an image and extract SIFT keypoints.

        Args:
            num_keypoints (int): number of keypoints to extract
            sigma (float): sigma for SIFT's Gaussian filter
        """
        self.num_keypoints = num_keypoints
        self.sigma = sigma

    def __call__(self, img_path: Path, label: int) -> Data:
        """Load an image and extract SIFT keypoints.

        Args:
            img_path (Path): path to the image
            label (int): DR grade

        Returns:
            Data: a PyG `Data` object with the following attributes:
                - `x` (Tensor): SIFT descriptors (num_keypoints, 128)
                - `pos` (Tensor): (x, y) positions of the keypoints (num_keypoints, 2)
                - `score` (Tensor): SIFT scores (num_keypoints,)
                - `y` (Tensor): DR grade (1,)
                - `name` (str): image name
        """
        img = cv2.imread(str(img_path))
        assert img is not None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Turn off the thresholding and keep the top `num_keypoints` keypoints.
        sift = cv2.SIFT_create(nfeatures=self.num_keypoints, contrastThreshold=0, sigma=self.sigma)

        kpts, desc = sift.detectAndCompute(img, None)

        if len(kpts) == 0:
            warnings.warn(f"Image {img_path} has no keypoints.")
        elif len(kpts) < self.num_keypoints:
            warnings.warn(f"Image {img_path} has less than {self.num_keypoints} keypoints.")

        data = Data(
            x=torch.from_numpy(desc),
            pos=torch.from_numpy(np.array([kpt.pt for kpt in kpts])),  # (x, y)
            score=torch.from_numpy(np.array([kpt.response for kpt in kpts])),
            y=torch.tensor([label]),
            name=img_path.stem,
        )

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_keypoints={self.num_keypoints}, sigma={self.sigma})"


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


class _LesionsTransform:
    def __init__(
        self,
        which_features: Literal["decoder", "encoder"] = "encoder",
        feature_layer: int = 3,
        features_reduction: Literal["mean", "max"] = "mean",
        compile=True,
    ):
        assert which_features in [
            "decoder",
            "encoder",
        ], f"which_features must be either 'decoder' or 'encoder', got {which_features}"
        self.which_features = which_features
        self.feature_layer = feature_layer
        self.device = torch.device("cuda" if cuda_is_available() else "cpu")
        self.features_reduction = features_reduction
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

        if features.shape[2] > 512:
            features = F.interpolate(features, size=(512, 512), mode="bilinear", align_corners=False)
            # 1536x1536  was just too big -> 512x512

        Horg, Worg = labelMap.shape[-2:]
        # Open question: should we match (downsample) image resolution with features resolution or the reverse
        # (upsample features resolution)?

        # Using former for now (more efficient memory wise)
        labelMap = F.adaptive_max_pool2d(labelMap, output_size=features.shape[2:])
        # Using max_pool to avoid losing lesions with interpolation (good practice?)

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
