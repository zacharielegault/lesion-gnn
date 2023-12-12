import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class SiftArgs:
    num_keypoints: int
    sigma: float

    def to_dict(self) -> dict:
        return asdict(self)


class SiftExtractor:
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
