import albumentations as A
import numpy as np


class FundusAutocrop(A.DualTransform):
    """Crops the black border from an RGB fundus image.

    Args:
        threshold (float, optional): The threshold for the red channel. Defaults to 25.0.
        always_apply (bool, optional): Whether to always apply the transform. Defaults to False.
        p (float, optional): The probability of applying the transform. Defaults to 1.0.
    """

    def __init__(self, threshold: float = 25.0, always_apply: bool = False, p: float = 1.0) -> None:
        super().__init__(always_apply, p)
        self.threshold = threshold

    def apply(self, image: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int, **params) -> np.ndarray:
        return image[y_min:y_max, x_min:x_max]

    def apply_to_mask(self, mask: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int, **params) -> np.ndarray:
        return mask[y_min:y_max, x_min:x_max]

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params) -> dict[str, int]:
        image: np.ndarray = params["image"]  # (H, W, 3)
        red = image[:, :, 0]
        fundus_mask = red > self.threshold
        not_null_pixels = np.argwhere(fundus_mask)

        if not not_null_pixels.size:
            return {"x_min": 0, "x_max": image.shape[1], "y_min": 0, "y_max": image.shape[0]}

        x_min = np.min(not_null_pixels[:, 1])
        x_max = np.max(not_null_pixels[:, 1])
        y_min = np.min(not_null_pixels[:, 0])
        y_max = np.max(not_null_pixels[:, 0])

        if (x_min == x_max) or (y_min == y_max):
            return {"x_min": 0, "x_max": image.shape[1], "y_min": 0, "y_max": image.shape[0]}

        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("threshold",)
