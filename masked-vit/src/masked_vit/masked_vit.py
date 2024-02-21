import warnings
from typing import Literal

import timm
import torch
import torch.nn.functional as F
from fundus_lesions_toolkit.models import batch_segment
from timm.models._manipulate import checkpoint_seq
from timm.models.vision_transformer import VisionTransformer as TimmViT


class MaskedVisionTransformer(TimmViT):
    def forward_features(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        mode: Literal["mask", "index", "nan_replace"] = "mask",
        flip_mask: bool = False,
        compute_masks: bool = False,
        keep_p: float = 0.0,
    ) -> torch.Tensor:
        if compute_masks and mask is not None:
            warnings.warn("compute_masks is True, but mask is not None. Ignoring compute_masks.")
            compute_masks = False

        if compute_masks:
            mask = batch_segment(x, already_normalized=True, device=x.device)
            mask = mask.argmax(dim=1)

        x = self.patch_embed(x)  # (B, C, H, W) -> (B, L, D)
        x = self._pos_embed(x)  # (B, L, D) -> (B, L+1, D)
        x = self.patch_drop(x)  # (B, L+1, D) -> (B, L+1, D)

        if mask is not None:
            mask = mask > 0  # TODO: keep specific classes
            patch_size = self.patch_embed.patch_size
            # Downsample mask to patch size. Need to cast `mask` because max_pool2d does not support bool
            mask = F.adaptive_max_pool2d(
                mask.to(x.dtype), (mask.shape[-2] // patch_size[-2], mask.shape[-1] // patch_size[-1])
            ).flatten(1)  # (B, H, W) -> (B, L)

            if keep_p > 0:
                # Keep a proportion `keep_p` of the background
                ...

            if flip_mask:
                mask = 1 - mask

            # keep class token untouched
            mask = torch.cat([mask.new_ones((mask.shape[0], 1)), mask], dim=-1)  # (B, L) -> (B, L+1)

            if mode == "mask":
                x = x * mask.unsqueeze(-1)
            elif mode == "nan_replace":
                warnings.warn("mode nan_replace does not work")
                mask = mask.bool().unsqueeze(-1).expand(-1, -1, x.shape[-1])
                x = torch.where(mask, x, torch.nan)
            elif mode == "neg_inf_replace":
                warnings.warn("mode neg_inf_replace does not work")
                mask = mask.bool().unsqueeze(-1).expand(-1, -1, x.shape[-1])
                x = torch.where(mask, x, -torch.inf)
            elif mode == "index":
                idx = mask.nonzero(as_tuple=True)  # get the indices of tokens to keep
                x_ = x.new_zeros(x.shape[0], idx[1].max() + 1, x.shape[-1])
                x_[idx] = x[idx]
                x = x_
            else:
                raise ValueError(f"Unknown mode {mode}")

        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        mode: Literal["mask", "index"] = "mask",
        flip_mask: bool = False,
        compute_masks: bool = False,
        keep_p: float = 0.0,
    ) -> torch.Tensor:
        x = self.forward_features(x, mask, mode, flip_mask, compute_masks, keep_p)
        x = self.forward_head(x)
        return x


# Monkey patch timm to use our VisionTransformer
timm.models.vision_transformer.VisionTransformer = MaskedVisionTransformer


def make_model() -> MaskedVisionTransformer:
    path = "cosmic-capybara-16.ckpt"
    state_dict = torch.load(path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k.replace("model.", "")] = state_dict.pop(k)

    model = timm.create_model("vit_base_patch16_384", num_classes=1, img_size=(1024, 1024))
    print(model.load_state_dict(state_dict))
    return model
