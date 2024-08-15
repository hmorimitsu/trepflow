from functools import partial
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_timm.norm import LayerNorm2d
from .multi_stats_bn import MultiStatsBatchNorm2d


def coords_grid(b, h, w, homogeneous=False, dtype=None, device=None):
    y, x = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij",
    )  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0)  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    return grid


def global_correlation_softmax(
    feature0,
    feature1,
    pred_bidir_flow=False,
):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (
        c**0.5
    )  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(
        b, h, w, dtype=correlation.dtype, device=correlation.device
    )  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

    if pred_bidir_flow:
        correlation = torch.cat(
            (correlation, correlation.permute(0, 2, 1)), dim=0
        )  # [2*B, H*W, H*W]
        init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        grid = grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        b = b * 2

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

    correspondence = (
        torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)
    )  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    return flow, prob


def bilinear_sampler(img, coords, mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.to(dtype=coords.dtype)

    return img


def get_norm(norm_name):
    if norm_name == "bn":
        return nn.BatchNorm2d
    elif norm_name == "layer":
        return partial(LayerNorm2d, affine=True)
    elif norm_name == "ms_bn":
        return MultiStatsBatchNorm2d
    elif norm_name == "none":
        return None


def split_feature(
    feature,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b_new, h_new, w_new, c)
        )  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b_new, c, h_new, w_new)
        )  # [B*K*K, C, H/K, W/K]

    return feature


def merge_splits(
    splits,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = (
            splits.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(new_b, num_splits * h, num_splits * w, c)
        )  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = (
            splits.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(new_b, c, num_splits * h, num_splits * w)
        )  # [B, C, H, W]

    return merge


def split_feature_1d(
    feature,
    num_splits=2,
):
    # feature: [B, W, C]
    b, w, c = feature.size()
    assert w % num_splits == 0

    b_new = b * num_splits
    w_new = w // num_splits

    feature = feature.view(b, num_splits, w // num_splits, c).view(
        b_new, w_new, c
    )  # [B*K, W/K, C]

    return feature


def merge_splits_1d(
    splits,
    h,
    num_splits=2,
):
    b, w, c = splits.size()
    new_b = b // num_splits // h

    splits = splits.view(new_b, h, num_splits, w, c)
    merge = splits.view(new_b, h, num_splits * w, c)  # [B, H, W, C]

    return merge


class InputScaler(object):
    """Scale 2D torch.Tensor input to a target size, and then rescale it back to the original size."""

    def __init__(
        self,
        orig_shape: Tuple[int, int],
        stride: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = 1.0,
        interpolation_mode: str = "bilinear",
        interpolation_align_corners: bool = False,
    ) -> None:
        """Initialize InputScaler.

        Parameters
        ----------
        orig_shape : Tuple[int, int]
            The shape of the input tensor before the scale. I.e., the shape to which it will be rescaled back.
        stride : Optional[int], optional
            If provided, the input will be resized to the closest larger multiple of stride.
        size : Optional[Tuple[int, int]], optional
            The desired size after scaling defined as (height, width). If not provided, then scale_factor will be used instead.
        scale_factor : Optional[float], default 1.0
            This value is only used if stride and size are None. The multiplier that will be applied to the original shape to scale
            the input.
        interpolation_mode : str, default 'bilinear'
            How to perform the interpolation. It must be a value accepted by the 'mode' argument from
            torch.nn.functional.interpolate function.
        interpolation_align_corners : bool, default False
            Whether the interpolation keep the corners aligned. As defined in torch.nn.functional.interpolate.

        See Also
        --------
        torch.nn.functional.interpolate : The function used to scale the inputs.
        """
        super().__init__()
        self.orig_height, self.orig_width = orig_shape[-2:]
        if stride is not None:
            assert size is None, "only stride OR size can be provided, NOT BOTH."
            self.tgt_height = int(math.ceil(float(self.orig_height) / stride)) * stride
            self.tgt_width = int(math.ceil(float(self.orig_width) / stride)) * stride
        elif size is not None:
            assert stride is None, "only stride OR size can be provided, NOT BOTH."
            self.tgt_height, self.tgt_width = size
        else:
            self.tgt_height = int(self.orig_height * scale_factor)
            self.tgt_width = int(self.orig_width * scale_factor)

        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

    def fill(self, x: torch.Tensor, is_flow: bool = False) -> torch.Tensor:
        """Scale the input to the target size specified during initialization.

        Parameters
        ----------
        x : torch.Tensor
            The input to be scaled. Its shape must be (..., C, H, W), where ... means any number of dimensions.
        is_flow : bool
            Whether the input is a flow field or not. If it is, then its values are multiplied by the rescale factor.

        Returns
        -------
        torch.Tensor
            The scaled input.
        """
        return self._scale_keep_dims(x, (self.tgt_height, self.tgt_width), is_flow)

    def unfill(self, x: torch.Tensor, is_flow: bool = False) -> torch.Tensor:
        """Scale the input to back to the original size defined during initialization.

        Parameters
        ----------
        x : torch.Tensor
            The input to be rescaled back. Its shape must be (..., C, H, W), where ... means any number of dimensions.
        is_flow : bool
            Whether the input is a flow field or not. If it is, then its values are multiplied by the rescale factor.

        Returns
        -------
        torch.Tensor
            The rescaled input.
        """
        return self._scale_keep_dims(x, (self.orig_height, self.orig_width), is_flow)

    def _scale_keep_dims(
        self, x: torch.Tensor, size: Tuple[int, int], is_flow: bool
    ) -> torch.Tensor:
        """Scale the input to a given size while keeping the other dimensions intact.

        Parameters
        ----------
        x : torch.Tensor
            The input to be rescaled back. Its shape must be (..., C, H, W), where ... means any number of dimensions.
        size : Tuple[int, int]
            The target size to scale the input.
        is_flow : bool
            Whether the input is a flow field or not. If it is, then its values are multiplied by the rescale factor.

        Returns
        -------
        torch.Tensor
            The rescaled input.
        """
        x_shape = x.shape
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x = F.interpolate(
            x,
            size=size,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
        )

        if is_flow:
            x[:, 0] = x[:, 0] * (float(x.shape[-1]) / x_shape[-1])
            x[:, 1] = x[:, 1] * (float(x.shape[-2]) / x_shape[-2])

        new_shape = list(x_shape)
        new_shape[-2], new_shape[-1] = x.shape[-2], x.shape[-1]
        x = x.view(new_shape)
        return x


class InputPadder:
    """Pads images such that dimensions are divisible by stride."""

    def __init__(
        self,
        dims,
        stride=8,
        two_side_pad=True,
        pad_mode="replicate",
        pad_value=0.0,
        size=None,
    ):
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        ht, wd = dims[-2:]
        if size is None:
            pad_ht = (((ht // stride) + 1) * stride - ht) % stride
            pad_wd = (((wd // stride) + 1) * stride - wd) % stride
        else:
            pad_ht = size[0] - ht
            pad_wd = size[1] - wd
        if two_side_pad:
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def fill(self, x):
        in_shape = x.shape
        if len(in_shape) > 4:
            x = x.view(-1, *in_shape[-3:])
        x = F.pad(x, self._pad, mode=self.pad_mode, value=self.pad_value)
        if len(in_shape) > 4:
            x = x.view(*in_shape[:-2], *x.shape[-2:])
        return x

    def unfill(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]
