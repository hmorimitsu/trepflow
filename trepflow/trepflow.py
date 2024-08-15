from argparse import ArgumentParser
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from .corr import get_corr_block
from .multi_stats_bn import MultiStatsBatchNorm2d
from .next_encoder import NeXtEncoder
from .pwc_modules import rescale_flow
from .update import UpdateBlock
from .utils import get_norm, InputPadder, InputScaler


class SequenceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, outputs, inputs):
        """Loss function defined over sequence of flow predictions"""

        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exclude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.args.max_flow)

        for i in range(n_predictions):
            pred = flow_preds[i]
            i_weight = self.args.gamma ** (n_predictions - i - 1)

            diff = pred - flow_gt
            i_loss = (diff).abs()
            valid_loss = valid * i_loss
            flow_loss += i_weight * valid_loss.mean()

        return flow_loss


class TrepFlow(pl.LightningModule):
    def __init__(self, args):
        if args.use_rep_encoder:
            args.use_rep_stem = True
            args.use_rep_stage = True
            args.use_rep_out = True
        if args.use_rep_decoder:
            args.use_rep_gru = True
            args.motenc = "rep"

        super().__init__()

        args.num_recurrent_layers = int(math.log2(max(args.pyramid_ranges))) - 2
        self.args = args
        self.loss_fn = SequenceLoss(args)
        self.output_stride = int(2 ** (args.num_recurrent_layers + 2))

        for v in self.args.pyramid_ranges:
            assert (
                v > 1
            ), f"--pyramid_ranges values must be larger than 1, but found {v}"
            log_res = math.log2(v)
            assert (log_res) - int(
                log_res
            ) < 1e-3, f"--pyramid_ranges values must be powers of 2, but found {v}"

        self.pyramid_levels = [
            args.num_recurrent_layers + 1 - int(math.log2(v))
            for v in self.args.pyramid_ranges
        ]

        out_chs = self.args.enc_out_chs
        max_pyr_range = (min(self.args.pyramid_ranges), max(self.args.pyramid_ranges))

        self.fnet = NeXtEncoder(
            recurrent=self.args.enc_rec,
            max_pyr_range=max_pyr_range,
            num_recurrent_layers=args.num_recurrent_layers,
            hidden_chs=self.args.enc_hidden_chs,
            out_chs=out_chs,
            mlp_ratio=self.args.enc_mlp_ratio,
            norm_layer=get_norm(args.enc_norm),
            mlp_use_norm=args.mlp_use_norm,
            mlp_middle_conv=args.use_mlp_middle_conv,
            mlp_in_kernel_size=args.enc_mlp_in_kernel_size,
            mlp_out_kernel_size=args.enc_mlp_out_kernel_size,
            depth=self.args.enc_depth,
            next_kernel_size=self.args.next_kernel_size,
            next_num_conv_dw_layers=self.args.next_num_conv_dw_layers,
            next_use_conv_dw_relu=self.args.next_use_conv_dw_relu,
            next_dilation=self.args.next_dilation,
            next_use_act_relu=self.args.next_use_act_relu,
            rep_stem=self.args.use_rep_stem,
            rep_stage=self.args.use_rep_stage,
            rep_out=self.args.use_rep_out,
            rep_branchless=args.use_rep_branchless,
            use_residual_shortcut=self.args.enc_use_residual_shortcut,
            use_layer_scale=self.args.enc_use_layer_scale,
            use_ms_bn=self.args.enc_use_ms_bn,
            deploy=self.args.deploy,
        )

        self.dim_corr = (self.args.corr_range * 2 + 1) ** 2 * self.args.corr_levels

        self.update_block = UpdateBlock(self.args)

        if not self.args.net_use_tanh and args.use_rep_gru and args.enc_use_ms_bn:
            self.net_norm = MultiStatsBatchNorm2d(
                self.args.dec_net_chs, num_stats=self.args.iters
            )

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--pyramid_ranges",
            type=int,
            nargs="+",
            default=(32, 8),
            help="(maximum, minimum) feature pyramid strides.",
        )
        parser.add_argument(
            "--iters",
            type=int,
            default=12,
            help="Total number of refinement iterations.",
        )

        parser.add_argument(
            "--corr_mode",
            type=str,
            choices=("local", "allpairs"),
            default="allpairs",
            help="Correlation mode. Use local for low memory consumption or allpairs for maximum speed.",
        )
        parser.add_argument(
            "--corr_levels",
            type=int,
            default=1,
            help="Number or correlation pooling levels.",
        )
        parser.add_argument(
            "--corr_range",
            type=int,
            default=4,
            help="The correlation range will be 2*corr_range+1.",
        )

        parser.add_argument(
            "--mlp_use_norm",
            action="store_true",
        )
        parser.add_argument(
            "--next_kernel_size",
            type=int,
            default=7,
        )
        parser.add_argument(
            "--next_num_conv_dw_layers",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--not_next_use_conv_dw_relu",
            action="store_false",
            dest="next_use_conv_dw_relu",
        )
        parser.add_argument(
            "--next_dilation",
            type=int,
            nargs="+",
            default=(1,),
        )
        parser.add_argument(
            "--not_next_use_act_relu",
            action="store_false",
            dest="next_use_act_relu",
        )
        parser.add_argument(
            "--use_mlp_middle_conv",
            action="store_true",
        )
        parser.add_argument(
            "--not_enc_use_residual_shortcut",
            action="store_false",
            dest="enc_use_residual_shortcut",
        )
        parser.add_argument(
            "--not_enc_use_layer_scale",
            action="store_false",
            dest="enc_use_layer_scale",
        )
        parser.add_argument("--not_enc_rec", action="store_false", dest="enc_rec")
        parser.add_argument(
            "--enc_hidden_chs",
            type=int,
            default=96,
            help="Number of hidden channels in the encoder.",
        )
        parser.add_argument(
            "--enc_out_chs",
            type=int,
            default=192,
            help="Number of channels of the encoder features.",
        )
        parser.add_argument(
            "--enc_feat_chs",
            type=int,
            default=128,
            help="Number of channels of the encoder features.",
        )
        parser.add_argument(
            "--enc_mlp_ratio",
            type=float,
            default=4.0,
            help="Reverse bottleneck ratio in the encoder MLP.",
        )
        parser.add_argument(
            "--enc_mlp_in_kernel_size",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--enc_mlp_out_kernel_size",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--enc_depth",
            type=int,
            default=4,
            help="Number of NeXt blocks in the encoder.",
        )
        parser.add_argument(
            "--enc_norm",
            type=str,
            default="layer",
            choices=("bn", "layer", "none"),
        )
        parser.add_argument(
            "--enc_use_ms_bn",
            action="store_true",
        )

        parser.add_argument(
            "--not_net_use_tanh",
            action="store_false",
            dest="net_use_tanh",
        )
        parser.add_argument(
            "--not_dec_use_residual_shortcut",
            action="store_false",
            dest="dec_use_residual_shortcut",
        )
        parser.add_argument(
            "--not_dec_use_layer_scale",
            action="store_false",
            dest="dec_use_layer_scale",
        )
        parser.add_argument(
            "--dec_net_chs",
            type=int,
            default=64,
            help="Number of net hidden channels in the decoder. Must follow: enc_out_chs=dec_net_chs+dec_inp_chs.",
        )
        parser.add_argument(
            "--dec_inp_chs",
            type=int,
            default=64,
            help="Number of input hidden channels in the decoder. Must follow: enc_out_chs=dec_net_chs+dec_inp_chs.",
        )
        parser.add_argument(
            "--dec_motion_chs",
            type=int,
            default=128,
            help="Number of channels of the motion encoder features.",
        )
        parser.add_argument(
            "--dec_depth",
            type=int,
            default=2,
            help="Number of NeXt blocks in the decoder.",
        )
        parser.add_argument(
            "--dec_mlp_ratio",
            type=float,
            default=4.0,
            help="Reverse bottleneck ratio in the decoder MLP.",
        )
        parser.add_argument(
            "--dec_mlp_in_kernel_size",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--dec_mlp_out_kernel_size",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--dec_norm",
            type=str,
            default="layer",
            choices=("bn", "layer", "none"),
        )
        parser.add_argument(
            "--dec_flow_kernel_size",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--dec_flow_head_chs",
            type=int,
            default=256,
        )
        parser.add_argument(
            "--dec_motenc_corr_hidden_chs",
            type=int,
            default=256,
        )
        parser.add_argument(
            "--dec_motenc_corr_out_chs",
            type=int,
            default=192,
        )
        parser.add_argument(
            "--dec_motenc_flow_hidden_chs",
            type=int,
            default=128,
        )
        parser.add_argument(
            "--dec_motenc_flow_out_chs",
            type=int,
            default=64,
        )
        parser.add_argument(
            "--dec_motenc_fuse_hidden_chs",
            type=int,
            default=64,
        )
        parser.add_argument(
            "--dec_use_ms_bn",
            action="store_true",
        )
        parser.add_argument(
            "--motenc", type=str, default="base", choices=("base", "next", "rep")
        )

        parser.add_argument(
            "--use_rep_stem",
            action="store_true",
        )
        parser.add_argument(
            "--use_rep_stage",
            action="store_true",
        )
        parser.add_argument(
            "--use_rep_out",
            action="store_true",
        )
        parser.add_argument(
            "--use_rep_encoder",
            action="store_true",
            help="A shortcut to simultaneously set: use_rep_stem, use_rep_stage, use_rep_out",
        )
        parser.add_argument(
            "--use_rep_gru",
            action="store_true",
        )
        parser.add_argument(
            "--use_rep_motenc",
            action="store_true",
        )
        parser.add_argument(
            "--use_rep_decoder",
            action="store_true",
            help="A shortcut to simultaneously set: use_rep_gru, motenc=rep",
        )
        parser.add_argument(
            "--use_rep_branchless",
            action="store_true",
        )
        parser.add_argument(
            "--deploy",
            action="store_true",
        )

        parser.add_argument(
            "--gamma",
            type=float,
            default=0.8,
            help="Used to compute the loss. Decaying factor for intermediate predictions.",
        )
        parser.add_argument(
            "--max_flow",
            type=float,
            default=400.0,
            help="Used to compute the loss. Groundtruth flows with magnitudes larger than this value are ignored.",
        )

        return parser

    def coords_grid(self, batch, ht, wd, dtype, device):
        coords = torch.meshgrid(
            torch.arange(ht, dtype=dtype, device=device),
            torch.arange(wd, dtype=dtype, device=device),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0).to(dtype=dtype)
        return coords[None].repeat(batch, 1, 1, 1)

    def upsample_flow(self, flow, mask, factor, ch=2):
        """Upsample flow field [H/f, W/f, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, ch, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, ch, factor * H, factor * W)

    def forward(self, inputs):
        images = inputs["images"]
        images, image_resizer = self.preprocess_images(
            images,
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=False,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        x1_raw = images[:, 0]
        x2_raw = images[:, 1]
        b, _, height_im, width_im = x1_raw.size()

        x_pyramid = self.fnet(torch.cat([x1_raw, x2_raw], 0))
        x1_pyramid = [x[:b] for x in x_pyramid]
        x2_pyramid = [x[b:] for x in x_pyramid]

        # outputs
        flows = []

        pred_stride = min(min(self.args.pyramid_ranges), 8)

        start_level, output_level = self.pyramid_levels

        iters_per_level = [
            int(math.ceil(float(self.args.iters) / (output_level - start_level + 1)))
        ] * (output_level - start_level + 1)

        # init
        (
            b_size,
            _,
            h_x1,
            w_x1,
        ) = x1_pyramid[0].size()

        flow = torch.zeros(
            b_size, 2, h_x1, w_x1, dtype=x1_raw.dtype, device=x1_raw.device
        )

        net = None
        it = 0
        curr_stride = max(self.args.pyramid_ranges)
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            xh = x1.shape[1]
            fc = self.args.enc_feat_chs
            oc = xh - fc
            x1, cn1 = torch.split(x1, [fc, oc], dim=1)
            x2, cn2 = torch.split(x2, [fc, oc], dim=1)
            oc2 = oc // 2
            i1, n1 = torch.split(cn1, [oc2, oc2], dim=1)
            i2, n2 = torch.split(cn2, [oc2, oc2], dim=1)
            inp = torch.cat([i1, i2], 1)
            inp = torch.relu(inp)
            net = torch.cat([n1, n2], 1)
            if self.args.net_use_tanh:
                net = torch.tanh(net)
            elif self.args.use_rep_gru and self.args.enc_use_ms_bn:
                net = self.net_norm(net, istat=it)

            coords0 = self.coords_grid(
                x1.shape[0], x1.shape[2], x1.shape[3], x1.dtype, x1.device
            )

            corr_fn = get_corr_block(
                x1,
                x2,
                self.args.corr_levels,
                self.args.corr_range,
                alternate_corr=self.args.corr_mode == "local",
            )

            if l > 0:
                flow = rescale_flow(flow, x1.shape[-1], x1.shape[-2], to_local=False)
                flow = F.interpolate(
                    flow,
                    [x1.shape[-2], x1.shape[-1]],
                    mode="bilinear",
                    align_corners=True,
                )
                curr_stride //= 2

            for k in range(iters_per_level[l]):
                flow = flow.detach()
                # correlation
                out_corr = corr_fn(coords0 + flow)

                get_mask = self.training or (
                    l == (output_level - start_level) and k == (iters_per_level[l] - 1)
                )
                flow_res, net, mask = self.update_block(
                    net,
                    inp,
                    out_corr,
                    flow,
                    get_mask=get_mask,
                    it=it,
                )

                flow = flow + flow_res

                if self.training:
                    out_flow = rescale_flow(flow, width_im, height_im, to_local=False)
                    if mask is not None and l == (output_level - start_level):
                        out_flow = self.upsample_flow(out_flow, mask, pred_stride)
                        out_flow = F.interpolate(
                            out_flow,
                            [x1_raw.shape[-2], x1_raw.shape[-1]],
                            mode="bilinear",
                            align_corners=True,
                        )
                    else:
                        out_flow = F.interpolate(
                            out_flow,
                            [x1_raw.shape[-2], x1_raw.shape[-1]],
                            mode="bilinear",
                            align_corners=True,
                        )
                    flows.append(out_flow)
                elif l == (output_level - start_level) and (
                    k == (iters_per_level[l] - 1)
                ):
                    out_flow = rescale_flow(flow, width_im, height_im, to_local=False)
                    if mask is not None:
                        out_flow = self.upsample_flow(out_flow, mask, pred_stride)
                        out_flow = F.interpolate(
                            out_flow,
                            [x1_raw.shape[-2], x1_raw.shape[-1]],
                            mode="bilinear",
                            align_corners=True,
                        )
                    else:
                        out_flow = F.interpolate(
                            out_flow,
                            [x1_raw.shape[-2], x1_raw.shape[-1]],
                            mode="bilinear",
                            align_corners=True,
                        )
                    out_flow = self.postprocess_predictions(
                        out_flow, image_resizer, is_flow=True
                    )

                it += 1

        outputs = {}
        outputs["flow_small"] = flow
        outputs["flows"] = out_flow[:, None]
        outputs["total_iters"] = it
        if self.training:
            outputs["flow_preds"] = flows
        return outputs

    def preprocess_images(
        self,
        images: torch.Tensor,
        bgr_add: Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor] = 0,
        bgr_mult: Union[
            float, Tuple[float, float, float], np.ndarray, torch.Tensor
        ] = 1,
        bgr_to_rgb: bool = False,
        image_resizer: Optional[Union[InputPadder, InputScaler]] = None,
        resize_mode: str = "pad",
        target_size: Optional[Tuple[int, int]] = None,
        pad_mode: str = "replicate",
        pad_value: float = 0.0,
        pad_two_side: bool = True,
        interpolation_mode: str = "bilinear",
        interpolation_align_corners: bool = True,
    ) -> Tuple[torch.Tensor, Union[InputPadder, InputScaler]]:
        """Applies basic pre-processing to the images.

        The pre-processing is done in this order:
        1. images = images + bgr_add
        2. images = images * bgr_mult
        3. (optional) Convert BGR channels to RGB
        4. Pad or resize the input to the closest larger size multiple of self.output_stride

        Parameters
        ----------
        images : torch.Tensor
            A tensor with at least 3 dimensions in this order: [..., 3, H, W].
        bgr_add : Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor], default 0
            BGR values to be added to the images. It can be a single value, a triple, or a tensor with a shape compatible with images.
        bgr_mult : Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor], default 1
            BGR values to be multiplied by the images. It can be a single value, a triple, or a tensor with a shape compatible with images.
        bgr_to_rgb : bool, default False
            If True, flip the channels to convert from BGR to RGB.
        image_resizer : Optional[Union[InputPadder, InputScaler]]
            An instance of InputPadder or InputScaler that will be used to resize the images.
            If not provided, a new one will be created based on the given resize_mode.
        resize_mode : str, default "pad"
            How to resize the input. Accepted values are "pad" and "interpolation".
        target_size : Optional[Tuple[int, int]], default None
            If given, the images will be resized to this size, instead of calculating a multiple of self.output_stride.
        pad_mode : str, default "replicate"
            Used if resize_mode == "pad". How to pad the input. Must be one of the values accepted by the 'mode' argument of torch.nn.functional.pad.
        pad_value : float, default 0.0
            Used if resize_mode == "pad" and pad_mode == "constant". The value to fill in the padded area.
        pad_two_side : bool, default True
            Used if resize_mode == "pad". If True, half of the padding goes to left/top and the rest to right/bottom. Otherwise, all the padding goes to the bottom right.
        interpolation_mode : str, default "bilinear"
            Used if resize_mode == "interpolation". How to interpolate the input. Must be one of the values accepted by the 'mode' argument of torch.nn.functional.interpolate.
        interpolation_align_corners : bool, default True
            Used if resize_mode == "interpolation". See 'align_corners' in https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html.

        Returns
        -------
        torch.Tensor
            A copy of the input images after applying all of the pre-processing steps.
        Union[InputPadder, InputScaler]
            An instance of InputPadder or InputScaler that was used to resize the images.
            Can be used to reverse the resizing operations.
        """
        bgr_add = bgr_val_as_tensor(
            bgr_add, reference_tensor=images, bgr_tensor_shape_position=-3
        )
        images = images + bgr_add
        bgr_mult = bgr_val_as_tensor(
            bgr_mult, reference_tensor=images, bgr_tensor_shape_position=-3
        )
        images *= bgr_mult
        if bgr_to_rgb:
            images = torch.flip(images, [-3])

        stride = self.output_stride
        if target_size is not None:
            stride = None

        if image_resizer is None:
            if resize_mode == "pad":
                image_resizer = InputPadder(
                    images.shape,
                    stride=stride,
                    size=target_size,
                    pad_mode=pad_mode,
                    two_side_pad=pad_two_side,
                    pad_value=pad_value,
                )
            elif resize_mode == "interpolation":
                image_resizer = InputScaler(
                    images.shape,
                    stride=stride,
                    size=target_size,
                    interpolation_mode=interpolation_mode,
                    interpolation_align_corners=interpolation_align_corners,
                )
            else:
                raise ValueError(
                    f"resize_mode must be one of (pad, interpolation). Found: {resize_mode}."
                )

        images = image_resizer.fill(images)
        images = images.contiguous()
        return images, image_resizer

    def postprocess_predictions(
        self,
        prediction: torch.Tensor,
        image_resizer: Optional[Union[InputPadder, InputScaler]],
        is_flow: bool,
    ) -> torch.Tensor:
        """Simple resizing post-processing. Just use image_resizer to revert the resizing operations.

        Parameters
        ----------
        prediction : torch.Tensor
            A tensor with at least 3 dimensions in this order: [..., C, H, W].
        image_resizer : Optional[Union[InputPadder, InputScaler]]
            An instance of InputPadder or InputScaler that will be used to reverse the resizing done to the inputs.
            Typically, this will be the instance returned by self.preprocess_images().
        is_flow : bool
            Indicates if prediction is an optical flow prediction of not.
            Only used if image_resizer is an instance of InputScaler, in which case the flow values need to be scaled.

        Returns
        -------
        torch.Tensor
            A copy of the prediction after reversing the resizing.
        """
        if isinstance(image_resizer, InputScaler):
            return image_resizer.unfill(prediction, is_flow=is_flow)
        else:
            return image_resizer.unfill(prediction)


def are_shapes_compatible(
    shape1: Sequence[int],
    shape2: Sequence[int],
) -> bool:
    """Check if two tensor shapes are compatible.

    Similar to PyTorch or Numpy, two shapes are considered "compatible" if either they have the same shape or if one shape can be broadcasted into the other.
    We consider two shapes compatible if, and only if:
    1. their shapes have the same length (same number of dimension), and
    2. each dimension size is either equal or at least one of them is one.

    Parameters
    ----------
    shape1 : Sequence[int]
        The dimensions of the first shape.
    shape2 : Sequence[int]
        The dimensions of the second shape.

    Returns
    -------
    bool
        Whether the two given shapes are compatible.
    """
    if len(shape1) != len(shape2):
        return False
    for v1, v2 in zip(shape1, shape2):
        if v1 != 1 and v2 != 1 and v1 != v2:
            return False
    return True


def bgr_val_as_tensor(
    bgr_val: Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor],
    reference_tensor: torch.Tensor,
    bgr_tensor_shape_position: int = -3,
) -> torch.Tensor:
    """Convert multiple types of BGR values given as input to a torch.Tensor where the BGR values are in the same position as a reference tensor.

    The bgr values can be:
    - a single number, in which case it will be repeated three times to represent BGR.
    - a tuple, list, np.ndarray, or torch.Tensor with three elements.

    The resulting tensor will have the BGR values in the same index position as the reference_tensor.
    For example, given a reference tensor with shape [B, 3, H, W] and setting bgr_tensor_shape_position == -3
    indicates that the BGR position in this reference_tensor is at shape index -3, which is equivalent to index 1.
    Given these inputs, the resulting BGR tensor will have shape [1, 3, 1, 1], and the BGR values will be at shape index 1.

    Parameters
    ----------
    bgr_val : Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor]
        The BGR values to be converted into a tensor that will have a compatible shape with the input.
    reference_tensor: torch.Tensor
        The tensor with the reference shape to convert the bgr_val.
    bgr_tensor_shape_position : int, default -3
        Which position of the reference_tensor corresponds to the BGR values.
        Typical values are -1 (used in channels-last tensors) or -3 (..., CHW tensors)

    Returns
    -------
    torch.Tensor
        The bgr_val converted into a tensor with a shape compatible with reference_tensor.
    """
    is_compatible = False
    if isinstance(bgr_val, torch.Tensor):
        is_compatible = are_shapes_compatible(bgr_val.shape, reference_tensor.shape)
        assert is_compatible or (len(bgr_val.shape) == 1 and bgr_val.shape[0] == 3)
    elif isinstance(bgr_val, np.ndarray):
        is_compatible = are_shapes_compatible(bgr_val.shape, reference_tensor.shape)
        assert is_compatible or (len(bgr_val.shape) == 1 and bgr_val.shape[0] == 3)
        bgr_val = torch.from_numpy(bgr_val).to(
            dtype=reference_tensor.dtype, device=reference_tensor.device
        )
    elif isinstance(bgr_val, (tuple, list)):
        assert len(bgr_val) == 3
        bgr_val = torch.Tensor(bgr_val).to(
            dtype=reference_tensor.dtype, device=reference_tensor.device
        )
    elif isinstance(bgr_val, (int, float)):
        bgr_val = (
            torch.zeros(3, dtype=reference_tensor.dtype, device=reference_tensor.device)
            + bgr_val
        )

    if not is_compatible:
        bgr_dims = [1] * len(reference_tensor.shape)
        bgr_dims[bgr_tensor_shape_position] = 3
        bgr_val = bgr_val.reshape(bgr_dims)
    return bgr_val
