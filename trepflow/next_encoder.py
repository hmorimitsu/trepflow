from typing import Sequence, Tuple, Union

import torch.nn as nn

from .local_timm.norm import LayerNorm2d
from .local_timm.weight_init import trunc_normal_
from .next import NeXtStage
from .repvgg import RepConv2d
from .repnext import RepNeXtStage


class NeXtEncoder(nn.Module):
    def __init__(
        self,
        recurrent: bool,
        max_pyr_range: Tuple[int, int],
        num_recurrent_layers: int,
        hidden_chs: int,
        out_chs: int,
        norm_layer=LayerNorm2d,
        mlp_use_norm=False,
        mlp_ratio: float = 4,
        mlp_middle_conv: bool = False,
        mlp_in_kernel_size: int = 1,
        mlp_out_kernel_size: int = 1,
        depth: int = 2,
        next_dilation: Union[int, Sequence[int]] = 1,
        next_kernel_size: int = 7,
        next_num_conv_dw_layers: int = 1,
        next_use_conv_dw_relu: bool = True,
        next_use_act_relu: bool = True,
        rep_stem: bool = False,
        rep_stage: bool = False,
        rep_out: bool = False,
        rep_branchless: bool = False,
        use_residual_shortcut: bool = True,
        use_layer_scale: bool = True,
        use_ms_bn: bool = False,
        deploy: bool = False,
    ):
        super(NeXtEncoder, self).__init__()
        self.recurrent = recurrent
        self.max_pyr_range = max_pyr_range
        self.num_recurrent_layers = num_recurrent_layers
        self.rep_stem = rep_stem
        self.rep_stage = rep_stage
        self.rep_out = rep_out

        self.stem = self._make_stem(
            hidden_chs,
            norm_layer=norm_layer,
            rep_stem=rep_stem,
            rep_branchless=rep_branchless,
            use_ms_bn=use_ms_bn,
            deploy=deploy,
        )
        self.rec_stage = self._make_stage(
            recurrent,
            num_recurrent_layers,
            hidden_chs,
            norm_layer=norm_layer,
            mlp_use_norm=mlp_use_norm,
            mlp_ratio=mlp_ratio,
            mlp_middle_conv=mlp_middle_conv,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            depth=depth,
            next_dilation=next_dilation,
            next_kernel_size=next_kernel_size,
            next_num_conv_dw_layers=next_num_conv_dw_layers,
            next_use_conv_dw_relu=next_use_conv_dw_relu,
            next_use_act_relu=next_use_act_relu,
            rep_stage=rep_stage,
            rep_branchless=rep_branchless,
            use_residual_shortcut=use_residual_shortcut,
            use_layer_scale=use_layer_scale,
            use_ms_bn=use_ms_bn,
            deploy=deploy,
        )
        self.out_layer = self._make_out_layer(
            recurrent,
            num_recurrent_layers,
            hidden_chs,
            out_chs,
            rep_out=rep_out,
            rep_branchless=rep_branchless,
            use_ms_bn=use_ms_bn,
            deploy=deploy,
        )

    def _make_stem(
        self, hidden_chs: int, norm_layer, rep_stem, rep_branchless, use_ms_bn, deploy
    ):
        if rep_stem:
            kernel_sizes = (3,) if rep_branchless else (3, 1, 0)
            return nn.Sequential(
                RepConv2d(
                    3,
                    hidden_chs // 3,
                    kernel_sizes=kernel_sizes,
                    stride=2,
                    use_ms_bn=use_ms_bn,
                    bn_num_stats=1,
                    deploy=deploy,
                ),
                RepConv2d(
                    hidden_chs // 3,
                    2 * hidden_chs // 3,
                    kernel_sizes=kernel_sizes,
                    stride=2,
                    use_ms_bn=use_ms_bn,
                    bn_num_stats=1,
                    deploy=deploy,
                ),
                RepConv2d(
                    2 * hidden_chs // 3,
                    hidden_chs,
                    kernel_sizes=kernel_sizes,
                    stride=1,
                    nonlinearity=None,
                    use_ms_bn=use_ms_bn,
                    bn_num_stats=1,
                    deploy=deploy,
                ),
            )
        else:
            bias = True
            return nn.Sequential(
                nn.Conv2d(3, hidden_chs // 3, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_chs // 3,
                    2 * hidden_chs // 3,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    2 * hidden_chs // 3,
                    hidden_chs,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                ),
                nn.Identity() if norm_layer is None else norm_layer(hidden_chs),
            )

    def _make_stage(
        self,
        recurrent: bool,
        num_recurrent_layers: int,
        hidden_chs: int,
        norm_layer,
        mlp_use_norm,
        mlp_ratio,
        mlp_middle_conv,
        mlp_in_kernel_size,
        mlp_out_kernel_size,
        depth,
        next_dilation,
        next_kernel_size,
        next_num_conv_dw_layers,
        next_use_conv_dw_relu,
        next_use_act_relu,
        rep_stage,
        rep_branchless,
        use_residual_shortcut,
        use_layer_scale,
        use_ms_bn,
        deploy,
    ):
        kernel_sizes = (
            (next_kernel_size,) if rep_branchless else (next_kernel_size, 1, 0)
        )
        if recurrent:
            if rep_stage:
                return RepNeXtStage(
                    hidden_chs,
                    hidden_chs,
                    stride=2,
                    depth=depth,
                    mlp_ratio=mlp_ratio,
                    mlp_middle_conv=mlp_middle_conv,
                    mlp_in_kernel_size=mlp_in_kernel_size,
                    mlp_out_kernel_size=mlp_out_kernel_size,
                    kernel_sizes=kernel_sizes,
                    num_conv_dw_layers=next_num_conv_dw_layers,
                    use_conv_dw_relu=next_use_conv_dw_relu,
                    use_residual_shortcut=use_residual_shortcut,
                    use_layer_scale=use_layer_scale,
                    use_ms_bn=use_ms_bn,
                    bn_num_stats=num_recurrent_layers,
                    use_act_relu=next_use_act_relu,
                    deploy=deploy,
                )
            else:
                return NeXtStage(
                    hidden_chs,
                    hidden_chs,
                    stride=2,
                    dilation=next_dilation,
                    depth=depth,
                    norm_layer=norm_layer,
                    mlp_use_norm=mlp_use_norm,
                    mlp_ratio=mlp_ratio,
                    mlp_middle_conv=mlp_middle_conv,
                    mlp_in_kernel_size=mlp_in_kernel_size,
                    mlp_out_kernel_size=mlp_out_kernel_size,
                    kernel_size=next_kernel_size,
                    num_conv_dw_layers=next_num_conv_dw_layers,
                    use_conv_dw_relu=next_use_conv_dw_relu,
                    use_residual_shortcut=use_residual_shortcut,
                    use_layer_scale=use_layer_scale,
                    use_act_relu=next_use_act_relu,
                )
        else:
            if rep_stage:
                return nn.ModuleList(
                    [
                        RepNeXtStage(
                            hidden_chs,
                            hidden_chs,
                            stride=2,
                            depth=depth,
                            mlp_ratio=mlp_ratio,
                            mlp_middle_conv=mlp_middle_conv,
                            mlp_in_kernel_size=mlp_in_kernel_size,
                            mlp_out_kernel_size=mlp_out_kernel_size,
                            kernel_sizes=kernel_sizes,
                            num_conv_dw_layers=next_num_conv_dw_layers,
                            use_conv_dw_relu=next_use_conv_dw_relu,
                            use_residual_shortcut=use_residual_shortcut,
                            use_layer_scale=use_layer_scale,
                            use_ms_bn=use_ms_bn,
                            bn_num_stats=1,
                            use_act_relu=next_use_act_relu,
                            deploy=deploy,
                        )
                        for _ in range(num_recurrent_layers)
                    ]
                )
            else:
                return nn.ModuleList(
                    [
                        NeXtStage(
                            hidden_chs,
                            hidden_chs,
                            stride=2,
                            dilation=next_dilation,
                            depth=depth,
                            norm_layer=norm_layer,
                            mlp_use_norm=mlp_use_norm,
                            mlp_ratio=mlp_ratio,
                            mlp_middle_conv=mlp_middle_conv,
                            mlp_in_kernel_size=mlp_in_kernel_size,
                            mlp_out_kernel_size=mlp_out_kernel_size,
                            kernel_size=next_kernel_size,
                            num_conv_dw_layers=next_num_conv_dw_layers,
                            use_conv_dw_relu=next_use_conv_dw_relu,
                            use_residual_shortcut=use_residual_shortcut,
                            use_layer_scale=use_layer_scale,
                            use_act_relu=next_use_act_relu,
                        )
                        for _ in range(num_recurrent_layers)
                    ]
                )

    def _make_out_layer(
        self,
        recurrent: bool,
        num_recurrent_layers: int,
        hidden_chs: int,
        out_chs: int,
        rep_out,
        rep_branchless,
        use_ms_bn,
        deploy,
    ):
        kernel_sizes = (1,) if rep_branchless else (1, 0)
        if recurrent:
            if rep_out:
                return RepConv2d(
                    hidden_chs,
                    out_chs,
                    kernel_sizes=kernel_sizes,
                    nonlinearity=None,
                    use_ms_bn=use_ms_bn,
                    bn_num_stats=num_recurrent_layers,
                    deploy=deploy,
                )
            else:
                return nn.Conv2d(hidden_chs, out_chs, kernel_size=1)
        else:
            if rep_out:
                return nn.ModuleList(
                    [
                        RepConv2d(
                            hidden_chs,
                            out_chs,
                            kernel_sizes=kernel_sizes,
                            nonlinearity=None,
                            use_ms_bn=use_ms_bn,
                            bn_num_stats=1,
                            deploy=deploy,
                        )
                        for _ in range(num_recurrent_layers)
                    ]
                )
            else:
                return nn.ModuleList(
                    [
                        nn.Conv2d(hidden_chs, out_chs, kernel_size=1)
                        for _ in range(num_recurrent_layers)
                    ]
                )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x_pyramid = []
        curr_stride = 4
        for i in range(self.num_recurrent_layers + 1):
            if i == 0:
                x = self.stem(x)
            else:
                if self.recurrent:
                    if self.rep_stage:
                        x = self.rec_stage(x, istat=i - 1).contiguous()
                    else:
                        x = self.rec_stage(x).contiguous()
                else:
                    x = self.rec_stage[i - 1](x).contiguous()
                curr_stride *= 2

            if curr_stride >= self.max_pyr_range[0]:
                x_pyramid.append(x)

        for i, x in enumerate(x_pyramid[::-1]):
            if self.recurrent:
                if self.rep_out:
                    x = self.out_layer(x, istat=i)
                else:
                    x = self.out_layer(x)
            else:
                x = self.out_layer[-i - 1](x)
            x_pyramid[i] = x

        return x_pyramid
