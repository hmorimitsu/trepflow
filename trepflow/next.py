import lightning.pytorch as pl
import torch
import torch.nn as nn

from .local_timm.create_conv2d import create_conv2d
from .local_timm.drop import DropPath
from .local_timm.mlp import ConvMlp


class NeXtBlock(pl.LightningModule):
    def __init__(
        self,
        in_chs,
        out_chs=None,
        kernel_size=7,
        stride=1,
        dilation=1,
        mlp_ratio=4,
        ls_init_value=1e-6,
        norm_layer=None,
        mlp_use_norm=False,
        mlp_middle_conv=False,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        drop_path=0.0,
        num_conv_dw_layers=1,
        use_conv_dw_relu=True,
        use_residual_shortcut=True,
        use_layer_scale=True,
        use_act_relu=True,
    ):
        super().__init__()
        self.use_residual_shortcut = use_residual_shortcut

        if isinstance(dilation, int):
            dilation = [dilation for _ in range(num_conv_dw_layers)]
        elif isinstance(dilation, (list, tuple)) and len(dilation) == 1:
            dilation = [dilation[0] for _ in range(num_conv_dw_layers)]
        assert len(dilation) == num_conv_dw_layers

        out_chs = out_chs or in_chs

        conv_dw_list = []
        for i in range(num_conv_dw_layers):
            bias = (
                True
                if ((norm_layer is None) or (i < (num_conv_dw_layers - 1)))
                else False
            )
            conv_dw_list.append(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size + (kernel_size - 1) * (dilation[i] - 1)) // 2,
                    dilation=dilation[i],
                    groups=in_chs,
                    bias=bias,
                )
            )
            if i < (num_conv_dw_layers - 1) and use_conv_dw_relu:
                conv_dw_list.append(
                    nn.ReLU(inplace=False) if use_act_relu else nn.GELU()
                )
        self.conv_dw = nn.Sequential(*conv_dw_list)

        self.norm = None if norm_layer is None else norm_layer(out_chs)
        self.mlp = ConvMlp(
            out_chs,
            int(mlp_ratio * out_chs),
            act_layer=nn.ReLU(inplace=True) if use_act_relu else nn.GELU(),
            norm_layer=norm_layer if mlp_use_norm else None,
            middle_conv=mlp_middle_conv,
            in_kernel_size=mlp_in_kernel_size,
            out_kernel_size=mlp_out_kernel_size,
        )
        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(out_chs))
            if (ls_init_value > 0 and self.use_residual_shortcut and use_layer_scale)
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None

    def forward(self, x):
        if self.use_residual_shortcut:
            shortcut = x
        x = self.conv_dw(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.use_residual_shortcut:
            x = x + shortcut
        return x


class NeXtStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=3,
        stride=2,
        depth=2,
        dilation=(1, 1),
        drop_path_rates=None,
        ls_init_value=1.0,
        norm_layer=None,
        mlp_use_norm=False,
        mlp_ratio=4,
        mlp_middle_conv=False,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        num_conv_dw_layers=1,
        use_conv_dw_relu=True,
        use_residual_shortcut=True,
        use_layer_scale=True,
        use_act_relu=False,
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1:
            ds_ks = 3 if stride > 1 else 1
            if norm_layer is None:
                self.downsample = create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=1,
                    padding=ds_ks // 2,
                    bias=True,
                )
            else:
                self.downsample = nn.Sequential(
                    norm_layer(in_chs),
                    create_conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=ds_ks,
                        stride=stride,
                        dilation=1,
                        padding=ds_ks // 2,
                        bias=True,
                    ),
                )
            in_chs = out_chs
        else:
            self.downsample = None

        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(
                NeXtBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    norm_layer=norm_layer,
                    mlp_use_norm=mlp_use_norm,
                    mlp_ratio=mlp_ratio,
                    mlp_middle_conv=mlp_middle_conv,
                    mlp_in_kernel_size=mlp_in_kernel_size,
                    mlp_out_kernel_size=mlp_out_kernel_size,
                    num_conv_dw_layers=num_conv_dw_layers,
                    use_conv_dw_relu=use_conv_dw_relu,
                    use_residual_shortcut=use_residual_shortcut,
                    use_layer_scale=use_layer_scale,
                    use_act_relu=use_act_relu,
                )
            )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.blocks(x)
        return x
