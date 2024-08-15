import lightning.pytorch as pl
import torch
import torch.nn as nn

from .local_timm.drop import DropPath
from .multi_stats_bn import MultiStatsSequential
from .repvgg import RepConv2d


class RepConvMlp(nn.Module):
    """MLP using 1x1 convs that keeps spatial dims"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU(inplace=True),
        middle_conv=False,
        in_kernel_size=1,
        out_kernel_size=1,
        use_ms_bn=False,
        bn_num_stats=1,
        deploy=False,
    ):
        super().__init__()
        self.middle_conv = middle_conv

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = RepConv2d(
            in_features,
            hidden_features,
            kernel_sizes=(1, 0) if in_kernel_size == 1 else (in_kernel_size, 1, 0),
            nonlinearity=act_layer,
            use_ms_bn=use_ms_bn,
            bn_num_stats=bn_num_stats,
            deploy=deploy,
        )
        self.fc2 = RepConv2d(
            hidden_features,
            out_features,
            kernel_sizes=(1, 0) if out_kernel_size == 1 else (out_kernel_size, 1, 0),
            nonlinearity=None,
            use_ms_bn=use_ms_bn,
            bn_num_stats=bn_num_stats,
            deploy=deploy,
        )

        if middle_conv:
            self.conv_middle = RepConv2d(
                hidden_features,
                hidden_features,
                kernel_sizes=(3, 1, 0),
                groups=hidden_features,
                nonlinearity=act_layer,
                use_ms_bn=use_ms_bn,
                bn_num_stats=bn_num_stats,
                deploy=deploy,
            )

    def forward(self, x, istat=-1):
        x = self.fc1(x, istat=istat)
        if self.middle_conv:
            x = self.conv_middle(x, istat=istat)
        x = self.fc2(x, istat=istat)
        return x


class RepNeXtBlock(pl.LightningModule):
    def __init__(
        self,
        in_chs,
        out_chs=None,
        kernel_sizes=(3, 1, 0),
        stride=1,
        mlp_ratio=4,
        ls_init_value=1e-6,
        mlp_middle_conv=False,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        drop_path=0.0,
        num_conv_dw_layers=1,
        use_conv_dw_relu=True,
        use_residual_shortcut=True,
        use_layer_scale=True,
        use_ms_bn=False,
        bn_num_stats=1,
        use_act_relu=True,
        deploy=False,
    ):
        super().__init__()
        self.use_residual_shortcut = use_residual_shortcut

        out_chs = out_chs or in_chs

        conv_dw_list = []
        for i in range(num_conv_dw_layers):
            conv_dw_list.append(
                RepConv2d(
                    in_chs,
                    out_chs,
                    kernel_sizes,
                    stride=stride,
                    groups=in_chs,
                    deploy=deploy,
                    nonlinearity=(
                        (nn.ReLU(inplace=False) if use_act_relu else nn.GELU)
                        if (i < (num_conv_dw_layers - 1) and use_conv_dw_relu)
                        else None
                    ),
                    use_ms_bn=use_ms_bn,
                    bn_num_stats=bn_num_stats,
                )
            )
        self.conv_dw = MultiStatsSequential(*conv_dw_list)

        self.mlp = RepConvMlp(
            out_chs,
            int(mlp_ratio * out_chs),
            act_layer=nn.ReLU(inplace=True) if use_act_relu else nn.GELU(),
            middle_conv=mlp_middle_conv,
            in_kernel_size=mlp_in_kernel_size,
            out_kernel_size=mlp_out_kernel_size,
            use_ms_bn=use_ms_bn,
            bn_num_stats=bn_num_stats,
            deploy=deploy,
        )

        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(out_chs))
            if (ls_init_value > 0 and self.use_residual_shortcut and use_layer_scale)
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None

    def forward(self, x, istat=-1):
        if self.use_residual_shortcut:
            shortcut = x
        x = self.conv_dw(x, istat=istat)
        x = self.mlp(x, istat=istat)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.use_residual_shortcut:
            x = x + shortcut
        return x


class RepNeXtStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_sizes=(3, 1, 0),
        stride=2,
        depth=2,
        drop_path_rates=None,
        ls_init_value=1e-6,
        mlp_ratio=4,
        mlp_middle_conv=False,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        num_conv_dw_layers=1,
        use_conv_dw_relu=True,
        use_residual_shortcut=True,
        use_layer_scale=True,
        use_ms_bn=False,
        bn_num_stats=1,
        use_act_relu=True,
        deploy=False,
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1:
            ds_ks = (3, 1, 0) if stride > 1 else (1, 0)
            self.downsample = RepConv2d(
                in_chs,
                out_chs,
                kernel_sizes=ds_ks,
                stride=stride,
                nonlinearity=None,
                use_ms_bn=use_ms_bn,
                bn_num_stats=bn_num_stats,
                deploy=deploy,
            )
            in_chs = out_chs
        else:
            self.downsample = None

        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(
                RepNeXtBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    kernel_sizes=kernel_sizes,
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    mlp_ratio=mlp_ratio,
                    mlp_middle_conv=mlp_middle_conv,
                    mlp_in_kernel_size=mlp_in_kernel_size,
                    mlp_out_kernel_size=mlp_out_kernel_size,
                    num_conv_dw_layers=num_conv_dw_layers,
                    use_conv_dw_relu=use_conv_dw_relu,
                    use_residual_shortcut=use_residual_shortcut,
                    use_layer_scale=use_layer_scale,
                    use_ms_bn=use_ms_bn,
                    bn_num_stats=bn_num_stats,
                    use_act_relu=use_act_relu,
                    deploy=deploy,
                )
            )
            in_chs = out_chs
        self.blocks = MultiStatsSequential(*stage_blocks)

    def forward(self, x, istat=-1):
        if self.downsample is not None:
            x = self.downsample(x, istat=istat)
        x = self.blocks(x, istat=istat)
        return x
