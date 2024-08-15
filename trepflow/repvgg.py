# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn
import numpy as np
import torch
import copy

from .multi_stats_bn import MultiStatsBatchNorm2d, MultiStatsSequential


def conv_bn(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    use_ms_bn=False,
    bn_num_stats=1,
):
    result = MultiStatsSequential()
    if kernel_size > 0:
        result.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                padding_mode=padding_mode,
            ),
        )
    if use_ms_bn:
        result.add_module(
            "bn",
            MultiStatsBatchNorm2d(num_features=out_channels, num_stats=bn_num_stats),
        )
    else:
        result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=(3, 1, 0),
        stride=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        nonlinearity=nn.ReLU(inplace=True),
        use_ms_bn=False,
        bn_num_stats=1,
        deploy=False,
    ):
        super(RepConv2d, self).__init__()
        kernel_sizes = sorted(kernel_sizes, reverse=True)
        if dilation > 1 and len(kernel_sizes) > 1:
            assert kernel_sizes[1] <= 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_ms_bn = use_ms_bn
        self.bn_num_stats = bn_num_stats
        self.deploy = deploy

        self.nonlinearity = nonlinearity

        if deploy:
            ks = kernel_sizes[0]
            padding = (ks + (ks - 1) * (dilation - 1)) // 2
            self.rbr_reparam = self._build_rbr_conv()
        else:
            self.parallel_branches = nn.ModuleDict()
            for ks in self.kernel_sizes:
                branch_name = "identity" if ks == 0 else f"ks{ks}"
                if ks == 0 and (out_channels != in_channels or stride != 1):
                    continue

                padding = (ks + (ks - 1) * (dilation - 1)) // 2
                self.parallel_branches[branch_name] = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ks,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    padding_mode=padding_mode,
                    use_ms_bn=use_ms_bn,
                    bn_num_stats=bn_num_stats,
                )

    def forward(self, inputs, istat=-1):
        if hasattr(self, "rbr_reparam"):
            if self.use_ms_bn:
                x = self.rbr_reparam[istat](inputs)
            else:
                x = self.rbr_reparam(inputs)
            if self.nonlinearity is not None:
                x = self.nonlinearity(x)
            return x

        x = None
        for branch in self.parallel_branches.values():
            if x is None:
                x = branch(inputs, istat=istat)
            else:
                x = x + branch(inputs, istat=istat)
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)
        return x

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def _get_equivalent_kernel_bias(self):
        fused_kernel = None
        fused_bias = None

        for branch in self.parallel_branches.values():
            kernel, bias = self._fuse_bn_tensor(branch)
            kernel = self._pad_kernel(kernel)
            if fused_kernel is None:
                fused_kernel = kernel
                fused_bias = bias
            else:
                fused_kernel = fused_kernel + kernel
                fused_bias = fused_bias + bias

        return fused_kernel, fused_bias

    def _get_equivalent_ms_kernel_bias(self):
        fused_kernel = [None for _ in range(self.bn_num_stats)]
        fused_bias = [None for _ in range(self.bn_num_stats)]

        for branch in self.parallel_branches.values():
            kernel_list, bias_list = self._fuse_msbn_tensor(branch)
            for i in range(self.bn_num_stats):
                kernel = self._pad_kernel(kernel_list[i])
                if fused_kernel[i] is None:
                    fused_kernel[i] = kernel
                    fused_bias[i] = bias_list[i]
                else:
                    fused_kernel[i] = fused_kernel[i] + kernel
                    fused_bias[i] = fused_bias[i] + bias_list[i]

        return fused_kernel, fused_bias

    def _pad_kernel(self, kernel):
        if kernel is None:
            return 0
        else:
            pad = (self.kernel_sizes[0] - kernel.shape[-1]) // 2
            return torch.nn.functional.pad(kernel, [pad, pad, pad, pad])

    def _fuse_msbn_tensor(self, branch):
        if branch is None:
            return [0 for _ in range(self.bn_num_stats)], [
                0 for _ in range(self.bn_num_stats)
            ]

        kernel_list = []
        bias_list = []
        for k in range(self.bn_num_stats):
            if hasattr(branch, "conv"):
                kernel = branch.conv.weight
                if self.bn_num_stats == 1:
                    running_mean = branch.bn.running_mean
                    running_var = branch.bn.running_var
                else:
                    running_mean = getattr(branch.bn, f"running_mean_{k}")
                    running_var = getattr(branch.bn, f"running_var_{k}")
                gamma = branch.bn.weight
                beta = branch.bn.bias
                eps = branch.bn.eps
            else:
                assert isinstance(branch.bn, MultiStatsBatchNorm2d)
                if not hasattr(self, "id_tensor"):
                    input_dim = self.in_channels // self.groups
                    max_ks = self.kernel_sizes[0]
                    kernel_value = np.zeros(
                        (self.in_channels, input_dim, max_ks, max_ks), dtype=np.float32
                    )
                    for i in range(self.in_channels):
                        kernel_value[i, i % input_dim, max_ks // 2, max_ks // 2] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(
                        branch.bn.weight.device
                    )
                kernel = self.id_tensor
                if self.bn_num_stats == 1:
                    running_mean = branch.bn.running_mean
                    running_var = branch.bn.running_var
                else:
                    running_mean = getattr(branch.bn, f"running_mean_{k}")
                    running_var = getattr(branch.bn, f"running_var_{k}")
                gamma = branch.bn.weight
                beta = branch.bn.bias
                eps = branch.bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            kernel_list.append(kernel * t)
            bias_list.append(beta - running_mean * gamma / std)

        return kernel_list, bias_list

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if hasattr(branch, "conv"):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch.bn, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                max_ks = self.kernel_sizes[0]
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, max_ks, max_ks), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, max_ks // 2, max_ks // 2] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.bn.weight.device
                )
            kernel = self.id_tensor
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _build_rbr_conv(self):
        ks = self.kernel_sizes[0]
        padding = (ks + (ks - 1) * (self.dilation - 1)) // 2
        if self.use_ms_bn:
            return nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_sizes[0],
                        stride=self.stride,
                        padding=padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=True,
                    )
                    for _ in range(self.bn_num_stats)
                ]
            )
        else:
            return nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_sizes[0],
                stride=self.stride,
                padding=padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=True,
            )

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        self.rbr_reparam = self._build_rbr_conv()
        if self.use_ms_bn:
            kernel_list, bias_list = self._get_equivalent_ms_kernel_bias()
            for i in range(self.bn_num_stats):
                self.rbr_reparam[i].weight.data = kernel_list[i]
                self.rbr_reparam[i].bias.data = bias_list[i]
        else:
            kernel, bias = self._get_equivalent_kernel_bias()
            self.rbr_reparam.weight.data = kernel
            self.rbr_reparam.bias.data = bias
        self.__delattr__("parallel_branches")
        self.deploy = True


class RepVGG(nn.Module):
    def __init__(
        self, num_blocks, block_dims=None, override_groups_map=None, deploy=False
    ):
        super(RepVGG, self).__init__()
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map

        self.in_planes = block_dims[1]
        self.stage0 = RepConv2d(
            in_channels=block_dims[0],
            out_channels=block_dims[1],
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=self.deploy,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(block_dims[2], num_blocks[0], stride=2)
        self.stage2 = self._make_stage(block_dims[3], num_blocks[1], stride=2)
        self.stage_out = RepConv2d(
            in_channels=block_dims[3],
            out_channels=block_dims[4],
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=self.deploy,
        )

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepConv2d(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups,
                    deploy=self.deploy,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2):
            for block in stage:
                out = block(out)
        out = self.stage_out(out)
        return out


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
