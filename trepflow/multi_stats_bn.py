# Code adapted from https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/batchnorm.py

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter, Sequential, init, functional as F


class MultiStatsSequential(Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_istat = None

    def forward(self, input, istat=-1):
        if self.has_istat is None:
            self.has_istat = []
            for module in self:
                try:
                    input = module(input, istat=istat)
                    self.has_istat.append(True)
                except TypeError:
                    input = module(input)
                    self.has_istat.append(False)
        else:
            for i, module in enumerate(self):
                if self.has_istat[i]:
                    input = module(input, istat=istat)
                else:
                    input = module(input)
        return input


class MultiStatsBatchNorm2d(Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    num_stats: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        num_stats: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.num_stats = num_stats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            if self.num_stats == 1:
                self.register_buffer(
                    "running_mean", torch.zeros(num_features, **factory_kwargs)
                )
                self.register_buffer(
                    "running_var", torch.ones(num_features, **factory_kwargs)
                )
                self.running_mean: Optional[Tensor]
                self.running_var: Optional[Tensor]
            else:
                for i in range(num_stats):
                    self.register_buffer(
                        f"running_mean_{i}", torch.zeros(num_features, **factory_kwargs)
                    )
                    self.register_buffer(
                        f"running_var_{i}", torch.ones(num_features, **factory_kwargs)
                    )

            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            if self.num_stats == 1:
                # running_mean/running_var/num_batches... are registered at runtime depending
                # if self.track_running_stats is on
                self.running_mean.zero_()  # type: ignore[union-attr]
                self.running_var.fill_(1)  # type: ignore[union-attr]
            else:
                for i in range(self.num_stats):
                    getattr(self, f"running_mean_{i}").zero_()
                    getattr(self, f"running_var_{i}").fill_(1)
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight, 0.1)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = (
                    self.num_batches_tracked
                    if self.num_batches_tracked is not None
                    and self.num_batches_tracked.device != torch.device("meta")
                    else torch.tensor(0, dtype=torch.long)
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: Tensor, istat: int = -1) -> Tensor:
        if self.num_stats > 1:
            assert istat >= 0

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            if self.num_stats == 1:
                bn_training = (self.running_mean is None) and (self.running_var is None)
            else:
                bn_training = (self.running_mean_0 is None) and (
                    self.running_var_0 is None
                )

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        running_mean = None
        running_var = None
        if not self.training or self.track_running_stats:
            if self.num_stats == 1:
                running_mean = self.running_mean
                running_var = self.running_var
            else:
                running_mean = getattr(self, f"running_mean_{istat}")
                running_var = getattr(self, f"running_var_{istat}")

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean,
            running_var,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
