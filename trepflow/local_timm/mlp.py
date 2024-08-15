""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""

from torch import nn as nn


class ConvMlp(nn.Module):
    """MLP using 1x1 convs that keeps spatial dims"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU(inplace=True),
        norm_layer=None,
        middle_conv=False,
        drop=0.0,
        in_kernel_size=1,
        out_kernel_size=1,
    ):
        super().__init__()
        self.middle_conv = middle_conv

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (
            False
            if (norm_layer is not None and issubclass(norm_layer, nn.BatchNorm2d))
            else True
        )

        self.fc1 = nn.Conv2d(
            in_features,
            hidden_features,
            kernel_size=in_kernel_size,
            padding=in_kernel_size // 2,
            bias=bias,
        )
        self.norm = norm_layer(hidden_features) if norm_layer else None
        self.act = act_layer
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(
            hidden_features,
            out_features,
            kernel_size=out_kernel_size,
            padding=out_kernel_size // 2,
            bias=True,
        )

        if middle_conv:
            self.conv_middle = nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=3,
                padding=1,
                bias=bias,
                groups=hidden_features,
            )
            self.norm_middle = norm_layer(hidden_features) if norm_layer else None

    def forward(self, x):
        x = self.fc1(x)
        if self.norm is not None:
            x = self.norm(x)

        if self.middle_conv:
            x = self.conv_middle(x)
            if self.norm_middle is not None:
                x = self.norm_middle(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
