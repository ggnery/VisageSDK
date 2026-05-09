from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence
from typing_extensions import override

import torch
from torch import nn, Tensor

from backbone.base_backbone import BaseBackbone
from config.backbone.base_backbone_config import BackboneConfig


def _make_divisible(v: float, divisor: int = 8) -> int:
    """This function ensures that all layers have a channel number divisible by 8"""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(nn.Sequential):
    """
    A modular convolutional block that combines a 2D convolutional layer, optional normalization, and optional activation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.PReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: bool = True,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            if activation_layer == nn.PReLU:
                layers.append(activation_layer(num_parameters=out_channels))
            else:
                params = {} if inplace is None else {"inplace": inplace}
                layers.append(activation_layer(**params))

        super().__init__(*layers)


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
    """
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input)
        return scale * input


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class GDC(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.features = nn.Sequential(
            LinearBlock(in_channels, in_channels, kernel_size=7, stride=1, padding=0, groups=in_channels),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {cnf.stride}")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.PReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            Conv2dNormActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, activation_layer=None)
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(BaseBackbone):
    """
    This block implements the MobileNetV3 from the framework on https://github.com/yakhyo/face-recognition.
    """
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        
        # Get configuration parameters
        self.model_size = config.model_size
        self.width_mult = config.width_mult
        self.reduced_tail = config.reduced_tail
        self.dilated = config.dilated
        self.input_size = config.input_size
        
        # Build model configuration
        inverted_residual_setting, last_channel = self._mobilenet_v3_conf(
            arch="mobilenet_v3_" + self.model_size,
            width_mult=self.width_mult,
            reduced_tail=self.reduced_tail,
            dilated=self.dilated
        )
        
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=1,  # change from 2 -> 1
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(InvertedResidual(cnf))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.output_layer = GDC(in_channels=lastconv_output_channels, embedding_dim=self.embedding_size)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:  # Check if bias exists
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.output_layer(x)
        return x

    def _mobilenet_v3_conf(
        self,
        arch: str,
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        **kwargs: Any
    ):
        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1

        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

        if arch == "mobilenet_v3_large":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
                bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
                bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
                bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
                bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
                bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
                bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
                bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            ]
            last_channel = adjust_channels(1280 // reduce_divider)  # C5
        elif arch == "mobilenet_v3_small":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
                bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
                bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            ]
            last_channel = adjust_channels(1024 // reduce_divider)  # C5
        else:
            raise ValueError(f"Unsupported model type {arch}")

        return inverted_residual_setting, last_channel