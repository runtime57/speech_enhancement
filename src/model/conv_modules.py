from __future__ import annotations

from typing import Iterable, Union

import torch.nn as nn
from torch import Tensor


def _pair_kernel(kernel_size: Union[int, Iterable[int]]) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        # DeepFilterNet semantics:
        # integer k means temporal kernel=k, frequency kernel stays 3
        return (kernel_size, 3)
    k = tuple(kernel_size)
    if len(k) != 2:
        raise ValueError("kernel_size must be int or iterable of length 2")
    return int(k[0]), int(k[1])


class _BaseULBlock(nn.Module):
    def _make_conv(
        self,
        conv_module,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int,
        groups: int,
        use_deconv: bool,
    ) -> nn.Module:
        kt, kf = kernel_size
        pf = kf // 2

        if use_deconv:
            # Match convkxf(..., mode="transposed") geometry as closely as possible
            return conv_module(
                in_channels,
                out_channels,
                kernel_size,
                stride=(1, stride),
                padding=(kt - 1, pf),
                output_padding=(0, pf),
                groups=groups,
                bias=False,
            )

        return conv_module(
            in_channels,
            out_channels,
            kernel_size,
            stride=(1, stride),
            padding=(0, pf),
            groups=groups,
            bias=False,
        )

    def _time_pad(self, kt: int, lookahead: int) -> nn.Module:
        pad = (0, 0, kt - 1 - lookahead, lookahead)
        return nn.ConstantPad2d(pad, 0.0) if any(pad) else nn.Identity()

    @staticmethod
    def infer_in_width(width: int, stride: int, use_deconv: bool) -> int:
        # DeepFilterNet-compatible frequency geometry for kf=3, pad=1:
        # encoder stride=2: 24->12, 12->6  => in = out * 2
        # decoder stride=2: 6->12, 12->24 => in = out // 2
        if stride != 2:
            return width
        return width // 2 if use_deconv else width * 2


class XConvBlock(_BaseULBlock):
    """
    Lightweight plain conv block:
        conv -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        kernel_size: Union[int, Iterable[int]],
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
        lookahead: int = 0,
    ):
        super().__init__()
        kernel_size = _pair_kernel(kernel_size)
        kt, _ = kernel_size
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        layers = [
            self._time_pad(kt, lookahead),
            self._make_conv(
                conv_module,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                groups,
                use_deconv,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if not is_last:
            layers.append(nn.ReLU(inplace=True))

        self.ops = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.ops(x)


class XDWSBlock(_BaseULBlock):
    """
    Lightweight depthwise-separable block:
        1x1 pointwise -> BN -> ReLU
        -> depthwise(kx3 or transposed) -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        kernel_size: Union[int, Iterable[int]],
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
        lookahead: int = 0,
    ):
        super().__init__()
        kernel_size = _pair_kernel(kernel_size)
        kt, _ = kernel_size
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        dconv_layers = [
            self._time_pad(kt, lookahead),
            self._make_conv(
                conv_module,
                out_channels,
                out_channels,
                kernel_size,
                stride,
                out_channels,
                use_deconv,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if not is_last:
            dconv_layers.append(nn.ReLU(inplace=True))
        self.dconv = nn.Sequential(*dconv_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pconv(x)
        x = self.dconv(x)
        return x


class XMBBlock(_BaseULBlock):
    """
    Lightweight MobileNet-like inverted bottleneck block:
        1x1 pointwise -> BN -> ReLU
        -> depthwise(kx3 or transposed) -> BN -> ReLU
        -> 1x1 pointwise -> BN
        -> residual if shape matches
        -> ReLU (unless last)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        kernel_size: Union[int, Iterable[int]],
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
        lookahead: int = 0,
    ):
        super().__init__()
        kernel_size = _pair_kernel(kernel_size)
        kt, _ = kernel_size
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.dconv = nn.Sequential(
            self._time_pad(kt, lookahead),
            self._make_conv(
                conv_module,
                out_channels,
                out_channels,
                kernel_size,
                stride,
                out_channels,
                use_deconv,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.pconv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.out_act = nn.Identity() if is_last else nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        input_x = x
        x = self.pconv1(x)
        x = self.dconv(x)
        x = self.pconv2(x)

        if x.shape == input_x.shape:
            x = x + input_x

        x = self.out_act(x)
        return x


def make_block(
    block_name: str,
    in_channels: int,
    out_channels: int,
    width: int,
    kernel_size: Union[int, Iterable[int]],
    stride: int = 1,
    groups: int = 1,
    use_deconv: bool = False,
    is_last: bool = False,
    lookahead: int = 0,
) -> nn.Module:
    name = block_name.lower()
    if name in {"xconv", "conv"}:
        cls = XConvBlock
    elif name in {"xdws", "dws"}:
        cls = XDWSBlock
    elif name in {"xmb", "mb", "mm"}:
        cls = XMBBlock
    else:
        raise ValueError(f"Unknown block_name: {block_name}")

    return cls(
        in_channels=in_channels,
        out_channels=out_channels,
        width=width,
        kernel_size=kernel_size,
        stride=stride,
        groups=groups,
        use_deconv=use_deconv,
        is_last=is_last,
        lookahead=lookahead,
    )