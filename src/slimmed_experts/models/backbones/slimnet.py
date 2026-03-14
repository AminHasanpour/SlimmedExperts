"""SlimNet backbone plugin implementation.

SlimNet is a MobileNetV2-like architecture tuned for small inputs (74x74 by
default in this project) and edge deployment. It uses inverted residual blocks
with depthwise separable convolutions and a width multiplier to scale capacity.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from slimmed_experts.models.backbones.base import Backbone


def _make_divisible(channels: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channels to the nearest divisible value.

    This follows common mobile architecture practice and avoids tiny channel
    counts that can hurt kernel efficiency on edge accelerators.
    """
    min_value = divisor if min_value is None else min_value
    new_channels = max(min_value, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(self, inp: int, oup: int, *, stride: int, expand_ratio: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class SlimNetBackbone(Backbone):
    """SlimNet feature extractor for multi-head models.

    Args:
        width_mult: Width multiplier that scales channel counts.
        small_input: If ``True`` (recommended for 74x74), use stride 1 in stem.
        round_nearest: Channel divisibility factor used by `_make_divisible`.
    """

    # (expand_ratio, output_channels, repeats, stride)
    _BLOCK_SETTINGS: list[tuple[int, int, int, int]] = [
        (1, 16, 1, 1),
        (4, 24, 2, 2),
        (4, 32, 2, 2),
        (4, 48, 2, 2),
        (6, 80, 2, 1),
        (6, 128, 1, 2),
    ]

    def __init__(
        self,
        *,
        width_mult: float = 1.0,
        small_input: bool = True,
        round_nearest: int = 8,
    ) -> None:
        super().__init__()
        if width_mult <= 0:
            raise ValueError(f"width_mult must be > 0, got {width_mult}")
        if round_nearest <= 0:
            raise ValueError(f"round_nearest must be > 0, got {round_nearest}")

        input_channel = _make_divisible(16 * width_mult, round_nearest)
        last_channel = _make_divisible(768 * max(1.0, width_mult), round_nearest)

        stem_stride = 1 if small_input else 2
        features: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(3, input_channel, kernel_size=3, stride=stem_stride, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True),
            )
        ]

        for expand_ratio, channels, repeats, stride in self._iter_settings(self._BLOCK_SETTINGS):
            output_channel = _make_divisible(channels * width_mult, round_nearest)
            for block_idx in range(repeats):
                block_stride = stride if block_idx == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                    )
                )
                input_channel = output_channel

        features.extend(
            [
                nn.Conv2d(input_channel, last_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True),
            ]
        )

        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._output_dim = last_channel

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    @classmethod
    def tier_presets(cls) -> dict[str, float]:
        """Return recommended width multipliers for Tiny/Small/Base tiers."""
        return {
            "tiny": 0.5,
            "small": 0.75,
            "base": 1.0,
        }

    @staticmethod
    def _iter_settings(settings: Iterable[tuple[int, int, int, int]]) -> Iterable[tuple[int, int, int, int]]:
        return settings
