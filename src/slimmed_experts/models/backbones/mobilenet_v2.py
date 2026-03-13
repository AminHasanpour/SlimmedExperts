"""MobileNetV2 backbone plugin implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import Tensor

from slimmed_experts.models.backbones.base import Backbone


class MobileNetV2Backbone(Backbone):
    """MobileNetV2 feature extractor for multi-head models.

    Args:
        width_mult: MobileNetV2 width multiplier (e.g. 0.35, 0.5, 0.75, 1.0).
        small_input: If ``True``, sets first conv stride to ``(1, 1)`` for
            low-resolution inputs such as 32x32 images.
    """

    def __init__(self, *, width_mult: float = 1.0, small_input: bool = False) -> None:
        super().__init__()
        base = tv_models.mobilenet_v2(weights=None, width_mult=width_mult)

        if small_input:
            # features[0] is Conv2dNormActivation; [0] is its Conv2d.
            base.features[0][0].stride = (1, 1)

        self.features: nn.Sequential = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._output_dim = int(base.last_channel)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
