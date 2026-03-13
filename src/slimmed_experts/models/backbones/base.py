"""Abstract backbone interfaces for pluggable model construction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class Backbone(nn.Module, ABC):
    """Abstract base class for feature extraction backbones.

    Backbones transform image tensors into 2D feature tensors with shape
    ``(B, F)`` where ``F`` is available through :attr:`output_dim`.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the feature dimension ``F`` produced by :meth:`forward_features`."""
        ...

    @abstractmethod
    def forward_features(self, x: Tensor) -> Tensor:
        """Extract feature vectors from image inputs.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, output_dim)``.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Alias to :meth:`forward_features` for regular ``nn.Module`` usage."""
        return self.forward_features(x)
