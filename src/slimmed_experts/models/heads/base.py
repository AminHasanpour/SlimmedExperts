"""Abstract multi-head interfaces for domain-specific prediction heads."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class MultiHead(nn.Module, ABC):
    """Abstract base class for domain-aware prediction heads."""

    @property
    @abstractmethod
    def domains(self) -> list[str]:
        """Return the ordered list of domains this head supports."""
        ...

    @abstractmethod
    def forward(self, features: Tensor, domain: str) -> Tensor:
        """Produce logits for a specific domain.

        Args:
            features: Feature tensor of shape ``(B, F)``.
            domain: Domain name selecting the target head.

        Returns:
            Logit tensor of shape ``(B, num_classes[domain])``.
        """
        ...
