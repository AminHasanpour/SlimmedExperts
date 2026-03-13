"""Composable multi-head model built from pluggable backbone and head modules."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from slimmed_experts.models.backbones.base import Backbone
from slimmed_experts.models.heads.base import MultiHead


class MultiHeadModel(nn.Module):
    """Glue model that composes a backbone and a domain-aware head."""

    def __init__(self, backbone: Backbone, head: MultiHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    @property
    def domains(self) -> list[str]:
        """Return ordered domains supported by the configured head."""
        return self.head.domains

    def forward(self, x: Tensor, domain: str) -> Tensor:
        """Run feature extraction followed by domain-specific head inference."""
        features = self.backbone.forward_features(x)
        return self.head(features, domain)
