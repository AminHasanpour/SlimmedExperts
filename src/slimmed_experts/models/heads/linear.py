"""Linear multi-head plugin implementation."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from slimmed_experts.models.heads.base import MultiHead


class LinearMultiHead(MultiHead):
    """Domain-specific linear classifier heads.

    Args:
        domains: Ordered list of domain names.
        num_classes: Mapping from domain name to number of classes.
        in_features: Input feature size produced by the backbone.

    Raises:
        ValueError: If ``num_classes`` is missing any requested domain.
    """

    def __init__(
        self,
        domains: list[str],
        num_classes: dict[str, int],
        *,
        in_features: int,
    ) -> None:
        super().__init__()
        missing = set(domains) - num_classes.keys()
        if missing:
            raise ValueError(f"num_classes is missing entries for domains: {sorted(missing)}")

        self._heads = nn.ModuleDict({domain: nn.Linear(in_features, num_classes[domain]) for domain in domains})

    @property
    def domains(self) -> list[str]:
        return list(self._heads.keys())

    def forward(self, features: Tensor, domain: str) -> Tensor:
        if domain not in self._heads:
            raise KeyError(f"Unknown domain '{domain}'. Known domains: {self.domains}")
        return self._heads[domain](features)
