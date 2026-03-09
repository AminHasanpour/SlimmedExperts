"""Multi-head model implementations for the Visual Domain Decathlon (VDD).

All model variants share the :class:`MultiHeadModel` interface, which exposes a
single backbone with one classification head per domain.  New architectures can
be added by subclassing :class:`MultiHeadModel`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import Tensor


class MultiHeadModel(nn.Module, ABC):
    """Abstract base class for multi-head domain adaptation models.

    Subclasses implement a shared backbone with one linear classification head
    per domain, enabling the same training loop to be used across different
    architectures.
    """

    @property
    @abstractmethod
    def domains(self) -> list[str]:
        """Return the ordered list of domain names this model has heads for."""
        ...

    @abstractmethod
    def forward(self, x: Tensor, domain: str) -> Tensor:
        """Run a forward pass and return logits for the given domain.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            domain: Name of the domain whose head should produce logits.

        Returns:
            Logit tensor of shape ``(B, num_classes[domain])``.
        """
        ...


class MobileNetV2MultiHead(MultiHeadModel):
    """MobileNetV2 backbone with one classification head per domain.

    Loads a MobileNetV2 with the requested *width_mult* (not pre-trained) from
    TorchVision and attaches a separate :class:`~torch.nn.Linear` head for each
    of the supplied domains.

    Args:
        domains: Ordered list of domain names.
        num_classes: Mapping from domain name to number of output classes.
        width_mult: MobileNetV2 width multiplier (e.g. 0.35, 0.5, 0.75, 1.0).
        small_input: If ``True``, sets the first convolutional layer's stride to
            ``(1, 1)`` instead of ``(2, 2)``.  Useful for datasets with small
            spatial dimensions (e.g. CIFAR-100 at 32x32).

    Raises:
        ValueError: If *domains* contains names absent from *num_classes*.

    Example:
        >>> model = MobileNetV2MultiHead(
        ...     domains=["aircraft", "cifar100"],
        ...     num_classes={"aircraft": 100, "cifar100": 100},
        ...     width_mult=0.5,
        ...     small_input=False,
        ... )
        >>> logits = model(torch.randn(4, 3, 224, 224), "aircraft")
        >>> logits.shape
        torch.Size([4, 100])
    """

    def __init__(
        self,
        domains: list[str],
        num_classes: dict[str, int],
        *,
        width_mult: float = 1.0,
        small_input: bool = False,
    ) -> None:
        super().__init__()

        missing = set(domains) - num_classes.keys()
        if missing:
            raise ValueError(f"num_classes is missing entries for domains: {sorted(missing)}")

        base = tv_models.mobilenet_v2(weights=None, width_mult=width_mult)

        if small_input:
            # features[0] is the first ConvBNActivation block; [0] is its Conv2d.
            base.features[0][0].stride = (1, 1)

        self.backbone: nn.Sequential = base.features
        self.pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self._heads = nn.ModuleDict({domain: nn.Linear(base.last_channel, num_classes[domain]) for domain in domains})

    @property
    def domains(self) -> list[str]:
        return list(self._heads.keys())

    def forward(self, x: Tensor, domain: str) -> Tensor:
        """Run a forward pass for the given domain.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            domain: Domain name selecting which head to use.

        Returns:
            Logits of shape ``(B, num_classes[domain])``.

        Raises:
            KeyError: If *domain* is not registered with this model.
        """
        if domain not in self._heads:
            raise KeyError(f"Unknown domain '{domain}'. Known domains: {self.domains}")
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self._heads[domain](x)
