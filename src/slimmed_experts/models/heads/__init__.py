"""Multi-head plugin implementations."""

from slimmed_experts.models.heads.base import MultiHead
from slimmed_experts.models.heads.linear import LinearMultiHead

__all__ = ["MultiHead", "LinearMultiHead"]
