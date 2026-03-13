"""Backbone plugin implementations."""

from slimmed_experts.models.backbones.base import Backbone
from slimmed_experts.models.backbones.mobilenet_v2 import MobileNetV2Backbone

__all__ = ["Backbone", "MobileNetV2Backbone"]
