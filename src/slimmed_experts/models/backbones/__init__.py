"""Backbone plugin implementations."""

from slimmed_experts.models.backbones.base import Backbone
from slimmed_experts.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from slimmed_experts.models.backbones.slimnet import SlimNetBackbone

__all__ = ["Backbone", "MobileNetV2Backbone", "SlimNetBackbone"]
