"""Tests for plugin-based model components and composition."""

from __future__ import annotations

import pytest
import torch

from slimmed_experts.models.backbones.mobilenet_v2 import MobileNetV2Backbone
from slimmed_experts.models.backbones.slimnet import SlimNetBackbone
from slimmed_experts.models.heads.linear import LinearMultiHead
from slimmed_experts.models.model import MultiHeadModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DOMAINS = ["d1", "d2"]
NUM_CLASSES = {"d1": 10, "d2": 20}


@pytest.fixture()
def model():
    """Tiny composed model with MobileNetV2 backbone and linear multi-head."""
    backbone = MobileNetV2Backbone(width_mult=0.35, small_input=True)
    head = LinearMultiHead(DOMAINS, NUM_CLASSES, in_features=backbone.output_dim)
    return MultiHeadModel(backbone=backbone, head=head)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelCompositionInit:
    def test_missing_num_classes_raises_value_error(self):
        with pytest.raises(ValueError, match="num_classes is missing entries"):
            LinearMultiHead(
                domains=["aircraft", "dtd"],
                num_classes={"aircraft": 100},  # "dtd" missing
                in_features=64,
            )

    def test_domains_property(self, model):
        assert model.domains == DOMAINS

    def test_small_input_sets_stride_to_one(self):
        backbone = MobileNetV2Backbone(width_mult=0.35, small_input=True)
        # features[0] is the first ConvBNActivation; [0] is its Conv2d.
        assert backbone.features[0][0].stride == (1, 1)

    def test_default_input_keeps_stride_two(self):
        backbone = MobileNetV2Backbone(width_mult=0.35, small_input=False)
        assert backbone.features[0][0].stride == (2, 2)

    def test_slimnet_small_input_sets_stride_to_one(self):
        backbone = SlimNetBackbone(width_mult=0.5, small_input=True)
        assert backbone.features[0][0].stride == (1, 1)

    def test_slimnet_default_presets_include_all_tiers(self):
        presets = SlimNetBackbone.tier_presets()
        assert set(presets.keys()) == {"tiny", "small", "base"}
        assert presets["tiny"] < presets["small"] < presets["base"]

    def test_slimnet_width_mult_increases_parameter_count(self):
        tiny = SlimNetBackbone(width_mult=0.5, small_input=True)
        small = SlimNetBackbone(width_mult=0.75, small_input=True)
        base = SlimNetBackbone(width_mult=1.0, small_input=True)

        tiny_params = sum(p.numel() for p in tiny.parameters())
        small_params = sum(p.numel() for p in small.parameters())
        base_params = sum(p.numel() for p in base.parameters())

        assert tiny_params < small_params < base_params


class TestModelCompositionForward:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_output_shape_per_domain(self, model, domain):
        x = torch.randn(2, 3, 32, 32)
        logits = model(x, domain)
        assert logits.shape == (2, NUM_CLASSES[domain])

    def test_output_dtype_is_float32(self, model):
        x = torch.randn(1, 3, 32, 32)
        logits = model(x, "d1")
        assert logits.dtype == torch.float32

    def test_unknown_domain_raises_key_error(self, model):
        x = torch.randn(1, 3, 32, 32)
        with pytest.raises(KeyError, match="Unknown domain"):
            model(x, "not_a_domain")

    def test_different_num_classes_per_head(self):
        backbone = MobileNetV2Backbone(width_mult=0.35, small_input=True)
        head = LinearMultiHead(["a", "b"], {"a": 5, "b": 50}, in_features=backbone.output_dim)
        m = MultiHeadModel(backbone=backbone, head=head)
        x = torch.randn(1, 3, 32, 32)
        assert m(x, "a").shape == (1, 5)
        assert m(x, "b").shape == (1, 50)

    def test_forward_with_small_input_32x32(self):
        backbone = MobileNetV2Backbone(width_mult=0.35, small_input=True)
        head = LinearMultiHead(["cifar"], {"cifar": 100}, in_features=backbone.output_dim)
        m = MultiHeadModel(backbone=backbone, head=head)
        x = torch.randn(4, 3, 32, 32)
        assert m(x, "cifar").shape == (4, 100)

    def test_batch_size_one(self, model):
        x = torch.randn(1, 3, 32, 32)
        logits = model(x, "d1")
        assert logits.shape == (1, NUM_CLASSES["d1"])

    def test_slimnet_forward_with_74x74_input(self):
        backbone = SlimNetBackbone(width_mult=0.75, small_input=True)
        head = LinearMultiHead(["cifar"], {"cifar": 100}, in_features=backbone.output_dim)
        m = MultiHeadModel(backbone=backbone, head=head)
        x = torch.randn(4, 3, 74, 74)
        assert m(x, "cifar").shape == (4, 100)
