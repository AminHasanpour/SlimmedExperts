"""Tests for slimmed_experts.model."""

from __future__ import annotations

import pytest
import torch

from slimmed_experts.model import MobileNetV2MultiHead

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DOMAINS = ["d1", "d2"]
NUM_CLASSES = {"d1": 10, "d2": 20}


@pytest.fixture()
def model():
    """Tiny MobileNetV2 with two heads, small images enabled for speed."""
    return MobileNetV2MultiHead(
        domains=DOMAINS,
        num_classes=NUM_CLASSES,
        width_mult=0.35,
        small_input=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMobileNetV2MultiHeadInit:
    def test_missing_num_classes_raises_value_error(self):
        with pytest.raises(ValueError, match="num_classes is missing entries"):
            MobileNetV2MultiHead(
                domains=["aircraft", "dtd"],
                num_classes={"aircraft": 100},  # "dtd" missing
            )

    def test_domains_property(self, model):
        assert model.domains == DOMAINS

    def test_small_input_sets_stride_to_one(self):
        m = MobileNetV2MultiHead(
            domains=["d1"],
            num_classes={"d1": 5},
            width_mult=0.35,
            small_input=True,
        )
        # features[0] is the first ConvBNActivation; [0] is its Conv2d.
        assert m.backbone[0][0].stride == (1, 1)

    def test_default_input_keeps_stride_two(self):
        m = MobileNetV2MultiHead(
            domains=["d1"],
            num_classes={"d1": 5},
            width_mult=0.35,
            small_input=False,
        )
        assert m.backbone[0][0].stride == (2, 2)


class TestMobileNetV2MultiHeadForward:
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
        m = MobileNetV2MultiHead(
            domains=["a", "b"],
            num_classes={"a": 5, "b": 50},
            width_mult=0.35,
            small_input=True,
        )
        x = torch.randn(1, 3, 32, 32)
        assert m(x, "a").shape == (1, 5)
        assert m(x, "b").shape == (1, 50)

    def test_forward_with_small_input_32x32(self):
        m = MobileNetV2MultiHead(
            domains=["cifar"],
            num_classes={"cifar": 100},
            width_mult=0.35,
            small_input=True,
        )
        x = torch.randn(4, 3, 32, 32)
        assert m(x, "cifar").shape == (4, 100)

    def test_batch_size_one(self, model):
        x = torch.randn(1, 3, 32, 32)
        logits = model(x, "d1")
        assert logits.shape == (1, NUM_CLASSES["d1"])
