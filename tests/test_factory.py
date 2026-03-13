"""Tests for class-path plugin loading and model factory behavior."""

from __future__ import annotations

import pytest
import torch

from slimmed_experts.models.backbones.base import Backbone
from slimmed_experts.models.factory import build_model, load_class


class TestLoadClass:
    def test_loads_valid_class(self):
        cls = load_class("slimmed_experts.models.backbones.mobilenet_v2.MobileNetV2Backbone")
        assert issubclass(cls, Backbone)

    def test_invalid_format_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid class_path"):
            load_class("not_a_valid_path")

    def test_unknown_module_raises_import_error(self):
        with pytest.raises(ImportError, match="Could not import module"):
            load_class("does.not.exist.SomeClass")

    def test_unknown_symbol_raises_import_error(self):
        with pytest.raises(ImportError, match="has no attribute"):
            load_class("torch.nn.DoesNotExist")

    def test_non_class_symbol_raises_type_error(self):
        with pytest.raises(TypeError, match="is not a class"):
            load_class("torch.tensor")


class TestBuildModel:
    def test_builds_model_and_runs_forward(self):
        model = build_model(
            domains=["d1", "d2"],
            num_classes={"d1": 10, "d2": 20},
            backbone_class_path="slimmed_experts.models.backbones.mobilenet_v2.MobileNetV2Backbone",
            backbone_args={"width_mult": 0.35, "small_input": True},
            head_class_path="slimmed_experts.models.heads.linear.LinearMultiHead",
            head_args={},
        )

        x = torch.randn(2, 3, 32, 32)
        assert model(x, "d1").shape == (2, 10)
        assert model(x, "d2").shape == (2, 20)

    def test_backbone_must_inherit_backbone_base(self):
        with pytest.raises(TypeError, match="must inherit"):
            build_model(
                domains=["d1"],
                num_classes={"d1": 10},
                backbone_class_path="torch.nn.Linear",
                backbone_args={"in_features": 5, "out_features": 5},
                head_class_path="slimmed_experts.models.heads.linear.LinearMultiHead",
                head_args={},
            )

    def test_head_must_inherit_multi_head_base(self):
        with pytest.raises(TypeError, match="must inherit"):
            build_model(
                domains=["d1"],
                num_classes={"d1": 10},
                backbone_class_path="slimmed_experts.models.backbones.mobilenet_v2.MobileNetV2Backbone",
                backbone_args={"width_mult": 0.35, "small_input": True},
                head_class_path="torch.nn.Linear",
                head_args={"out_features": 5},
            )

    @pytest.mark.parametrize("reserved_key", ["domains", "num_classes", "in_features"])
    def test_head_args_reject_reserved_runtime_keys(self, reserved_key: str):
        with pytest.raises(ValueError, match="head.args cannot define reserved keys"):
            build_model(
                domains=["d1"],
                num_classes={"d1": 10},
                backbone_class_path="slimmed_experts.models.backbones.mobilenet_v2.MobileNetV2Backbone",
                backbone_args={"width_mult": 0.35, "small_input": True},
                head_class_path="slimmed_experts.models.heads.linear.LinearMultiHead",
                head_args={reserved_key: "blocked"},
            )
