"""Tests for experiment config flattening into pipeline kwargs."""

from __future__ import annotations

from omegaconf import OmegaConf

from slimmed_experts.experiment import _config_to_pipeline_kwargs


def test_config_to_pipeline_kwargs_maps_plugin_fields() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "load": {"domains": ["d1"], "data_dir": "data"},
                "preprocess": {
                    "batch_size": 4,
                    "shuffle": True,
                    "augment": False,
                    "input_size": 74,
                    "normalize": False,
                    "seed": 42,
                },
            },
            "model": {
                "backbone": {
                    "class_path": "slimmed_experts.models.backbones.mobilenet_v2.MobileNetV2Backbone",
                    "args": {"width_mult": 0.35, "small_input": True},
                },
                "head": {
                    "class_path": "slimmed_experts.models.heads.linear.LinearMultiHead",
                    "args": {},
                },
            },
            "train": {
                "total_steps": 5,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "optimizer": "adam",
                "scheduler": "cosine",
                "val_every_n_steps": 1,
                "output_dir": None,
                "device": "cpu",
            },
            "wandb": {"project": "demo", "run_name": None},
        }
    )

    kwargs = _config_to_pipeline_kwargs(cfg)

    assert kwargs["backbone_class_path"] == cfg.model.backbone.class_path
    assert kwargs["head_class_path"] == cfg.model.head.class_path
    assert kwargs["backbone_args"] == {"width_mult": 0.35, "small_input": True}
    assert kwargs["head_args"] == {}
    assert kwargs["input_size"] == 74
    assert kwargs["normalize"] is False
    assert kwargs["scheduler"] == "cosine"
