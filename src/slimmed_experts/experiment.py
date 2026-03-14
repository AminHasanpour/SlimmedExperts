"""Experiment runner for multi-run pipeline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

import typer
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from slimmed_experts.pipeline import run_pipeline


def run_experiment(configs: list[dict]) -> list[dict[str, float]]:
    """Run the pipeline for each configuration.

    Args:
        configs: List of keyword argument dicts to pass to
            :func:`~slimmed_experts.pipeline.run_pipeline`.

    Returns:
        List of final validation metrics dicts, one per configuration.
    """
    results = []
    for i, cfg in enumerate(configs):
        logger.info(f"Starting run {i + 1}/{len(configs)}...")
        result = run_pipeline(**cfg)
        results.append(result)
        logger.info(f"Run {i + 1}/{len(configs)} complete. Metrics: {result}")
    return results


def _config_to_pipeline_kwargs(cfg: DictConfig) -> dict:
    """Flatten a merged OmegaConf config into keyword arguments for run_pipeline.

    Args:
        cfg: Merged experiment configuration.

    Returns:
        Dictionary of keyword arguments accepted by :func:`~slimmed_experts.pipeline.run_pipeline`.
    """
    backbone_args_raw = OmegaConf.to_container(cfg.model.backbone.args, resolve=True)
    head_args_raw = OmegaConf.to_container(cfg.model.head.args, resolve=True)
    backbone_args = cast(dict[str, Any], backbone_args_raw or {})
    head_args = cast(dict[str, Any], head_args_raw or {})

    return {
        "domains": list(cfg.data.load.domains),
        "data_dir": cfg.data.load.data_dir,
        "batch_size": cfg.data.preprocess.batch_size,
        "shuffle": cfg.data.preprocess.shuffle,
        "augment": cfg.data.preprocess.augment,
        "input_size": cfg.data.preprocess.input_size,
        "normalize": cfg.data.preprocess.normalize,
        "seed": cfg.data.preprocess.seed,
        "backbone_class_path": cfg.model.backbone.class_path,
        "backbone_args": backbone_args,
        "head_class_path": cfg.model.head.class_path,
        "head_args": head_args,
        "total_steps": cfg.train.total_steps,
        "learning_rate": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
        "optimizer": cfg.train.optimizer,
        "scheduler": cfg.train.scheduler,
        "warmup_steps": cfg.train.warmup_steps,
        "label_smoothing": cfg.train.label_smoothing,
        "val_every_n_steps": cfg.train.val_every_n_steps,
        "output_dir": cfg.train.output_dir,
        "wandb_project": cfg.wandb.project,
        "wandb_run_name": cfg.wandb.run_name,
        "device": cfg.train.device,
    }


def _prepare_configs(experiment_cfg: DictConfig) -> list[dict]:
    """Prepare pipeline kwargs from a loaded experiment config.

    Merges each variant's overrides on top of the base config and converts
    the result to a flat dict of :func:`~slimmed_experts.pipeline.run_pipeline`
    keyword arguments.

    Args:
        experiment_cfg: Top-level experiment config with ``base`` and ``variants`` keys.

    Returns:
        List of pipeline kwarg dicts, one per variant.
    """
    base: DictConfig = experiment_cfg.base
    variants = experiment_cfg.variants

    configs = []
    for variant in variants:
        merged = cast(DictConfig, OmegaConf.merge(base, variant))
        kwargs = _config_to_pipeline_kwargs(merged)
        configs.append(kwargs)

    return configs


app = typer.Typer()


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to experiment YAML configuration file."),
    ] = Path("configs/experiment.yaml"),
) -> None:
    """Run an experiment defined by a YAML configuration file.

    The YAML file must have a ``base`` field with the default pipeline
    configuration and a ``variants`` field with a list of per-run overrides.
    """
    logger.add("logs/experiment_{time}.log")
    logger.info(f"Loading experiment config from {config}")
    experiment_cfg = cast(DictConfig, OmegaConf.load(config))
    configs = _prepare_configs(experiment_cfg)
    logger.info(f"Running experiment with {len(configs)} variant(s)...")
    results = run_experiment(configs)
    logger.info(f"Experiment complete. Results: {results}")


if __name__ == "__main__":
    app()
