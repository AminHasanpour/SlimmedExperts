"""End-to-end training pipeline for multi-domain models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, cast

import typer
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from slimmed_experts.data import VDD_DOMAINS, load_domains, make_dataloader
from slimmed_experts.models import build_model
from slimmed_experts.train import train


def run_pipeline(
    domains: list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
    # Data preprocessing
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    input_size: int = 74,
    normalize: bool = False,
    seed: int | None = None,
    # Model
    backbone_class_path: str = "slimmed_experts.models.backbones.mobilenet_v2.MobileNetV2Backbone",
    backbone_args: dict[str, Any] | None = None,
    head_class_path: str = "slimmed_experts.models.heads.linear.LinearMultiHead",
    head_args: dict[str, Any] | None = None,
    # Training
    total_steps: int = 1000,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    scheduler: str | None = "cosine",
    warmup_steps: int = 0,
    backbone_steps: int | None = None,
    label_smoothing: float = 0.0,
    val_every_n_steps: int = 100,
    output_dir: str | Path | None = None,
    wandb_project: str = "slimmed-experts",
    wandb_run_name: str | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Run the full training pipeline end-to-end.

    Loads data for each domain, builds a :class:`~slimmed_experts.models.model.MultiHeadModel`,
    and trains it with :func:`~slimmed_experts.train.train`.

    Args:
        domains: List of VDD domain names to train on (must be a subset of
            :data:`~slimmed_experts.data.VDD_DOMAINS`).
        data_dir: Root directory containing the domain folders.  Defaults to
            ``"data"``.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the training split before batching.
        augment: If ``True``, applies train-time augmentation to train split.
        input_size: Target square image size used by preprocessing transforms.
        normalize: If ``True``, normalise inputs using per-domain statistics
            computed from each domain's training split.
        seed: Random seed for reproducible data shuffling.
        backbone_class_path: Dotted import path to the backbone class.
        backbone_args: Constructor arguments for the selected backbone.
        head_class_path: Dotted import path to the head class.
        head_args: Constructor arguments for the selected head.
        total_steps: Total number of gradient update steps across all domains.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularisation coefficient.
        optimizer: ``"adam"`` or ``"sgd"``.
        scheduler: LR scheduler to use after warmup (``"cosine"`` or ``None``).
        warmup_steps: Number of warmup steps with linear LR increase.
        backbone_steps: Number of initial training steps where the backbone
            remains trainable. After this, backbone parameters are frozen and
            only heads are updated. ``None`` means no freezing.
        label_smoothing: Label smoothing factor for cross entropy loss.
        val_every_n_steps: Run validation every this many gradient steps.
        output_dir: Directory to write ``best.pt`` and ``last.pt`` checkpoints.
            ``None`` disables checkpoint saving.
        wandb_project: Weights & Biases project name.
        wandb_run_name: W&B run display name (``None`` = W&B auto-generates).
        device: PyTorch device string or object (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        Dictionary of final validation metrics
        ``{"acc/{domain}": float, "loss/{domain}": float, ...}``.

    Raises:
        ValueError: If any entry in *domains* is not in :data:`~slimmed_experts.data.VDD_DOMAINS`.
    """
    invalid = [d for d in domains if d not in VDD_DOMAINS]
    if invalid:
        raise ValueError(f"Unknown domains: {sorted(invalid)}. Valid domains: {sorted(VDD_DOMAINS)}")

    resolved_data_dir = Path(data_dir) if data_dir is not None else Path("data")

    # --- Load and preprocess data ---
    logger.info("Loading and preprocessing datasets...")
    loaded = load_domains(
        domains,
        ["train", "val"],
        data_dir=resolved_data_dir,
        augment=augment,
        input_size=input_size,
        normalize=normalize,
    )
    num_classes: dict[str, int] = {}
    train_datasets = {}
    val_datasets = {}
    for domain in domains:
        train_ds = cast(ImageFolder, loaded[domain]["train"])
        val_ds = cast(Dataset, loaded[domain]["val"])
        num_classes[domain] = len(train_ds.classes)
        train_datasets[domain] = make_dataloader(train_ds, batch_size=batch_size, shuffle=shuffle, seed=seed)
        val_datasets[domain] = make_dataloader(val_ds, batch_size=batch_size)

    logger.info(f"num_classes: {num_classes}")

    # --- Build model ---
    logger.info(f"Building model from: backbone={backbone_class_path}, head={head_class_path}")
    model = build_model(
        domains=domains,
        num_classes=num_classes,
        backbone_class_path=backbone_class_path,
        backbone_args=backbone_args,
        head_class_path=head_class_path,
        head_args=head_args,
    )

    # --- W&B ---
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "domains": domains,
            "data_dir": str(resolved_data_dir),
            "batch_size": batch_size,
            "shuffle": shuffle,
            "augment": augment,
            "input_size": input_size,
            "normalize": normalize,
            "seed": seed,
            "backbone": {"class_path": backbone_class_path, "args": backbone_args or {}},
            "head": {"class_path": head_class_path, "args": head_args or {}},
            "total_steps": total_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "warmup_steps": warmup_steps,
            "backbone_steps": backbone_steps,
            "label_smoothing": label_smoothing,
            "val_every_n_steps": val_every_n_steps,
            "output_dir": str(output_dir) if output_dir is not None else None,
            "device": str(device),
        },
    )

    # --- Train ---
    logger.info("Starting training...")
    final_metrics = train(
        model,
        train_datasets,
        val_datasets,
        total_steps=total_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        backbone_steps=backbone_steps,
        label_smoothing=label_smoothing,
        val_every_n_steps=val_every_n_steps,
        output_dir=output_dir,
        wandb_run=run,
        device=device,
    )
    run.finish()
    return final_metrics


app = typer.Typer()


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to pipeline YAML configuration file."),
    ] = Path("configs/pipeline.yaml"),
) -> None:
    """Run a single pipeline defined by a YAML configuration file."""
    logger.add("logs/pipeline_{time}.log")
    logger.info(f"Loading pipeline config from {config}")
    cfg = cast(DictConfig, OmegaConf.load(config))
    backbone_args_raw = OmegaConf.to_container(cfg.model.backbone.args, resolve=True)
    head_args_raw = OmegaConf.to_container(cfg.model.head.args, resolve=True)
    backbone_args = cast(dict[str, Any], backbone_args_raw or {})
    head_args = cast(dict[str, Any], head_args_raw or {})
    kwargs = {
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
        "backbone_steps": cfg.train.backbone_steps,
        "label_smoothing": cfg.train.label_smoothing,
        "val_every_n_steps": cfg.train.val_every_n_steps,
        "output_dir": cfg.train.output_dir,
        "wandb_project": cfg.wandb.project,
        "wandb_run_name": cfg.wandb.run_name,
        "device": cfg.train.device,
    }
    results = run_pipeline(**kwargs)
    logger.info(f"Pipeline complete. Metrics: {results}")


if __name__ == "__main__":
    app()
