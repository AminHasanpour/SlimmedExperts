"""End-to-end training pipeline for multi-domain MobileNetV2 models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, cast

import typer
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from slimmed_experts.data import VDD_DOMAINS, load_domain, make_dataloader
from slimmed_experts.model import MobileNetV2MultiHead
from slimmed_experts.train import train


def run_pipeline(
    domains: list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
    # Data preprocessing
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    seed: int | None = None,
    # Model
    width_mult: float = 1.0,
    small_input: bool = False,
    # Training
    total_steps: int = 1000,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    scheduler: str | None = "cosine",
    val_every_n_steps: int = 100,
    output_dir: str | Path | None = None,
    wandb_project: str = "slimmed-experts",
    wandb_run_name: str | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Run the full training pipeline end-to-end.

    Loads data for each domain, builds a :class:`~slimmed_experts.model.MobileNetV2MultiHead`
    model, and trains it with :func:`~slimmed_experts.train.train`.

    Args:
        domains: List of VDD domain names to train on (must be a subset of
            :data:`~slimmed_experts.data.VDD_DOMAINS`).
        data_dir: Root directory containing the domain folders.  Defaults to
            ``"data"``.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the training split before batching.
        augment: If ``True``, applies random horizontal flips during training.
        seed: Random seed for reproducible data shuffling.
        width_mult: MobileNetV2 width multiplier (e.g. 0.35, 0.5, 0.75, 1.0).
        small_input: If ``True``, sets the first conv layer's stride to ``(1, 1)``
            (useful for small-resolution datasets such as CIFAR-100).
        total_steps: Total number of gradient update steps across all domains.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularisation coefficient.
        optimizer: ``"adam"`` or ``"sgd"``.
        scheduler: LR scheduler to use (``"cosine"`` or ``None``).
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
    _train_ds_raw: dict[str, ImageFolder] = {}
    train_datasets = {}
    val_datasets = {}
    for domain in domains:
        train_ds = cast(ImageFolder, load_domain(domain, "train", data_dir=resolved_data_dir, augment=augment))
        val_ds = cast(Dataset, load_domain(domain, "val", data_dir=resolved_data_dir))
        _train_ds_raw[domain] = train_ds
        train_datasets[domain] = make_dataloader(train_ds, batch_size=batch_size, shuffle=shuffle, seed=seed)
        val_datasets[domain] = make_dataloader(val_ds, batch_size=batch_size)

    # --- Infer num_classes per domain from the loaded train datasets ---
    logger.info("Inferring number of classes per domain...")
    num_classes: dict[str, int] = {d: len(_train_ds_raw[d].classes) for d in domains}
    logger.info(f"num_classes: {num_classes}")

    # --- Build model ---
    logger.info(f"Building MobileNetV2MultiHead (width_mult={width_mult}, small_input={small_input})...")
    model = MobileNetV2MultiHead(
        domains=domains,
        num_classes=num_classes,
        width_mult=width_mult,
        small_input=small_input,
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
            "seed": seed,
            "width_mult": width_mult,
            "small_input": small_input,
            "total_steps": total_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "scheduler": scheduler,
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
    kwargs = {
        "domains": list(cfg.data.load.domains),
        "data_dir": cfg.data.load.data_dir,
        "batch_size": cfg.data.preprocess.batch_size,
        "shuffle": cfg.data.preprocess.shuffle,
        "augment": cfg.data.preprocess.augment,
        "seed": cfg.data.preprocess.seed,
        "width_mult": cfg.model.width_mult,
        "small_input": cfg.model.small_input,
        "total_steps": cfg.train.total_steps,
        "learning_rate": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
        "optimizer": cfg.train.optimizer,
        "scheduler": cfg.train.scheduler,
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
