"""Training loop for multi-domain MobileNetV2 models."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.optim import Adam, SGD

import wandb

from slimmed_experts.model import MultiHeadModel


def _tf_batch_to_torch(
    images: Any,
    labels: Any,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Convert a TensorFlow batch to PyTorch tensors.

    Converts images from BHWC (TensorFlow convention) to BCHW (PyTorch
    convention) and moves both tensors to the requested device.

    Args:
        images: TF eagertensor of shape ``(B, H, W, C)`` with dtype ``float32``.
        labels: TF eagertensor of shape ``(B,)`` with integer labels.
        device: Target PyTorch device.

    Returns:
        ``(images, labels)`` as PyTorch tensors on *device*.
    """
    img = torch.from_numpy(images.numpy()).permute(0, 3, 1, 2).to(device)
    lbl = torch.from_numpy(labels.numpy()).long().to(device)
    return img, lbl


def _save_checkpoint(
    path: Path,
    model: MultiHeadModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, float],
) -> None:
    """Save a training checkpoint.

    Args:
        path: File path to write the checkpoint to.
        model: The model whose state dict is saved.
        optimizer: The optimizer whose state dict is saved.
        step: Current global training step.
        metrics: Validation metrics at this checkpoint.
    """
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def _evaluate(
    model: MultiHeadModel,
    val_datasets: dict[str, Any],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    domains: list[str],
) -> dict[str, float]:
    """Evaluate the model on all validation datasets.

    Args:
        model: The model to evaluate (sets to eval mode).
        val_datasets: Mapping ``{domain: preprocessed tf.data.Dataset}``.
        criterion: Loss function.
        device: PyTorch device.
        domains: Ordered list of domain names to evaluate.

    Returns:
        Flat dict ``{"acc/{domain}": float, "loss/{domain}": float, ...}``.
    """
    model.eval()
    metrics: dict[str, float] = {}
    with torch.no_grad():
        for domain in domains:
            if domain not in val_datasets:
                continue
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for images_tf, labels_tf in val_datasets[domain]:
                images, labels = _tf_batch_to_torch(images_tf, labels_tf, device)
                logits = model(images, domain)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                total_loss += loss.item() * labels.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            metrics[f"acc/{domain}"] = total_correct / total_samples if total_samples > 0 else 0.0
            metrics[f"loss/{domain}"] = total_loss / total_samples if total_samples > 0 else 0.0
    return metrics


def train(
    model: MultiHeadModel,
    train_datasets: dict[str, Any],
    val_datasets: dict[str, Any],
    *,
    total_steps: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    val_every_n_steps: int = 100,
    output_dir: str | Path | None = None,
    wandb_project: str = "slimmed-experts",
    wandb_run_name: str | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Train a multi-head model on multiple domains using round-robin batching.

    Datasets must be fully preprocessed and batched before being passed in.
    Training cycles through domains in round-robin order for *total_steps*
    gradient updates and evaluates on *val_datasets* every *val_every_n_steps*
    steps.

    Args:
        model: A :class:`~slimmed_experts.model.MultiHeadModel` instance.
        train_datasets: Mapping ``{domain: preprocessed tf.data.Dataset}``
            for training.  Datasets must be pre-batched, pre-shuffled, and ready
            to iterate.
        val_datasets: Mapping ``{domain: preprocessed tf.data.Dataset}``
            for validation.  Same requirements as *train_datasets*.
        total_steps: Total number of gradient update steps across all domains.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularisation coefficient.
        optimizer: ``"adam"`` or ``"sgd"``.
        val_every_n_steps: Run validation every this many steps.
        output_dir: Directory to write ``best.pt`` and ``last.pt``.  ``None``
            disables checkpoint saving.
        wandb_project: W&B project name.
        wandb_run_name: W&B run display name (``None`` = W&B auto-generates one).
        device: PyTorch device string or object (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        Dictionary of final validation metrics
        ``{"acc/{domain}": float, "loss/{domain}": float, ...}``.

    Raises:
        ValueError: If *optimizer* is not ``"adam"`` or ``"sgd"``.
    """
    device_ = torch.device(device)
    model = model.to(device_)
    domains = list(train_datasets.keys())

    # --- Optimizer ---
    if optimizer == "adam":
        opt: torch.optim.Optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose 'adam' or 'sgd'.")

    criterion = nn.CrossEntropyLoss()

    # --- W&B ---
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "total_steps": total_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "val_every_n_steps": val_every_n_steps,
            "domains": domains,
        },
    )

    # --- Output directory ---
    out: Path | None = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    # --- Infinite round-robin iterators ---
    train_iters = {d: iter(itertools.cycle(train_datasets[d])) for d in domains}
    domain_cycle = itertools.cycle(domains)

    best_val_acc: float = -1.0
    final_metrics: dict[str, float] = {}

    logger.info(f"Starting training: total_steps={total_steps}, domains={domains}, device={device_}")

    model.train()
    for step in range(1, total_steps + 1):
        domain = next(domain_cycle)
        images_tf, labels_tf = next(train_iters[domain])
        images, labels = _tf_batch_to_torch(images_tf, labels_tf, device_)

        opt.zero_grad()
        logits = model(images, domain)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        wandb.log({f"train/loss/{domain}": loss.item(), f"train/acc/{domain}": acc}, step=step)

        if step % 100 == 0:
            logger.info(f"Step {step}/{total_steps} | domain={domain} | loss={loss.item():.4f} | acc={acc:.4f}")

        # --- Validation ---
        if step % val_every_n_steps == 0 or step == total_steps:
            val_metrics = _evaluate(model, val_datasets, criterion, device_, domains)
            final_metrics = val_metrics

            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)

            mean_val_acc = float(np.mean([v for k, v in val_metrics.items() if k.startswith("acc/")]))
            logger.info(f"Step {step} val | mean_acc={mean_val_acc:.4f} | {val_metrics}")

            if out is not None:
                _save_checkpoint(out / "last.pt", model, opt, step, val_metrics)
                logger.info(f"Saved last checkpoint → {out / 'last.pt'}")

                if mean_val_acc > best_val_acc:
                    best_val_acc = mean_val_acc
                    _save_checkpoint(out / "best.pt", model, opt, step, val_metrics)
                    logger.info(f"New best val acc={best_val_acc:.4f} → saved best checkpoint → {out / 'best.pt'}")

            model.train()

    run.finish()
    logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    return final_metrics
