"""Training loop for multi-domain models."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from slimmed_experts.models.model import MultiHeadModel


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
    val_datasets: dict[str, DataLoader],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    domains: list[str],
) -> dict[str, float]:
    """Evaluate the model on all validation datasets.

    Args:
        model: The model to evaluate (sets to eval mode).
        val_datasets: Mapping ``{domain: DataLoader}``.
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
            for images, labels in val_datasets[domain]:
                images, labels = images.to(device), labels.to(device)
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
    train_datasets: dict[str, DataLoader],
    val_datasets: dict[str, DataLoader],
    *,
    total_steps: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    scheduler: str | None = "cosine",
    warmup_steps: int = 0,
    label_smoothing: float = 0.0,
    val_every_n_steps: int = 100,
    output_dir: str | Path | None = None,
    wandb_run: Any | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Train a multi-head model on multiple domains using round-robin batching.

    Datasets must be wrapped in :class:`~torch.utils.data.DataLoader` before
    being passed in.  Training cycles through domains in round-robin order for
    *total_steps* gradient updates and evaluates on *val_datasets* every
    *val_every_n_steps* steps.

    Args:
        model: A :class:`~slimmed_experts.models.model.MultiHeadModel` instance.
        train_datasets: Mapping ``{domain: DataLoader}`` for training.
        val_datasets: Mapping ``{domain: DataLoader}`` for validation.
        total_steps: Total number of gradient update steps across all domains.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularisation coefficient.
        optimizer: ``"adam"`` or ``"sgd"``.
        scheduler: LR scheduler to use after warmup (``"cosine"`` or ``None``).
        warmup_steps: Number of warmup steps. Learning rate increases linearly
            from 0 to ``learning_rate`` over this many initial steps.
        label_smoothing: Label smoothing for
            :class:`~torch.nn.CrossEntropyLoss`.
        val_every_n_steps: Run validation every this many steps.
        output_dir: Directory to write ``best.pt`` and ``last.pt``.  ``None``
            disables checkpoint saving.
        wandb_run: Active W&B run object to log metrics to.  Pass ``None`` to
            disable W&B logging.
        device: PyTorch device string or object (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        Dictionary of final validation metrics
        ``{"acc/{domain}": float, "loss/{domain}": float, ...}``.

    Raises:
        ValueError: If *optimizer* is not ``"adam"`` or ``"sgd"``.
        ValueError: If *scheduler* is not ``"cosine"`` or ``None``.
        ValueError: If *warmup_steps* is negative or exceeds *total_steps*.
        ValueError: If *label_smoothing* is not in ``[0.0, 1.0)``.
    """
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError(f"warmup_steps must be in [0, total_steps]. Got {warmup_steps}.")
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError(f"label_smoothing must be in [0.0, 1.0). Got {label_smoothing}.")

    device_ = torch.device(device)
    model = model.to(device_)
    domains = list(train_datasets.keys())

    # --- Optimizer ---
    if optimizer == "adam":
        opt: torch.optim.Optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose 'adam' or 'sgd'.")

    if scheduler not in {"cosine", None}:
        raise ValueError(f"Unknown scheduler '{scheduler}'. Choose 'cosine' or None.")

    if warmup_steps > 0:
        for group in opt.param_groups:
            group["lr"] = 0.0

    def _lr_at_step(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx <= warmup_steps:
            return learning_rate * (step_idx / warmup_steps)

        if scheduler is None:
            return learning_rate

        if total_steps == warmup_steps:
            return learning_rate

        decay_steps = total_steps - warmup_steps
        progress = (step_idx - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
        images, labels = next(train_iters[domain])
        images, labels = images.to(device_), labels.to(device_)

        opt.zero_grad()
        logits = model(images, domain)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        if warmup_steps > 0 or scheduler == "cosine":
            new_lr = _lr_at_step(step)
            for group in opt.param_groups:
                group["lr"] = new_lr

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        if wandb_run is not None:
            lr = opt.param_groups[0]["lr"]
            wandb_run.log(
                {f"train/loss/{domain}": loss.item(), f"train/acc/{domain}": acc, "train/lr": lr},
                step=step,
            )

        if step % 100 == 0:
            logger.info(f"Step {step}/{total_steps} | domain={domain} | loss={loss.item():.4f} | acc={acc:.4f}")

        # --- Validation ---
        if step % val_every_n_steps == 0 or step == total_steps:
            val_metrics = _evaluate(model, val_datasets, criterion, device_, domains)
            final_metrics = val_metrics

            if wandb_run is not None:
                wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)

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

    logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    return final_metrics
