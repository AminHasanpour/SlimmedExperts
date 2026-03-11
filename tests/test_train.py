"""Tests for slimmed_experts.train."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from slimmed_experts.model import MobileNetV2MultiHead
from slimmed_experts.train import _evaluate, _save_checkpoint, train

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataloader(
    num_batches: int = 3,
    batch_size: int = 4,
    height: int = 32,
    width: int = 32,
    num_classes: int = 5,
) -> DataLoader:
    """Return a tiny batched DataLoader with synthetic BCHW images."""
    n = num_batches * batch_size
    images = torch.rand(n, 3, height, width)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_model():
    """Tiny two-domain MobileNetV2 with small-input stride for 32x32 speed."""
    return MobileNetV2MultiHead(
        domains=["d1", "d2"],
        num_classes={"d1": 5, "d2": 8},
        width_mult=0.35,
        small_input=True,
    )


# ---------------------------------------------------------------------------
# _save_checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_file_is_created(self, tmp_path, tiny_model):
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        path = tmp_path / "ckpt.pt"

        _save_checkpoint(path, tiny_model, opt, step=5, metrics={"acc/d1": 0.9})

        assert path.exists()

    def test_checkpoint_has_expected_keys(self, tmp_path, tiny_model):
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        path = tmp_path / "ckpt.pt"
        _save_checkpoint(path, tiny_model, opt, step=1, metrics={})

        ckpt = torch.load(path, weights_only=True)
        assert set(ckpt.keys()) == {"step", "model_state_dict", "optimizer_state_dict", "metrics"}

    def test_step_is_stored_correctly(self, tmp_path, tiny_model):
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        path = tmp_path / "ckpt.pt"
        _save_checkpoint(path, tiny_model, opt, step=42, metrics={})

        ckpt = torch.load(path, weights_only=True)
        assert ckpt["step"] == 42

    def test_metrics_are_stored_correctly(self, tmp_path, tiny_model):
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        path = tmp_path / "ckpt.pt"
        metrics = {"acc/d1": 0.75, "loss/d1": 0.3}
        _save_checkpoint(path, tiny_model, opt, step=1, metrics=metrics)

        ckpt = torch.load(path, weights_only=True)
        assert ckpt["metrics"] == metrics

    def test_state_dict_loadable_into_fresh_model(self, tmp_path, tiny_model):
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        path = tmp_path / "ckpt.pt"
        _save_checkpoint(path, tiny_model, opt, step=1, metrics={})

        ckpt = torch.load(path, weights_only=True)
        fresh = MobileNetV2MultiHead(
            domains=["d1", "d2"],
            num_classes={"d1": 5, "d2": 8},
            width_mult=0.35,
            small_input=True,
        )
        fresh.load_state_dict(ckpt["model_state_dict"])  # must not raise


# ---------------------------------------------------------------------------
# _evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_acc_and_loss_for_each_domain(self, tiny_model):
        criterion = nn.CrossEntropyLoss()
        val_ds = {
            "d1": _make_dataloader(num_classes=5),
            "d2": _make_dataloader(num_classes=8),
        }

        metrics = _evaluate(tiny_model, val_ds, criterion, torch.device("cpu"), ["d1", "d2"])

        for domain in ("d1", "d2"):
            assert f"acc/{domain}" in metrics
            assert f"loss/{domain}" in metrics

    def test_acc_is_between_zero_and_one(self, tiny_model):
        criterion = nn.CrossEntropyLoss()
        val_ds = {"d1": _make_dataloader(num_classes=5)}

        metrics = _evaluate(tiny_model, val_ds, criterion, torch.device("cpu"), ["d1"])

        assert 0.0 <= metrics["acc/d1"] <= 1.0

    def test_loss_is_nonnegative(self, tiny_model):
        criterion = nn.CrossEntropyLoss()
        val_ds = {"d1": _make_dataloader(num_classes=5)}

        metrics = _evaluate(tiny_model, val_ds, criterion, torch.device("cpu"), ["d1"])

        assert metrics["loss/d1"] >= 0.0

    def test_domain_absent_from_val_datasets_is_skipped(self, tiny_model):
        criterion = nn.CrossEntropyLoss()
        val_ds = {"d1": _make_dataloader(num_classes=5)}  # "d2" intentionally omitted

        metrics = _evaluate(tiny_model, val_ds, criterion, torch.device("cpu"), ["d1", "d2"])

        assert "acc/d1" in metrics
        assert "acc/d2" not in metrics


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrain:
    def test_invalid_optimizer_raises_value_error(self, tiny_model):
        train_ds = {"d1": _make_dataloader(num_classes=5)}
        val_ds = {"d1": _make_dataloader(num_classes=5)}

        # ValueError is raised before wandb.init(), so no mocking needed.
        with pytest.raises(ValueError, match="Unknown optimizer"):
            train(
                tiny_model,
                train_ds,
                val_ds,
                total_steps=1,
                learning_rate=1e-3,
                optimizer="bad_opt",
            )

    def test_returns_metrics_dict_with_expected_keys(self, tiny_model):
        train_ds = {"d1": _make_dataloader(num_classes=5)}
        val_ds = {"d1": _make_dataloader(num_classes=5)}

        result = train(
            tiny_model,
            train_ds,
            val_ds,
            total_steps=2,
            learning_rate=1e-3,
            val_every_n_steps=2,
            device="cpu",
        )

        assert isinstance(result, dict)
        assert "acc/d1" in result
        assert "loss/d1" in result

    def test_creates_last_and_best_checkpoints(self, tiny_model, tmp_path):
        train_ds = {"d1": _make_dataloader(num_classes=5)}
        val_ds = {"d1": _make_dataloader(num_classes=5)}

        train(
            tiny_model,
            train_ds,
            val_ds,
            total_steps=2,
            learning_rate=1e-3,
            val_every_n_steps=2,
            output_dir=tmp_path,
            device="cpu",
        )

        assert (tmp_path / "last.pt").exists()
        assert (tmp_path / "best.pt").exists()

    def test_no_checkpoint_when_output_dir_is_none(self, tiny_model, tmp_path):
        train_ds = {"d1": _make_dataloader(num_classes=5)}
        val_ds = {"d1": _make_dataloader(num_classes=5)}

        train(
            tiny_model,
            train_ds,
            val_ds,
            total_steps=2,
            learning_rate=1e-3,
            val_every_n_steps=2,
            output_dir=None,
            device="cpu",
        )

        assert not (tmp_path / "last.pt").exists()

    def test_sgd_optimizer_runs_successfully(self, tiny_model):
        train_ds = {"d1": _make_dataloader(num_classes=5)}
        val_ds = {"d1": _make_dataloader(num_classes=5)}

        result = train(
            tiny_model,
            train_ds,
            val_ds,
            total_steps=2,
            learning_rate=1e-3,
            optimizer="sgd",
            val_every_n_steps=2,
            device="cpu",
        )

        assert isinstance(result, dict)

    def test_multi_domain_round_robin(self, tiny_model):
        train_ds = {
            "d1": _make_dataloader(num_classes=5),
            "d2": _make_dataloader(num_classes=8),
        }
        val_ds = {
            "d1": _make_dataloader(num_classes=5),
            "d2": _make_dataloader(num_classes=8),
        }

        result = train(
            tiny_model,
            train_ds,
            val_ds,
            total_steps=4,
            learning_rate=1e-3,
            val_every_n_steps=4,
            device="cpu",
        )

        assert "acc/d1" in result
        assert "acc/d2" in result
