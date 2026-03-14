"""Tests for slimmed_experts.data."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from slimmed_experts.data import (
    DEFAULT_INPUT_SIZE,
    VDD_DOMAINS,
    _build_transform,
    _load_domain,
    load_domains,
    make_dataloader,
)


def _write_jpeg(path, height: int = 8, width: int = 8) -> None:
    img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    img.save(path)


@pytest.fixture()
def data_dir(tmp_path):
    """Create a minimal fake VDD directory structure with small JPEG images."""
    for domain in VDD_DOMAINS:
        for split in ("train", "val"):
            for cls in ("0001", "0002"):
                cls_dir = tmp_path / domain / split / cls
                cls_dir.mkdir(parents=True)
                _write_jpeg(cls_dir / "000001.jpg")
        test_dir = tmp_path / domain / "test"
        test_dir.mkdir(parents=True)
        _write_jpeg(test_dir / "000001.jpg")
    return tmp_path


class TestBuildTransform:
    def test_output_shape(self):
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        out = _build_transform(DEFAULT_INPUT_SIZE)(img)
        assert out.shape == (3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)

    def test_output_dtype(self):
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        out = _build_transform(DEFAULT_INPUT_SIZE)(img)
        assert out.dtype == torch.float32

    def test_value_range_normalized_with_given_stats(self):
        mean = [0.5, 0.4, 0.3]
        std = [0.2, 0.25, 0.5]
        expected = [-m / s for m, s in zip(mean, std)]

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        out = _build_transform(DEFAULT_INPUT_SIZE, mean=mean, std=std)(img)

        for c, exp in enumerate(expected):
            assert abs(float(out[c, 0, 0]) - exp) < 1e-4

    def test_augment_flag_does_not_change_shape(self):
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        out = _build_transform(DEFAULT_INPUT_SIZE, augment=True)(img)
        assert out.shape == (3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)


class TestMakeDataloader:
    def test_batch_size(self, data_dir):
        ds = load_domains(["aircraft"], ["train"], data_dir=data_dir)["aircraft"]["train"]
        loader = make_dataloader(ds, batch_size=1)
        images, labels = next(iter(loader))
        assert images.shape[0] == 1

    def test_output_shape(self, data_dir):
        ds = load_domains(["aircraft"], ["train"], data_dir=data_dir)["aircraft"]["train"]
        loader = make_dataloader(ds, batch_size=2)
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)
        assert labels.shape == (2,)

    def test_no_shuffle_by_default(self, data_dir):
        ds = load_domains(["aircraft"], ["train"], data_dir=data_dir)["aircraft"]["train"]
        r1 = [lbl for _, lbl in make_dataloader(ds, batch_size=4, shuffle=False)]
        r2 = [lbl for _, lbl in make_dataloader(ds, batch_size=4, shuffle=False)]
        assert all(torch.equal(a, b) for a, b in zip(r1, r2))

    def test_shuffle_with_seed_is_reproducible(self, data_dir):
        # Build a larger split so shuffle has something to permute
        for cls in ("0003", "0004"):
            cls_dir = data_dir / "aircraft" / "train" / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                _write_jpeg(cls_dir / f"00000{i}.jpg")
        ds = load_domains(["aircraft"], ["train"], data_dir=data_dir)["aircraft"]["train"]
        kwargs = dict(batch_size=8, shuffle=True, seed=42)
        labels_1 = [lbl for _, lbl in make_dataloader(ds, **kwargs)]
        labels_2 = [lbl for _, lbl in make_dataloader(ds, **kwargs)]
        assert all(torch.equal(a, b) for a, b in zip(labels_1, labels_2))


class TestLoadDomain:
    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            _load_domain("not_a_domain", ["train"])

    def test_split_must_be_list(self, data_dir):
        with pytest.raises(TypeError, match="split must be a list"):
            _load_domain("aircraft", "train", data_dir=data_dir)  # type: ignore[arg-type]

    def test_returns_dict_for_requested_splits(self, data_dir):
        result = _load_domain("aircraft", ["train", "test"], data_dir=data_dir)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"train", "test"}
        assert all(isinstance(v, Dataset) for v in result.values())

    def test_all_valid_domains_accepted(self, data_dir):
        for domain in VDD_DOMAINS:
            result = _load_domain(domain, ["train"], data_dir=data_dir)
            assert isinstance(result["train"], Dataset)

    def test_labeled_split_has_correct_num_classes(self, data_dir):
        ds = _load_domain("aircraft", ["train"], data_dir=data_dir)["train"]
        # The fixture creates 2 class dirs per labeled split
        assert len(ds.classes) == 2  # type: ignore[attr-defined]

    def test_test_split_returns_minus_one_labels(self, data_dir):
        ds = _load_domain("aircraft", ["test"], data_dir=data_dir)["test"]
        _, label = ds[0]
        assert label == -1

    def test_input_size_parameter_changes_output_shape(self, data_dir):
        ds = _load_domain("aircraft", ["train"], data_dir=data_dir, input_size=32)["train"]
        image, _ = ds[0]
        assert image.shape == (3, 32, 32)

    def test_normalize_parameter_applies_normalization(self, data_dir):
        ds = _load_domain("aircraft", ["train"], data_dir=data_dir, normalize=True)["train"]
        image, _ = ds[0]
        assert torch.isfinite(image).all()


class TestLoadDomains:
    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            load_domains(["invalid_domain"], ["train"])

    def test_returns_correct_keys(self, data_dir):
        result = load_domains(["aircraft", "dtd"], ["train"], data_dir=data_dir)
        assert set(result.keys()) == {"aircraft", "dtd"}
        assert set(result["aircraft"].keys()) == {"train"}

    def test_multi_split_values_are_dicts(self, data_dir):
        result = load_domains(["aircraft", "dtd"], ["train", "test"], data_dir=data_dir)
        for ds_dict in result.values():
            assert isinstance(ds_dict, dict)
            assert set(ds_dict.keys()) == {"train", "test"}
