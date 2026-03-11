"""Tests for slimmed_experts.data."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from slimmed_experts.data import (
    MOBILENET_INPUT_SIZE,
    VDD_DOMAINS,
    load_domain,
    load_domains,
    make_dataloader,
    mobilenet_transform,
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


class TestMobilenetTransform:
    def test_output_shape(self):
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        out = mobilenet_transform()(img)
        assert out.shape == (3, MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE)

    def test_output_dtype(self):
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        out = mobilenet_transform()(img)
        assert out.dtype == torch.float32

    def test_value_range_normalized(self):
        # An all-zero image produces (-mean/std) per channel after normalisation.
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        expected = [-m / s for m, s in zip(imagenet_mean, imagenet_std)]

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        out = mobilenet_transform()(img)

        for c, exp in enumerate(expected):
            assert abs(float(out[c, 0, 0]) - exp) < 1e-4

    def test_augment_flag_does_not_change_shape(self):
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        out = mobilenet_transform(augment=True)(img)
        assert out.shape == (3, MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE)


class TestMakeDataloader:
    def test_batch_size(self, data_dir):
        ds = load_domain("aircraft", "train", data_dir=data_dir)
        loader = make_dataloader(ds, batch_size=1)
        images, labels = next(iter(loader))
        assert images.shape[0] == 1

    def test_output_shape(self, data_dir):
        ds = load_domain("aircraft", "train", data_dir=data_dir)
        loader = make_dataloader(ds, batch_size=2)
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE)
        assert labels.shape == (2,)

    def test_no_shuffle_by_default(self, data_dir):
        ds = load_domain("aircraft", "train", data_dir=data_dir)
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
        ds = load_domain("aircraft", "train", data_dir=data_dir)
        kwargs = dict(batch_size=8, shuffle=True, seed=42)
        labels_1 = [lbl for _, lbl in make_dataloader(ds, **kwargs)]
        labels_2 = [lbl for _, lbl in make_dataloader(ds, **kwargs)]
        assert all(torch.equal(a, b) for a, b in zip(labels_1, labels_2))


class TestLoadDomain:
    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            load_domain("not_a_domain", "train")

    def test_single_split_returns_dataset(self, data_dir):
        result = load_domain("aircraft", "train", data_dir=data_dir)
        assert isinstance(result, Dataset)

    def test_multi_split_returns_dict(self, data_dir):
        result = load_domain("aircraft", ["train", "test"], data_dir=data_dir)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"train", "test"}
        assert all(isinstance(v, Dataset) for v in result.values())

    def test_all_valid_domains_accepted(self, data_dir):
        for domain in VDD_DOMAINS:
            result = load_domain(domain, "train", data_dir=data_dir)
            assert isinstance(result, Dataset)

    def test_labeled_split_has_correct_num_classes(self, data_dir):
        ds = load_domain("aircraft", "train", data_dir=data_dir)
        # The fixture creates 2 class dirs per labeled split
        assert len(ds.classes) == 2  # type: ignore[attr-defined]

    def test_test_split_returns_minus_one_labels(self, data_dir):
        ds = load_domain("aircraft", "test", data_dir=data_dir)
        _, label = ds[0]
        assert label == -1


class TestLoadDomains:
    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            load_domains(["invalid_domain"], "train")

    def test_returns_correct_keys(self, data_dir):
        result = load_domains(["aircraft", "dtd"], "train", data_dir=data_dir)
        assert set(result.keys()) == {"aircraft", "dtd"}

    def test_multi_split_values_are_dicts(self, data_dir):
        result = load_domains(["aircraft", "dtd"], ["train", "test"], data_dir=data_dir)
        for ds_dict in result.values():
            assert isinstance(ds_dict, dict)
            assert set(ds_dict.keys()) == {"train", "test"}
