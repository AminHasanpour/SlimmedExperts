"""Tests for slimmed_experts.data."""

from __future__ import annotations

import pytest
import tensorflow as tf

import slimmed_experts.data as data_module
from slimmed_experts.data import (
    MOBILENET_INPUT_SIZE,
    VDD_DOMAINS,
    load_domain,
    load_domains,
    preprocess_domain,
    preprocess_for_mobilenet,
)


def _make_ds(num_samples: int = 12, height: int = 32, width: int = 32, channels: int = 3) -> tf.data.Dataset:
    images = tf.random.uniform((num_samples, height, width, channels), 0, 255, dtype=tf.float32)
    labels = tf.random.uniform((num_samples,), 0, 10, dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices((images, labels))


@pytest.fixture()
def mock_tfds_load(monkeypatch):
    """Replace tfds.load with a function returning a small synthetic dataset."""
    monkeypatch.setattr(data_module.tfds, "load", lambda *a, **kw: _make_ds())


class TestPreprocessForMobilenet:
    def test_output_shape(self):
        image = tf.random.uniform((64, 64, 3), 0, 255)
        out_image, _ = preprocess_for_mobilenet(image, tf.constant(0))
        assert out_image.shape == (MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE, 3)

    def test_output_dtype(self):
        image = tf.random.uniform((64, 64, 3), 0, 255)
        out_image, _ = preprocess_for_mobilenet(image, tf.constant(0))
        assert out_image.dtype == tf.float32

    def test_value_range(self):
        image = tf.constant([[[0.0, 0.0, 0.0]] * 64] * 64)
        out_image, _ = preprocess_for_mobilenet(image, tf.constant(0))
        assert float(tf.reduce_min(out_image)) >= -1.0
        assert float(tf.reduce_max(out_image)) <= 1.0

    def test_label_unchanged(self):
        label = tf.constant(7)
        _, out_label = preprocess_for_mobilenet(tf.random.uniform((16, 16, 3)), label)
        assert int(out_label) == 7


class TestPreprocessDomain:
    def test_output_shape(self):
        ds = _make_ds(num_samples=8)
        result = preprocess_domain(ds, batch_size=4)
        batch_images, batch_labels = next(iter(result))
        assert batch_images.shape == (4, MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE, 3)
        assert batch_labels.shape == (4,)

    def test_batch_size(self):
        ds = _make_ds(num_samples=10)
        result = preprocess_domain(ds, batch_size=5)
        batch_images, _ = next(iter(result))
        assert batch_images.shape[0] == 5

    def test_no_shuffle_by_default(self):
        ds = _make_ds(num_samples=8)
        # Should not raise and should yield deterministic output when shuffle=False.
        r1 = list(preprocess_domain(ds, batch_size=8, shuffle=False).as_numpy_iterator())
        r2 = list(preprocess_domain(ds, batch_size=8, shuffle=False).as_numpy_iterator())
        import numpy as np

        assert np.allclose(r1[0][0], r2[0][0])

    def test_shuffle_with_seed_is_reproducible(self):
        ds = _make_ds(num_samples=20)
        kwargs = dict(batch_size=5, shuffle=True, shuffle_buffer_size=20, seed=42)
        labels_1 = next(iter(preprocess_domain(ds, **kwargs)))[1].numpy().tolist()
        labels_2 = next(iter(preprocess_domain(ds, **kwargs)))[1].numpy().tolist()
        assert labels_1 == labels_2

    def test_shuffle_without_seed_differs(self):
        # With no seed two passes are very unlikely to produce the same order.
        ds = _make_ds(num_samples=50)
        kwargs = dict(batch_size=50, shuffle=True, shuffle_buffer_size=50, seed=None)
        labels_1 = next(iter(preprocess_domain(ds, **kwargs)))[1].numpy().tolist()
        labels_2 = next(iter(preprocess_domain(ds, **kwargs)))[1].numpy().tolist()
        assert labels_1 != labels_2


class TestLoadDomain:
    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            load_domain("not_a_domain", "train")

    def test_single_split_returns_dataset(self, mock_tfds_load):
        result = load_domain("aircraft", "train")
        assert isinstance(result, tf.data.Dataset)

    def test_multi_split_returns_dict(self, mock_tfds_load):
        result = load_domain("aircraft", ["train", "test"])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"train", "test"}
        assert all(isinstance(v, tf.data.Dataset) for v in result.values())

    def test_all_valid_domains_accepted(self, mock_tfds_load):
        for domain in VDD_DOMAINS:
            result = load_domain(domain, "train")
            assert isinstance(result, tf.data.Dataset)


class TestLoadDomains:
    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            load_domains(["invalid_domain"], "train")

    def test_returns_correct_keys(self, mock_tfds_load):
        result = load_domains(["aircraft", "dtd"], "train")
        assert set(result.keys()) == {"aircraft", "dtd"}

    def test_multi_split_values_are_dicts(self, mock_tfds_load):
        result = load_domains(["aircraft", "dtd"], ["train", "test"])
        for ds_dict in result.values():
            assert isinstance(ds_dict, dict)
            assert set(ds_dict.keys()) == {"train", "test"}
