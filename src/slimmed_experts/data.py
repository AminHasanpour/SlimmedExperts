"""Data loading and preprocessing utilities for the Visual Domain Decathlon (VDD) dataset."""

from __future__ import annotations

import os

import tensorflow as tf
import tensorflow_datasets as tfds


# MobileNetV2 expected input spatial size (height and width).
MOBILENET_INPUT_SIZE: int = 224

# List of valid VDD domains.
VDD_DOMAINS: list[str] = [
    "aircraft",
    "cifar100",
    "daimlerpedcls",
    "dtd",
    "gtsrb",
    "imagenet12",
    "omniglot",
    "svhn",
    "ucf101",
    "vgg-flowers",
]


def preprocess_for_mobilenet(
    image: tf.Tensor,
    label: tf.Tensor,
    augment: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocess a single image-label pair for MobileNetV2.

    Resizes the image to ``(MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE)``,
    optionally applies random horizontal flipping, casts to ``float32``, and
    scales pixel values to the ``[-1, 1]`` range expected by MobileNetV2.

    Args:
        image: Raw image tensor with shape ``[H, W, C]`` and dtype ``uint8`` or
            ``float32``.
        label: Corresponding integer class label.
        augment: If ``True``, applies a random horizontal flip.

    Returns:
        A ``(image, label)`` tuple where ``image`` is ``float32`` with shape
        ``[MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE, C]``.
    """
    image = tf.image.resize(image, [MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE])
    if augment:
        image = tf.image.random_flip_left_right(image)
    # preprocess_input expects float32 and maps [0, 255] → [-1, 1].
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def preprocess_domain(
    ds: tf.data.Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer_size: int = 1000,
    augment: bool = False,
    prefetch: int = tf.data.AUTOTUNE,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Apply the MobileNetV2 preprocessing pipeline to a dataset.

    Applies (in order): preprocess → shuffle → batch → prefetch.

    Args:
        ds: ``tf.data.Dataset`` yielding ``(image, label)`` tuples.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle before batching.
        shuffle_buffer_size: Number of elements held in the shuffle buffer.
        augment: If ``True``, applies random horizontal flips.
        prefetch: Number of batches to prefetch.
        seed: Random seed for reproducible shuffling. ``None`` for non-deterministic behaviour.

    Returns:
        A preprocessed, batched, and prefetched ``tf.data.Dataset``.
    """
    ds = ds.map(
        lambda image, label: preprocess_for_mobilenet(image, label, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch)
    return ds


def load_domain(
    domain: str,
    split: str | list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
) -> tf.data.Dataset | dict[str, tf.data.Dataset]:
    """Load a VDD domain from TFDS.

    Args:
        domain: Name of the VDD domain (e.g. ``"aircraft"``, ``"dtd"``). Must be
            in :data:`VDD_DOMAINS`.
        split: A single TFDS split string (e.g. ``"train"``) or a list of split
            strings (e.g. ``["train", "test"]``).
        data_dir: TFDS cache directory. Defaults to ``~/tensorflow_datasets/``.

    Returns:
        A single ``tf.data.Dataset`` when *split* is a string, or a
        ``dict[split_name → tf.data.Dataset]`` when *split* is a list.

    Raises:
        ValueError: If *domain* is not found in :data:`VDD_DOMAINS`.

    Example:
        >>> raw = load_domain("aircraft", "train")
        >>> train_ds = preprocess_domain(raw, batch_size=64, shuffle=True, augment=True)
        >>> splits = load_domain("cifar100", ["train", "test"])
        >>> val_ds = preprocess_domain(splits["test"], batch_size=32)
    """
    if domain not in VDD_DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Valid domains: {sorted(VDD_DOMAINS)}")

    tfds_name = f"visual_domain_decathlon/{domain}"
    if isinstance(split, list):
        return {s: tfds.load(tfds_name, split=s, as_supervised=True, data_dir=data_dir) for s in split}
    return tfds.load(tfds_name, split=split, as_supervised=True, data_dir=data_dir)


def load_domains(
    domains: list[str],
    split: str | list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
) -> dict[str, tf.data.Dataset | dict[str, tf.data.Dataset]]:
    """Load multiple VDD domains from TFDS.

    Convenience wrapper around :func:`load_domain` for multiple domains.

    Args:
        domains: List of domain names (e.g. ``["aircraft", "dtd"]``). Each entry
            must be in :data:`VDD_DOMAINS`.
        split: A single TFDS split string or a list of split strings.
        data_dir: TFDS cache directory.

    Returns:
        ``dict[domain_name → dataset]`` where each value mirrors the return type
        of :func:`load_domain`.

    Raises:
        ValueError: If any entry in *domains* is not found in :data:`VDD_DOMAINS`.

    Example:
        >>> raw = load_domains(["aircraft", "dtd"], ["train", "test"])
        >>> train_ds = preprocess_domain(raw["aircraft"]["train"], batch_size=32, shuffle=True)
    """
    return {domain: load_domain(domain, split, data_dir=data_dir) for domain in domains}
