"""Data loading and preprocessing utilities for the Visual Domain Decathlon (VDD) dataset."""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf


# MobileNetV2 expected input spatial size (height and width).
MOBILENET_INPUT_SIZE: int = 224

# List of valid VDD domains.
VDD_DOMAINS: list[str] = [
    "aircraft",
    "cifar100",
    "daimlerpedcls",
    "dtd",
    "gtsrb",
    "omniglot",
    "svhn",
    "ucf101",
    "vgg-flowers",
]

# ImageNet channel-wise mean and std used by TorchVision MobileNetV2.
_IMAGENET_MEAN: tf.Tensor = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_IMAGENET_STD: tf.Tensor = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def preprocess_for_mobilenet(
    image: tf.Tensor,
    label: tf.Tensor,
    augment: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocess a single image-label pair for MobileNetV2.

    Resizes the image to ``(MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE)``,
    optionally applies random horizontal flipping, casts to ``float32``, scales
    pixel values to ``[0, 1]``, and normalises with ImageNet channel-wise mean
    and standard deviation (``mean=[0.485, 0.456, 0.406]``,
    ``std=[0.229, 0.224, 0.225]``).

    Args:
        image: Raw image tensor with shape ``[H, W, C]`` and dtype ``uint8`` or
            ``float32``.
        label: Corresponding integer class label.
        augment: If ``True``, applies a random horizontal flip.

    Returns:
        A ``(image, label)`` tuple where ``image`` is ``float32`` with shape
        ``[MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE, C]``, normalised with
        ImageNet statistics.
    """
    image = tf.image.resize(image, [MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE])
    if augment:
        image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
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


def _load_split_from_dir(split_dir: str) -> tf.data.Dataset:
    """Load images from a local split directory into a tf.data.Dataset.

    For labeled splits (train/val): expects ``<split_dir>/<class_id>/<image>.jpg``.
    For unlabeled splits (test): expects a flat ``<split_dir>/<image>.jpg``.
    Labels are zero-indexed integers (sorted class-folder order) or ``-1`` for test.
    """
    entries = os.listdir(split_dir)
    has_labels = any(os.path.isdir(os.path.join(split_dir, e)) for e in entries)

    if has_labels:
        class_dirs = sorted(e for e in entries if os.path.isdir(os.path.join(split_dir, e)))
        paths: list[str] = []
        labels: list[int] = []
        for label_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(split_dir, class_dir)
            for fname in sorted(os.listdir(class_path)):
                if fname.lower().endswith(".jpg"):
                    paths.append(os.path.join(class_path, fname))
                    labels.append(label_idx)
        path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        return path_ds.map(
            lambda p, label: (tf.image.decode_jpeg(tf.io.read_file(p), channels=3), tf.cast(label, tf.int64)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        paths = sorted(os.path.join(split_dir, f) for f in entries if f.lower().endswith(".jpg"))
        path_ds = tf.data.Dataset.from_tensor_slices(paths)
        return path_ds.map(
            lambda p: (tf.image.decode_jpeg(tf.io.read_file(p), channels=3), tf.constant(-1, dtype=tf.int64)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )


def load_domain(
    domain: str,
    split: str | list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
) -> tf.data.Dataset | dict[str, tf.data.Dataset]:
    """Load a VDD domain from local files.

    Args:
        domain: Name of the VDD domain (e.g. ``"aircraft"``, ``"dtd"``). Must be
            in :data:`VDD_DOMAINS`.
        split: A single split string (e.g. ``"train"``) or a list of split
            strings (e.g. ``["train", "test"]``).
        data_dir: Root directory containing the domain folders. Defaults to
            ``"data"``.

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

    base = Path(data_dir) if data_dir is not None else Path("data")
    if isinstance(split, list):
        return {s: _load_split_from_dir(str(base / domain / s)) for s in split}
    return _load_split_from_dir(str(base / domain / split))


def load_domains(
    domains: list[str],
    split: str | list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
) -> dict[str, tf.data.Dataset | dict[str, tf.data.Dataset]]:
    """Load multiple VDD domains from local files.

    Convenience wrapper around :func:`load_domain` for multiple domains.

    Args:
        domains: List of domain names (e.g. ``["aircraft", "dtd"]``). Each entry
            must be in :data:`VDD_DOMAINS`.
        split: A single split string or a list of split strings.
        data_dir: Root directory containing the domain folders.

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
