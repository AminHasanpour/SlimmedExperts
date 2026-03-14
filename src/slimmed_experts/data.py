"""Data loading and preprocessing utilities for the Visual Domain Decathlon (VDD) dataset."""

from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from collections.abc import Sized
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


# Default square input size used by image transforms.
DEFAULT_INPUT_SIZE: int = 74

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

# URL to download the full VDD dataset archive.
_VDD_DOWNLOAD_URL: str = "http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-data.tar.gz"

# Maximum number of training samples used to estimate normalization stats.
# Set to None to use the full training split.
_NORMALIZATION_MAX_SAMPLES: int | None = 4096

# Fixed seed for deterministic random sampling when limiting normalization data.
_NORMALIZATION_SAMPLING_SEED: int = 42


def _ensure_domains(data_dir: Path, domains: list[str]) -> None:
    """Download and extract VDD datasets if any of the requested domains are missing.

    Downloads the full VDD archive to a temporary directory, extracts it there
    (producing ``<domain>.tar`` files), then extracts each domain tar to
    *data_dir*. The temporary directory is removed afterwards.

    Args:
        data_dir: Root directory where domain folders are expected.
        domains: List of domain names to ensure are present.
    """
    missing = [d for d in domains if not (data_dir / d).is_dir()]
    if not missing:
        return

    logger.info(f"Domains {missing} not found in {data_dir}. Downloading VDD dataset...")
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        archive = tmp / "decathlon-1.0-data.tar.gz"

        logger.info(f"Downloading {_VDD_DOWNLOAD_URL} ...")
        urllib.request.urlretrieve(_VDD_DOWNLOAD_URL, archive)

        logger.info("Extracting main archive...")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp)

        logger.info(f"Extracting domain archives to {data_dir} ...")
        for domain_tar in sorted(tmp.glob("*.tar")):
            with tarfile.open(domain_tar) as tar:
                tar.extractall(data_dir)

    logger.info("Download and extraction complete.")


def _build_transform(
    input_size: int,
    *,
    augment: bool = False,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """Build an image transform for a target input size.

    Args:
        input_size: Target square image size after preprocessing.
        augment: If ``True``, applies train-time augmentation.
        mean: Per-channel mean for normalisation. If ``None``, no normalisation.
        std: Per-channel standard deviation for normalisation. If ``None``, no normalisation.

    Returns:
        A ``torchvision.transforms.Compose`` transform.
    """
    larger = int(input_size * 1.15)
    ops: list[transforms.Transform] = []

    if augment:
        ops.extend(
            [
                transforms.Resize((larger, larger)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(15),
            ]
        )
    else:
        ops.append(transforms.Resize((input_size, input_size)))

    ops.append(transforms.ToTensor())
    if mean is not None and std is not None:
        ops.append(transforms.Normalize(mean=mean, std=std))
    if augment:
        ops.append(transforms.RandomErasing(p=0.25))

    return transforms.Compose(ops)


class _FlatImageDataset(Dataset):
    """Dataset for a flat directory of JPEG images without class labels.

    All images are assigned label ``-1``.
    """

    def __init__(self, paths: list[str], transform: transforms.Compose) -> None:
        self._paths = paths
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self._paths[idx]).convert("RGB")
        return self._transform(image), -1


def _load_split(split_dir: Path, transform: transforms.Compose) -> Dataset:
    """Load a split directory as a PyTorch Dataset.

    Uses ``ImageFolder`` for labeled splits (train/val) that contain class
    subdirectories, or :class:`_FlatImageDataset` (label ``-1``) for unlabeled
    splits (test) with a flat layout.

    Args:
        split_dir: Path to the split directory.
        transform: Transform to apply to each image.

    Returns:
        A PyTorch ``Dataset`` of ``(image_tensor, label)`` pairs.
    """
    has_labels = any(e.is_dir() for e in split_dir.iterdir())
    if has_labels:
        return ImageFolder(str(split_dir), transform=transform)
    image_paths = sorted(str(p) for p in split_dir.glob("*.jpg"))
    return _FlatImageDataset(image_paths, transform=transform)


def _compute_normalization_stats(data_dir: Path, domain: str, input_size: int) -> tuple[list[float], list[float]]:
    """Compute per-channel mean and std from a domain's training split.

    Statistics are computed after resizing and converting to tensors in [0, 1].

    Args:
        data_dir: Root data directory.
        domain: Domain name.
        input_size: Target square image size.

    Returns:
        Tuple ``(mean, std)`` where each is a list of 3 floats.

    Raises:
        ValueError: If the training split has no samples.
    """
    train_split_dir = data_dir / domain / "train"
    dataset = _load_split(train_split_dir, _build_transform(input_size, augment=False))

    assert isinstance(dataset, Sized), "Dataset must have a finite length to compute normalization stats."
    if _NORMALIZATION_MAX_SAMPLES is not None and len(dataset) > _NORMALIZATION_MAX_SAMPLES:
        generator = torch.Generator().manual_seed(_NORMALIZATION_SAMPLING_SEED)
        indices = torch.randperm(len(dataset), generator=generator)[:_NORMALIZATION_MAX_SAMPLES].tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images * images).sum(dim=(0, 2, 3))
        total_pixels += images.size(0) * images.size(2) * images.size(3)

    if total_pixels == 0:
        raise ValueError(f"Training split for domain '{domain}' has no images.")

    mean = channel_sum / total_pixels
    var = (channel_sq_sum / total_pixels) - (mean * mean)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.tolist(), std.tolist()


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    """Wrap a dataset in a :class:`~torch.utils.data.DataLoader`.

    Args:
        dataset: PyTorch ``Dataset`` to load from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        num_workers: Number of subprocess workers for data loading.
        seed: Random seed for reproducible shuffling. ``None`` for
            non-deterministic behaviour.

    Returns:
        A configured ``DataLoader`` instance.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator)


def _load_domain(
    domain: str,
    split: list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
    augment: bool = False,
    input_size: int = DEFAULT_INPUT_SIZE,
    normalize: bool = False,
) -> dict[str, Dataset]:
    """Load selected splits for one VDD domain.

    Args:
        domain: Name of the VDD domain (e.g. ``"aircraft"``). Must be in
            :data:`VDD_DOMAINS`.
        split: Split names to load (e.g. ``["train", "val"]``).
        data_dir: Root directory containing the domain folders. Defaults to
            ``"data"``.
        augment: If ``True``, applies train-time augmentation to the ``"train"`` split.
        input_size: Target square image size after preprocessing.
        normalize: If ``True``, compute per-domain normalisation statistics from
            the training split and apply them to all requested splits.

    Returns:
        ``dict[split_name, Dataset]``.

    Raises:
        ValueError: If *domain* is not found in :data:`VDD_DOMAINS`.
    """
    if domain not in VDD_DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Valid domains: {sorted(VDD_DOMAINS)}")
    if not isinstance(split, list):
        raise TypeError("split must be a list of split names.")
    if not split:
        raise ValueError("split must contain at least one split name.")
    if any(not isinstance(s, str) for s in split):
        raise TypeError("split must contain only strings.")

    base = Path(data_dir) if data_dir is not None else Path("data")
    _ensure_domains(base, [domain])

    mean: list[float] | None = None
    std: list[float] | None = None
    if normalize:
        mean, std = _compute_normalization_stats(base, domain, input_size)
        logger.debug(f"Computed normalization stats for domain '{domain}': mean={mean}, std={std}")

    datasets: dict[str, Dataset] = {}
    for split_name in split:
        split_augment = augment and split_name == "train"
        transform = _build_transform(input_size, augment=split_augment, mean=mean, std=std)
        datasets[split_name] = _load_split(base / domain / split_name, transform)
    return datasets


def load_domains(
    domains: list[str],
    split: list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
    augment: bool = False,
    input_size: int = DEFAULT_INPUT_SIZE,
    normalize: bool = False,
) -> dict[str, dict[str, Dataset]]:
    """Load multiple VDD domains from local files, downloading first if necessary.

    Convenience wrapper around :func:`_load_domain` for multiple domains.

    Args:
        domains: List of domain names (e.g. ``["aircraft", "dtd"]``). Each entry
            must be in :data:`VDD_DOMAINS`.
        split: Split names to load (e.g. ``["train", "val"]``).
        data_dir: Root directory containing the domain folders. Defaults to
            ``"data"``.
        augment: If ``True``, applies train-time augmentation to the ``"train"`` split.
        input_size: Target square image size after preprocessing.
        normalize: If ``True``, normalise using per-domain statistics computed
            from each domain's training split.

    Returns:
        ``dict[domain_name, dict[split_name, Dataset]]``.

    Raises:
        ValueError: If any entry in *domains* is not found in :data:`VDD_DOMAINS`.
    """
    if not isinstance(split, list):
        raise TypeError("split must be a list of split names.")

    return {
        domain: _load_domain(
            domain,
            split,
            data_dir=data_dir,
            augment=augment,
            input_size=input_size,
            normalize=normalize,
        )
        for domain in domains
    }
