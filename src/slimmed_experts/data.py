"""Data loading and preprocessing utilities for the Visual Domain Decathlon (VDD) dataset."""

from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


# MobileNetV2 expected input spatial size (height and width).
MOBILENET_INPUT_SIZE: int = 74

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

# ImageNet channel-wise mean and std used by TorchVision MobileNetV2.
_IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


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


def mobilenet_transform(augment: bool = False) -> transforms.Compose:
    """Return the MobileNetV2 preprocessing transform.

    Resizes to ``(MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE)``, optionally
    applies random horizontal flipping, converts to a ``float32`` tensor, and
    normalises with ImageNet channel-wise mean and standard deviation
    (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``).

    Args:
        augment: If ``True``, applies a random horizontal flip.

    Returns:
        A ``torchvision.transforms.Compose`` transform ready to apply to PIL images.
    """
    t: list = [transforms.Resize((MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE))]
    if augment:
        t.append(transforms.RandomHorizontalFlip())
    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    return transforms.Compose(t)


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


def load_domain(
    domain: str,
    split: str | list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
    augment: bool = False,
) -> Dataset | dict[str, Dataset]:
    """Load a VDD domain from local files, downloading first if necessary.

    Args:
        domain: Name of the VDD domain (e.g. ``"aircraft"``). Must be in
            :data:`VDD_DOMAINS`.
        split: A single split string (e.g. ``"train"``) or a list of split
            strings (e.g. ``["train", "val"]``).
        data_dir: Root directory containing the domain folders. Defaults to
            ``"data"``.
        augment: If ``True``, applies random horizontal flips. Pass ``False``
            for validation and test splits.

    Returns:
        A single ``Dataset`` when *split* is a string, or a
        ``dict[split_name → Dataset]`` when *split* is a list.

    Raises:
        ValueError: If *domain* is not found in :data:`VDD_DOMAINS`.
    """
    if domain not in VDD_DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Valid domains: {sorted(VDD_DOMAINS)}")

    base = Path(data_dir) if data_dir is not None else Path("data")
    _ensure_domains(base, [domain])
    transform = mobilenet_transform(augment)
    if isinstance(split, list):
        return {s: _load_split(base / domain / s, transform) for s in split}
    return _load_split(base / domain / split, transform)


def load_domains(
    domains: list[str],
    split: str | list[str],
    *,
    data_dir: str | os.PathLike[str] | None = None,
    augment: bool = False,
) -> dict[str, Dataset | dict[str, Dataset]]:
    """Load multiple VDD domains from local files, downloading first if necessary.

    Convenience wrapper around :func:`load_domain` for multiple domains.

    Args:
        domains: List of domain names (e.g. ``["aircraft", "dtd"]``). Each entry
            must be in :data:`VDD_DOMAINS`.
        split: A single split string or a list of split strings.
        data_dir: Root directory containing the domain folders. Defaults to
            ``"data"``.
        augment: If ``True``, applies random horizontal flips.

    Returns:
        ``dict[domain_name → dataset]`` where each value mirrors the return type
        of :func:`load_domain`.

    Raises:
        ValueError: If any entry in *domains* is not found in :data:`VDD_DOMAINS`.
    """
    return {domain: load_domain(domain, split, data_dir=data_dir, augment=augment) for domain in domains}
