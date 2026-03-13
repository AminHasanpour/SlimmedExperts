"""Factory utilities for class-path based model plugin construction."""

from __future__ import annotations

import importlib
from typing import Any

from slimmed_experts.models.backbones.base import Backbone
from slimmed_experts.models.heads.base import MultiHead
from slimmed_experts.models.model import MultiHeadModel


def load_class(class_path: str) -> type[Any]:
    """Load a class from a fully-qualified class path.

    Args:
        class_path: Dotted path like ``package.module.ClassName``.

    Returns:
        Loaded class object.

    Raises:
        ValueError: If class path format is invalid.
        ImportError: If module or symbol cannot be imported.
    """
    module_path, sep, class_name = class_path.rpartition(".")
    if not sep or not module_path or not class_name:
        raise ValueError(f"Invalid class_path '{class_path}'. Expected format 'package.module.ClassName'.")

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise ImportError(f"Could not import module '{module_path}' from class_path '{class_path}'.") from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Module '{module_path}' has no attribute '{class_name}'.") from exc

    if not isinstance(cls, type):
        raise TypeError(f"Symbol '{class_path}' is not a class.")
    return cls


def _validate_subclass(loaded_class: type[Any], expected_base: type[Any], class_path: str) -> None:
    if not issubclass(loaded_class, expected_base):
        raise TypeError(f"Class '{class_path}' must inherit from {expected_base.__module__}.{expected_base.__name__}.")


def build_model(
    *,
    domains: list[str],
    num_classes: dict[str, int],
    backbone_class_path: str,
    backbone_args: dict[str, Any] | None,
    head_class_path: str,
    head_args: dict[str, Any] | None,
) -> MultiHeadModel:
    """Build a composable multi-head model from plugin configs.

    Args:
        domains: Ordered list of domain names.
        num_classes: Mapping from domain to class count.
        backbone_class_path: Fully-qualified backbone class path.
        backbone_args: Backbone constructor keyword arguments.
        head_class_path: Fully-qualified multi-head class path.
        head_args: Head constructor keyword arguments.

    Returns:
        Instantiated :class:`MultiHeadModel`.
    """
    backbone_kwargs = dict(backbone_args or {})
    head_kwargs = dict(head_args or {})

    backbone_class = load_class(backbone_class_path)
    _validate_subclass(backbone_class, Backbone, backbone_class_path)
    backbone = backbone_class(**backbone_kwargs)

    reserved_keys = {"domains", "num_classes", "in_features"}
    conflicting_keys = reserved_keys.intersection(head_kwargs.keys())
    if conflicting_keys:
        raise ValueError(f"head.args cannot define reserved keys used by the pipeline: {sorted(conflicting_keys)}")

    head_class = load_class(head_class_path)
    _validate_subclass(head_class, MultiHead, head_class_path)
    head = head_class(
        domains=domains,
        num_classes=num_classes,
        in_features=backbone.output_dim,
        **head_kwargs,
    )

    return MultiHeadModel(backbone=backbone, head=head)
