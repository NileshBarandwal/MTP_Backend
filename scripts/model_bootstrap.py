from __future__ import annotations

"""Path helpers for model bootstrap scripts."""

from pathlib import Path


def project_root() -> Path:
    """Return absolute path to repository root."""
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    """Return absolute path to models directory, creating it if missing."""
    path = project_root() / "models"
    path.mkdir(exist_ok=True)
    return path


def registry_path() -> Path:
    """Return path to model registry JSON file."""
    return models_dir() / "registry.json"


def resolve_model_path(path_from_registry: str) -> Path:
    """Resolve a model path from registry to an absolute path."""
    return models_dir() / path_from_registry
