from __future__ import annotations

"""Model registry backed by models/registry.json."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json

from model_loader import BaseModel, KerasModel, ModelSpec, ONNXModel, TorchModel

MODEL_DIR = Path(__file__).resolve().parent / "models"
REGISTRY_FILE = MODEL_DIR / "registry.json"


def _load_labels(item_labels) -> List[str]:
    if isinstance(item_labels, list):
        return item_labels
    path = MODEL_DIR / str(item_labels)
    if path.exists():
        return path.read_text().splitlines()
    return []


def _parse_registry() -> Dict[str, ModelSpec]:
    if not REGISTRY_FILE.exists():
        return {}
    data = json.loads(REGISTRY_FILE.read_text())
    specs: Dict[str, ModelSpec] = {}
    for item in data.get("models", []):
        labels = _load_labels(item.get("labels", []))
        framework = item["framework"]
        fmt = "torch" if framework == "pytorch" else framework
        size = item.get("input_spec", {}).get("size", [0, 0])
        spec = ModelSpec(
            key=item["id"],
            path=MODEL_DIR / item["path"],
            format=fmt,
            description=item.get("task", item["id"]),
            input_size=tuple(size),
            mode=item.get("input_spec", {}).get("mode", "RGB"),
            mean=item.get("preprocess_profile", {}).get("mean", [0.0, 0.0, 0.0]),
            std=item.get("preprocess_profile", {}).get("std", [1.0, 1.0, 1.0]),
            labels=labels,
        )
        specs[item["id"]] = spec
    return specs


MODEL_SPECS: Dict[str, ModelSpec] = _parse_registry()

_model_cache: Dict[str, BaseModel] = {}


def list_models() -> List[dict]:
    """Return lightweight info about available models."""
    return [
        {"key": spec.key, "description": spec.description, "format": spec.format}
        for spec in MODEL_SPECS.values()
    ]


def get_model(key: str) -> BaseModel:
    """Return a loaded model, loading it on first use."""
    if key not in MODEL_SPECS:
        raise KeyError(key)
    if key not in _model_cache:
        spec = MODEL_SPECS[key]
        try:
            if spec.format == "keras":
                _model_cache[key] = KerasModel(spec)
            elif spec.format == "torch":
                _model_cache[key] = TorchModel(spec)
            elif spec.format == "onnx":
                _model_cache[key] = ONNXModel(spec)
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported format: {spec.format}")
        except FileNotFoundError as exc:  # pragma: no cover - runtime message
            raise FileNotFoundError(
                f"Model file '{spec.path}' not found. Run 'python scripts/fetch_model_zoo.py'."
            ) from exc
    return _model_cache[key]
