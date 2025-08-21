from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from model_loader import BaseModel, KerasModel, ModelSpec, ONNXModel, TorchModel

MODEL_DIR = Path(__file__).resolve().parent / "models"

# ImageNet labels used for both PyTorch and ONNX models
_IMAGENET_LABELS: List[str]
labels_file = MODEL_DIR / "imagenet_labels.txt"
if labels_file.exists():
    _IMAGENET_LABELS = labels_file.read_text().splitlines()
else:  # Fallback to empty list
    _IMAGENET_LABELS = []

MODEL_SPECS: Dict[str, ModelSpec] = {
    "mnist_digits": ModelSpec(
        key="mnist_digits",
        path=MODEL_DIR / "mnist_digits.h5",
        format="keras",
        description="MNIST digit classifier (Keras)",
        input_size=(28, 28),
        mode="L",
        mean=[0.0],
        std=[1.0],
        labels=[str(i) for i in range(10)],
    ),
    "fashion_mnist": ModelSpec(
        key="fashion_mnist",
        path=MODEL_DIR / "fashion_mnist.h5",
        format="keras",
        description="Fashion-MNIST classifier (Keras)",
        input_size=(28, 28),
        mode="L",
        mean=[0.0],
        std=[1.0],
        labels=[
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    ),
    "resnet18_imagenet": ModelSpec(
        key="resnet18_imagenet",
        path=MODEL_DIR / "resnet18.pt",
        format="torch",
        description="ResNet18 ImageNet (PyTorch)",
        input_size=(224, 224),
        mode="RGB",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        labels=_IMAGENET_LABELS,
    ),
    "mobilenet_v3_small": ModelSpec(
        key="mobilenet_v3_small",
        path=MODEL_DIR / "mobilenet_v3_small.onnx",
        format="onnx",
        description="MobileNetV3 Small (ONNX)",
        input_size=(224, 224),
        mode="RGB",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        labels=_IMAGENET_LABELS,
    ),
}

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
        if spec.format == "keras":
            _model_cache[key] = KerasModel(spec)
        elif spec.format == "torch":
            _model_cache[key] = TorchModel(spec)
        elif spec.format == "onnx":
            _model_cache[key] = ONNXModel(spec)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported format: {spec.format}")
    return _model_cache[key]
