from __future__ import annotations

"""Fetch or build models required by the server."""

from pathlib import Path
import hashlib
import json
from typing import Dict

import numpy as np
import torch
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow missing
    tf = None  # type: ignore

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REGISTRY_PATH = MODEL_DIR / "registry.json"
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# utility helpers
# ---------------------------------------------------------------------------

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:  # pragma: no cover - trivial
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Keras fallbacks
# ---------------------------------------------------------------------------


def _train_mnist(model_name: str, dataset: str, out_path: Path) -> None:
    if tf is None:  # pragma: no cover - defensive
        raise RuntimeError("TensorFlow not available for training")
    tf.random.set_seed(0)
    np.random.seed(0)
    try:
        if dataset == "mnist":
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        else:
            (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_train = x_train[..., None]
    except Exception:
        x_train = np.random.rand(256, 28, 28, 1).astype("float32")
        y_train = np.random.randint(0, 10, size=(256,))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, 3, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0, shuffle=False)
    model.save(out_path)


def ensure_keras(id_: str, repo: str, filename: str) -> str:
    target = MODEL_DIR / f"{id_}.h5"
    if target.exists():
        return "existing"
    try:
        from huggingface_hub import hf_hub_download

        src = hf_hub_download(repo_id=repo, filename=filename)
        Path(src).replace(target)
        return "downloaded"
    except Exception:
        _train_mnist(id_, "mnist" if "mnist" in id_ else "fashion_mnist", target)
        return "trained"


# ---------------------------------------------------------------------------
# Torch / ONNX helpers
# ---------------------------------------------------------------------------

def ensure_resnet18() -> str:
    path = MODEL_DIR / "resnet18.pt"
    if path.exists():
        return "existing"
    try:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        meta = weights.meta
        status = "downloaded"
    except Exception:
        torch.manual_seed(0)
        model = resnet18(weights=None)
        meta = {"categories": [f"class_{i}" for i in range(1000)]}
        status = "random"
    model.eval()
    torch.save({"model": model, "meta": meta}, path)
    labels = MODEL_DIR / "imagenet_labels.txt"
    if not labels.exists():
        with labels.open("w") as f:
            for c in meta["categories"]:
                f.write(c + "\n")
    return status


def ensure_mobilenet_onnx() -> str:
    path = MODEL_DIR / "mobilenet_v3_small.onnx"
    if path.exists():
        return "existing"
    try:
        import onnx  # noqa: F401
    except Exception:
        if REGISTRY_PATH.exists():
            data = json.loads(REGISTRY_PATH.read_text())
            data["models"] = [m for m in data.get("models", []) if m.get("id") != "mobilenet_v3_small"]
            REGISTRY_PATH.write_text(json.dumps(data, indent=2) + "\n")
        return "skipped"
    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        meta = weights.meta
        status = "downloaded"
    except Exception:
        model = mobilenet_v3_small(weights=None)
        meta = {"categories": [f"class_{i}" for i in range(1000)]}
        status = "random"
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=11,
    )
    labels = MODEL_DIR / "imagenet_labels.txt"
    if not labels.exists():
        with labels.open("w") as f:
            for c in meta["categories"]:
                f.write(c + "\n")
    return status


# ---------------------------------------------------------------------------


def update_registry() -> None:
    if not REGISTRY_PATH.exists():
        return
    data = json.loads(REGISTRY_PATH.read_text())
    modified = False
    for entry in data.get("models", []):
        path = MODEL_DIR / entry["path"]
        if not path.exists():
            continue
        checksum = sha256sum(path)
        if entry.get("checksum") != checksum:
            entry["checksum"] = checksum
            modified = True
    if modified:
        REGISTRY_PATH.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    summary: Dict[str, str] = {}
    summary["mnist_digits"] = ensure_keras(
        "mnist_digits", "paulpall/Beyond_MNIST", "Best_Model.h5"
    )
    summary["fashion_mnist"] = ensure_keras(
        "fashion_mnist", "Eehjie/fashion-mnist-tf-keras-model", "fashion_mnist_model.h5"
    )
    summary["resnet18_imagenet"] = ensure_resnet18()
    summary["mobilenet_v3_small"] = ensure_mobilenet_onnx()
    update_registry()
    print("\nModel fetch summary:")
    for k, v in summary.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
