"""Smoke tests for downloaded models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

try:  # optional dependency
    import onnxruntime as ort
except Exception:  # noqa: BLE001
    ort = None

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def test_keras(model_file: str, label: str) -> None:
    path = MODEL_DIR / model_file
    if not path.exists():
        print(f"❌ {model_file} missing; run scripts/download_models.py")
        return
    try:
        model = tf.keras.models.load_model(path)
        dummy = np.random.rand(1, 28, 28, 1).astype("float32")
        preds = model.predict(dummy)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        print(f"✅ {label}: class {idx} (confidence {conf:.4f})")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {label} failed: {exc}")


def test_torch(model_file: str, label: str) -> None:
    path = MODEL_DIR / model_file
    if not path.exists():
        print(f"❌ {model_file} missing; run scripts/download_models.py")
        return
    try:
        bundle = torch.load(path, map_location="cpu")
        model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
            probs = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(probs))
        conf = float(probs[idx])
        print(f"✅ {label}: class {idx} (confidence {conf:.4f})")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {label} failed: {exc}")


def test_onnx(model_file: str, label: str) -> None:
    if ort is None:
        print("⚠️ onnxruntime not installed; skipping ONNX tests")
        return
    path = MODEL_DIR / model_file
    if not path.exists():
        print(f"❌ {model_file} missing; run scripts/download_models.py")
        return
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, 3, 224, 224).astype("float32")
        inputs = {session.get_inputs()[0].name: dummy}
        outputs = session.run(None, inputs)[0]
        idx = int(np.argmax(outputs[0]))
        conf = float(np.max(outputs[0]))
        print(f"✅ {label}: class {idx} (confidence {conf:.4f})")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {label} failed: {exc}")


def main() -> None:
    if not MODEL_DIR.exists():
        print("❌ models directory missing; run scripts/download_models.py")
        return
    test_keras("mnist_digits.h5", "MNIST digits")
    test_keras("fashion_mnist.h5", "Fashion-MNIST")
    test_torch("resnet18.pt", "ResNet18 ImageNet")
    test_onnx("mobilenet_v3_small.onnx", "MobileNetV3 Small")


if __name__ == "__main__":
    main()
