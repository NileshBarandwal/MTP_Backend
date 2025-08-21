"""Flask inference server with simple web UI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(__file__).resolve().parent / "models"
FASHION_LABELS = [
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
]

_keras_cache: dict[str, tf.keras.Model] = {}
_resnet_cache: dict[str, object] = {}


def load_keras_model(key: str, filename: str) -> tf.keras.Model:
    if key not in _keras_cache:
        model_path = MODEL_DIR / filename
        _keras_cache[key] = tf.keras.models.load_model(model_path)
    return _keras_cache[key]


def load_resnet18() -> tuple[torch.nn.Module, ResNet18_Weights]:
    if not _resnet_cache:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.eval()
        _resnet_cache["model"] = model
        _resnet_cache["weights"] = weights
    return _resnet_cache["model"], _resnet_cache["weights"]


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/infer")
def infer():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    model_key = request.form.get("model_key")
    if model_key not in {"mnist_digits", "fashion_mnist", "resnet18_imagenet"}:
        return jsonify({"error": "invalid model_key"}), 400

    file = request.files["file"]
    try:
        if model_key in {"mnist_digits", "fashion_mnist"}:
            img = Image.open(file.stream).convert("L").resize((28, 28))
            arr = np.array(img, dtype="float32") / 255.0
            arr = arr.reshape(1, 28, 28, 1)
            model = load_keras_model(
                model_key,
                "mnist_digits.h5" if model_key == "mnist_digits" else "fashion_mnist.h5",
            )
            preds = model.predict(arr)
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            label = str(idx) if model_key == "mnist_digits" else FASHION_LABELS[idx]
        else:
            model, weights = load_resnet18()
            img = Image.open(file.stream).convert("RGB")
            preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
                ]
            )
            tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
            idx = int(torch.argmax(probs))
            conf = float(probs[idx])
            label = weights.meta["categories"][idx]
        return jsonify(
            {
                "prediction_index": idx,
                "prediction_label": label,
                "confidence": conf,
                "model_used": model_key,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
