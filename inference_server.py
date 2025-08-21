"""Flask inference server with multi-model support and preprocessing."""
from __future__ import annotations

from collections import Counter
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

import os

from model_registry import MODEL_SPECS, get_model, list_models
from preprocess import prepare_image

app = Flask(__name__)
CORS(app)
metrics = Counter()


def _warmup_models() -> None:
    for spec in MODEL_SPECS.values():
        try:
            model = get_model(spec.key)
            h, w = spec.input_size
            if spec.mode == "L":
                dummy = np.zeros((1, h, w, 1), dtype="float32")
            else:
                dummy = np.zeros((1, 3, h, w), dtype="float32")
            model.predict(dummy)
        except FileNotFoundError as exc:
            print(exc)
        except Exception as exc:  # pragma: no cover - warmup is best effort
            print(f"Warmup failed for {spec.key}: {exc}")


if os.getenv("WARMUP", "true").lower() not in {"0", "false", "no"}:
    _warmup_models()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/health")
def health() -> tuple[str, int]:
    return jsonify({"status": "ok"})


@app.get("/models")
def models_endpoint():
    return jsonify({"models": list_models()})


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    model_key = request.form.get("model_key")
    if model_key not in MODEL_SPECS:
        return jsonify({"error": "invalid model_key"}), 400
    spec = MODEL_SPECS[model_key]
    arr = prepare_image(request.files["file"].stream, spec)
    try:
        model = get_model(model_key)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    label = spec.labels[idx] if spec.labels and idx < len(spec.labels) else str(idx)
    metrics[model_key] += 1
    return jsonify(
        {
            "prediction_index": idx,
            "prediction_label": label,
            "confidence": conf,
            "model_used": model_key,
        }
    )


@app.post("/batch_predict")
def batch_predict():
    files = request.files.getlist("files")
    model_key = request.form.get("model_key")
    if not files:
        return jsonify({"error": "missing files"}), 400
    if model_key not in MODEL_SPECS:
        return jsonify({"error": "invalid model_key"}), 400
    spec = MODEL_SPECS[model_key]
    arrs = [prepare_image(f.stream, spec) for f in files]
    batch = np.concatenate(arrs, axis=0)
    try:
        model = get_model(model_key)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    outputs = model.predict(batch)
    results = []
    for preds in outputs:
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        label = spec.labels[idx] if spec.labels and idx < len(spec.labels) else str(idx)
        results.append({"prediction_index": idx, "prediction_label": label, "confidence": conf})
    metrics[model_key] += len(files)
    return jsonify({"model_used": model_key, "predictions": results})


@app.get("/metrics")
def metrics_endpoint():
    lines = [f"requests_total{{model='{k}'}} {v}" for k, v in metrics.items()]
    return "\n".join(lines), 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)
