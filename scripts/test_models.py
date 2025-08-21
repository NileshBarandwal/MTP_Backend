"""Smoke tests for downloaded models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

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


def test_gpt2() -> None:
    model_dir = MODEL_DIR / "gpt2"
    if not model_dir.exists():
        print("❌ gpt2 missing; run scripts/download_models.py")
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        prompt = "Patient reports mild chest pain. Recommended next step:"
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(inputs, max_new_tokens=30)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"✅ GPT-2 output: {text}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ GPT-2 failed: {exc}")


def test_distilbert() -> None:
    model_dir = MODEL_DIR / "distilbert-sst2"
    if not model_dir.exists():
        print("❌ distilbert-sst2 missing; run scripts/download_models.py")
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        text = "The medication worked surprisingly well and I feel better."
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        score = float(probs[idx])
        label = model.config.id2label.get(idx, str(idx))
        print(f"✅ DistilBERT sentiment: {label} (score {score:.4f})")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ DistilBERT failed: {exc}")


def main() -> None:
    if not MODEL_DIR.exists():
        print("❌ models directory missing; run scripts/download_models.py")
        return
    test_keras("mnist_digits.h5", "MNIST digits")
    test_keras("fashion_mnist.h5", "Fashion-MNIST")
    test_gpt2()
    test_distilbert()


if __name__ == "__main__":
    main()
