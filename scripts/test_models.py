"""Basic smoke tests for downloaded models.

Each test loads one model of a different framework and performs a tiny
inference to verify that the model files are usable.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import tensorflow as tf

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def test_fashion_mnist() -> None:
    model_path = MODEL_DIR / "fashion_mnist.h5"
    if not model_path.exists():
        print("❌ fashion_mnist.h5 missing; run download_models.py")
        return
    model = tf.keras.models.load_model(model_path)
    dummy = np.random.rand(1, 28, 28, 1).astype("float32")
    preds = model.predict(dummy)
    print(f"✅ Fashion MNIST output shape: {preds.shape}")


def test_mnist_digits() -> None:
    model_path = MODEL_DIR / "mnist_digits.h5"
    if not model_path.exists():
        print("❌ mnist_digits.h5 missing; run download_models.py")
        return
    model = tf.keras.models.load_model(model_path)
    dummy = np.random.rand(1, 28, 28, 1).astype("float32")
    preds = model.predict(dummy)
    print(f"✅ MNIST digits output shape: {preds.shape}")


def test_gpt2() -> None:
    weight_path = MODEL_DIR / "gpt2_pytorch.bin"
    config_path = MODEL_DIR / "gpt2_config.json"
    if not weight_path.exists():
        print("❌ gpt2_pytorch.bin missing; run download_models.py")
        return
    config = GPT2Config.from_json_file(str(config_path)) if config_path.exists() else GPT2Config()
    model = GPT2LMHeadModel(config)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
    input_ids = tokenizer.encode("Hello", return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=20)
    print("✅ GPT-2 sample:", tokenizer.decode(output_ids[0]))


def test_distilbert() -> None:
    weight_path = MODEL_DIR / "distilbert_sentiment.bin"
    config_path = MODEL_DIR / "distilbert_config.json"
    if not weight_path.exists():
        print("❌ distilbert_sentiment.bin missing; run download_models.py")
        return
    config = AutoConfig.from_json_file(str(config_path)) if config_path.exists() else AutoConfig.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_config(config)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english")
    inputs = tokenizer("I love this library!", return_tensors="pt")
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    print(f"✅ DistilBERT sentiment label: {pred}")


def main() -> None:
    test_fashion_mnist()
    test_mnist_digits()
    test_gpt2()
    test_distilbert()


if __name__ == "__main__":
    main()
