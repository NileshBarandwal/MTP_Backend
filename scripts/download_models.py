"""Download various ML models into the models/ directory.

Uses different sources (Hugging Face Hub, Keras utility, and plain HTTP
links) to fetch a diverse set of models. Downloads are skipped if the
file already exists.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from tensorflow.keras.utils import get_file

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    """Download a file from a direct HTTP link with a progress bar."""
    if dest.exists():
        print(f"✅ {dest.name} already exists, skipping.")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as file, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as progress:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress.update(len(chunk))
    print(f"✅ Downloaded {dest.name}")


def download_with_keras(url: str, dest: Path) -> None:
    """Download using Keras's get_file utility."""
    if dest.exists():
        print(f"✅ {dest.name} already exists, skipping.")
        return
    path = get_file(fname=dest.name, origin=url, cache_dir=str(MODEL_DIR), cache_subdir=".")
    shutil.move(path, dest)
    print(f"✅ Downloaded {dest.name}")


def download_from_hf(
    repo_id: str,
    filename: str,
    dest: Path,
    config_filename: Optional[str] = None,
    config_dest: Optional[Path] = None,
) -> None:
    """Download a file from Hugging Face Hub (optionally config)."""
    if dest.exists():
        print(f"✅ {dest.name} already exists, skipping.")
        return
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy(path, dest)
    if config_filename and config_dest:
        cfg_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
        shutil.copy(cfg_path, config_dest)
    print(f"✅ Downloaded {dest.name}")


def main() -> None:
    print("\n⬇️ Starting model downloads...\n")
    # Fashion-MNIST classifier via Keras utility
    download_with_keras(
        url="https://github.com/nnUyi/FashionMNIST-Keras/raw/master/fashion_mnist.h5",
        dest=MODEL_DIR / "fashion_mnist.h5",
    )

    # MNIST digit CNN via plain HTTP
    download_file(
        url="https://github.com/llSourcell/mnist_digit_recognition/blob/master/DeepLearning/mnist_cnn_model.h5?raw=1",
        dest=MODEL_DIR / "mnist_digits.h5",
    )

    # GPT-2 tiny model via Hugging Face Hub
    download_from_hf(
        repo_id="sshleifer/tiny-gpt2",
        filename="pytorch_model.bin",
        dest=MODEL_DIR / "gpt2_pytorch.bin",
        config_filename="config.json",
        config_dest=MODEL_DIR / "gpt2_config.json",
    )

    # DistilBERT sentiment model via Hugging Face Hub
    download_from_hf(
        repo_id="sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english",
        filename="pytorch_model.bin",
        dest=MODEL_DIR / "distilbert_sentiment.bin",
        config_filename="config.json",
        config_dest=MODEL_DIR / "distilbert_config.json",
    )

    print("\n🎉 Downloads complete. Models are stored in the models/ directory.")


if __name__ == "__main__":
    main()
