"""Download ML models for inference server.

This script fetches a variety of vision and NLP models from the Hugging Face
Hub and stores them under the local ``models/`` directory. Downloads are
idempotent; existing files or directories are skipped.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def download_from_hf(repo_id: str, filename: str, target_path: Path) -> bool:
    """Download a single file from Hugging Face Hub."""
    if target_path.exists():
        print(f"⚠ skipped {target_path.name} (already exists)")
        return False
    try:
        src_path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy(src_path, target_path)
        print(f"✅ downloaded {target_path.name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"❌ failed to download {filename} from {repo_id}: {exc}")
        return False


def snapshot_model(repo_id: str, local_dir: Path) -> bool:
    """Download an entire model repository as a snapshot."""
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"⚠ skipped snapshot {repo_id} (directory exists)")
        return False
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"✅ downloaded snapshot {repo_id}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"❌ failed to snapshot {repo_id}: {exc}")
        return False


def main() -> None:
    print("\n⬇️ Starting model downloads...\n")
    download_from_hf(
        repo_id="Eehjie/fashion-mnist-tf-keras-model",
        filename="fashion_mnist_model.h5",
        target_path=MODEL_DIR / "fashion_mnist.h5",
    )
    download_from_hf(
        repo_id="paulpall/Beyond_MNIST",
        filename="Best_Model.h5",
        target_path=MODEL_DIR / "mnist_digits.h5",
    )
    snapshot_model("openai-community/gpt2", MODEL_DIR / "gpt2")
    snapshot_model("distilbert-base-uncased-finetuned-sst-2-english", MODEL_DIR / "distilbert-sst2")
    print("\n🎉 Downloads complete. Models are stored in ./models/\n")


if __name__ == "__main__":
    main()
