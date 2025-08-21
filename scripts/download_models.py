"""Download or generate models for the inference server."""
from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def download_keras_models() -> None:
    """Fetch small Keras models for MNIST and Fashion-MNIST."""
    keras_models = {
        "mnist_digits.h5": ("paulpall/Beyond_MNIST", "Best_Model.h5"),
        "fashion_mnist.h5": ("Eehjie/fashion-mnist-tf-keras-model", "fashion_mnist_model.h5"),
    }
    for target, (repo, filename) in keras_models.items():
        path = MODEL_DIR / target
        if path.exists():
            print(f"⚠ skipped {target} (exists)")
            continue
        try:
            src = hf_hub_download(repo_id=repo, filename=filename)
            Path(src).replace(path)
            print(f"✅ downloaded {target}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ failed to download {target}: {exc}")


def save_resnet18() -> None:
    path = MODEL_DIR / "resnet18.pt"
    if path.exists():
        print("⚠ skipped resnet18.pt (exists)")
        return
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    torch.save({"model": model, "meta": weights.meta}, path)
    with open(MODEL_DIR / "imagenet_labels.txt", "w") as f:
        for c in weights.meta["categories"]:
            f.write(c + "\n")
    print("✅ saved resnet18.pt")


def export_mobilenet_onnx() -> None:
    path = MODEL_DIR / "mobilenet_v3_small.onnx"
    if path.exists():
        print("⚠ skipped mobilenet_v3_small.onnx (exists)")
        return
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
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
    print("✅ exported mobilenet_v3_small.onnx")
    if not (MODEL_DIR / "imagenet_labels.txt").exists():
        with open(MODEL_DIR / "imagenet_labels.txt", "w") as f:
            for c in weights.meta["categories"]:
                f.write(c + "\n")


def main() -> None:
    print("\n⬇️ Starting model downloads...\n")
    download_keras_models()
    save_resnet18()
    export_mobilenet_onnx()
    print("\n🎉 Downloads complete. Models are stored in ./models/\n")


if __name__ == "__main__":
    main()
