from __future__ import annotations

from PIL import Image, ImageOps
import numpy as np

from model_loader import ModelSpec


def prepare_image(file, spec: ModelSpec) -> np.ndarray:
    """Load an image file and transform it into a model-ready numpy array."""
    img = Image.open(file).convert(spec.mode)
    img = ImageOps.exif_transpose(img)
    img = ImageOps.fit(img, spec.input_size, method=Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    if spec.mode == "L":
        arr = arr[..., None]
    if spec.format in {"torch", "onnx"}:
        arr = arr.transpose(2, 0, 1)
        for c in range(arr.shape[0]):
            arr[c] = (arr[c] - spec.mean[c]) / spec.std[c]
    else:  # keras models
        arr = (arr - spec.mean[0]) / spec.std[0]
    return arr[None, ...]
