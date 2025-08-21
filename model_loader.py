from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch


@dataclass
class ModelSpec:
    """Description of a model that can be served."""

    key: str
    path: Path
    format: str  # keras | torch | onnx
    description: str
    input_size: tuple[int, int]
    mode: str
    mean: List[float]
    std: List[float]
    labels: List[str]


class BaseModel:
    """Simple wrapper around a loaded model with a unified predict method."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def predict(self, batch: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class KerasModel(BaseModel):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.model = tf.keras.models.load_model(spec.path)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return self.model.predict(batch)


class TorchModel(BaseModel):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.model = torch.load(spec.path, map_location="cpu")
        self.model.eval()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(batch)
            outputs = self.model(tensor)
            return outputs.numpy() if isinstance(outputs, torch.Tensor) else outputs


class ONNXModel(BaseModel):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.session = ort.InferenceSession(str(spec.path), providers=["CPUExecutionProvider"])

    def predict(self, batch: np.ndarray) -> np.ndarray:
        inputs = {self.session.get_inputs()[0].name: batch}
        return self.session.run(None, inputs)[0]
