from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from .utils import read_json

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')

REGISTRY_ROOT = Path('./registry/models')
DEFAULT_MODEL_NAME = 'mnist_cnn'
MODEL_CACHE: dict[tuple[str, str], tuple[tf.keras.Model, dict]] = {}
DEFAULT_VERSION = ''


def _latest_version(model_name: str) -> str | None:
    model_root = REGISTRY_ROOT / model_name
    latest_file = model_root / 'latest.txt'
    if latest_file.exists():
        return latest_file.read_text().strip()
    return None


def _load_model(model_name: str, version: str) -> tuple[tf.keras.Model, dict]:
    key = (model_name, version)
    if key not in MODEL_CACHE:
        model_path = REGISTRY_ROOT / model_name / version / 'model.h5'
        params_path = REGISTRY_ROOT / model_name / version / 'params.json'
        if not model_path.exists():
            raise HTTPException(status_code=404, detail='Model version not found')
        model = tf.keras.models.load_model(model_path)
        params = read_json(params_path) if params_path.exists() else {}
        MODEL_CACHE[key] = (model, params)
    return MODEL_CACHE[key]


def get_model(model_name: str | None, version: str | None) -> tuple[tf.keras.Model, dict, str]:
    name = model_name or DEFAULT_MODEL_NAME
    ver = version or _latest_version(name)
    if ver is None:
        raise HTTPException(status_code=404, detail='No model version found')
    model, params = _load_model(name, ver)
    return model, params, ver


def load_default() -> None:
    global DEFAULT_VERSION
    _, _, DEFAULT_VERSION = get_model(DEFAULT_MODEL_NAME, None)


load_default()


class ImagePayload(BaseModel):
    image_base64: str


@app.get('/', response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = Path('templates/index.html')
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse('<h1>index.html not found</h1>', status_code=404)


@app.get('/healthz')
def healthz():
    return {'status': 'ok', 'model_version': DEFAULT_VERSION}


@app.get('/models')
def list_models():
    models = {}
    if REGISTRY_ROOT.exists():
        for m in REGISTRY_ROOT.iterdir():
            if m.is_dir():
                versions = [d.name for d in sorted(m.iterdir()) if d.is_dir()]
                models[m.name] = versions
    return models


@app.post('/predict')
async def predict(
    model_name: str | None = None,
    version: str | None = None,
    payload: ImagePayload | None = None,
    file: UploadFile | None = File(None),
):
    if payload is None and file is None:
        raise HTTPException(status_code=400, detail='No image provided')
    if payload is not None:
        img_bytes = base64.b64decode(payload.image_base64)
    else:
        img_bytes = await file.read()

    image = Image.open(io.BytesIO(img_bytes)).convert('L').resize((28, 28))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))

    model, _, used_version = get_model(model_name, version)
    preds = model.predict(arr)
    label = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return {'label': label, 'confidence': confidence, 'model_version': used_version}
