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
MODEL_NAME = 'mnist_cnn'
MODEL = None
MODEL_VERSION = ''
PARAMS = {}


def load_latest() -> None:
    global MODEL, MODEL_VERSION, PARAMS
    model_root = REGISTRY_ROOT / MODEL_NAME
    latest_file = model_root / 'latest.txt'
    if not latest_file.exists():
        return
    version = latest_file.read_text().strip()
    model_path = model_root / version / 'model.h5'
    params_path = model_root / version / 'params.json'
    if model_path.exists():
        MODEL = tf.keras.models.load_model(model_path)
        MODEL_VERSION = version
        PARAMS = read_json(params_path)


load_latest()


class ImagePayload(BaseModel):
    image_base64: str


@app.get('/', response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = Path('static/index.html')
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse('<h1>index.html not found</h1>', status_code=404)


@app.get('/healthz')
def healthz():
    return {'status': 'ok', 'model_version': MODEL_VERSION}


@app.post('/predict')
async def predict(payload: ImagePayload | None = None, file: UploadFile | None = File(None)):
    if payload is None and file is None:
        raise HTTPException(status_code=400, detail='No image provided')
    if payload is not None:
        img_bytes = base64.b64decode(payload.image_base64)
    else:
        img_bytes = await file.read()

    image = Image.open(io.BytesIO(img_bytes)).convert('L').resize((28, 28))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))

    if MODEL is None:
        raise HTTPException(status_code=500, detail='Model not loaded')

    preds = MODEL.predict(arr)
    label = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return {'label': label, 'confidence': confidence, 'model_version': MODEL_VERSION}
