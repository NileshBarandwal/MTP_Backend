# MTP

This repository simulates a lightweight healthcare AI inference service. A Flask
server exposes REST endpoints and a small web UI where users can upload images
and choose from multiple vision models. Models are downloaded locally so all
inference runs offline once the assets are present.

## Features

* Keras, PyTorch and ONNX model formats handled through a unified interface
* Auto image preprocessing (resize, normalization, color/EXIF handling)
* Endpoints for health checks, model registry, single & batch prediction, and
  simple request metrics
* Basic web interface to upload an image and view predictions

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download models

```bash
python scripts/download_models.py
```

The script fetches or exports several vision models and stores them under
`models/`. Downloads are skipped if the files already exist.

## Run smoke tests

```bash
python scripts/test_models.py
```

Each test loads a model and performs a tiny inference to confirm everything is
working. Failures include a message suggesting you download the missing model.

## Run server

```bash
python inference_server.py
```

The server hosts the UI at http://127.0.0.1:5000. Upload an image, pick a model
from the drop‑down (populated from `/models`), and view the predicted label and
confidence.

### API Endpoints

| Method | Path             | Description                              |
| ------ | ---------------- | ---------------------------------------- |
| GET    | `/health`        | Basic health check                       |
| GET    | `/models`        | List available models                    |
| POST   | `/predict`       | Predict for a single uploaded image      |
| POST   | `/batch_predict` | Predict for multiple images in one call  |
| GET    | `/metrics`       | Simple request counters per model        |

## Models

| Key                  | Description                       | Format   |
| -------------------- | --------------------------------- | -------- |
| `mnist_digits`       | CNN trained on MNIST digits       | Keras    |
| `fashion_mnist`      | Fashion-MNIST classifier          | Keras    |
| `resnet18_imagenet`  | ResNet18 ImageNet classifier      | PyTorch  |
| `mobilenet_v3_small` | MobileNetV3 Small ImageNet model  | ONNX     |

## Repository tree

```
MTP/
├── models/
│   └── .gitkeep
├── scripts/
│   ├── download_models.py
│   └── test_models.py
├── static/
│   └── app.js
├── templates/
│   └── index.html
├── inference_server.py
├── model_loader.py
├── model_registry.py
├── preprocess.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Disclaimer

This project is a demo only and must not be used for clinical decision making.
