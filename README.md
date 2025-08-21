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
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_model_zoo.py
python scripts/test_models.py
flask --app inference_server run --port 8000
```

`fetch_model_zoo.py` downloads small pre-trained checkpoints when possible and
falls back to training tiny models locally if the download fails. Artifacts and
checksums are written to `models/registry.json` and stored under `models/`.

`test_models.py` verifies that each registered model can perform a forward pass
and, if the server is running on `localhost:8000`, performs an HTTP inference
request as well. The script exits non-zero if any check fails.

## Run server

```bash
flask --app inference_server run --port 8000
```

The server hosts the UI at http://127.0.0.1:8000. Upload an image, pick a model
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

## Adding models

To register a new model:

1. Add an entry to `models/registry.json` describing the framework, relative
   path, input specification and preprocessing profile.
2. Update `scripts/fetch_model_zoo.py` with logic to download or train the
   artifact.
3. Run `python scripts/fetch_model_zoo.py && python scripts/test_models.py` to
   verify the model.

## Troubleshooting

* The repository is CPU-only. Large TensorFlow installations may emit oneDNN
  warnings; set `TF_ENABLE_ONEDNN_OPTS=0` to silence them.
* If downloads fail, the fetch script will train small fallback models. These
  are sufficient for tests but may not be accurate.
* The ONNX export for `mobilenet_v3_small` requires the `onnx` package. If it
  cannot be installed, the fetch script will fail for that model.
* Ensure enough disk space for temporary checkpoints.

## Disclaimer

This project is a demo only and must not be used for clinical decision making.
