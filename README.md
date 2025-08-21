# MTP

This repository simulates a simple healthcare AI inference setup. A Flask server
serves a small web interface where users can upload an image and choose a model
for prediction. Models are downloaded locally via helper scripts so inference
runs entirely offline once the assets are fetched.

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

The script retrieves several vision and NLP models and stores them under
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

The server hosts the UI at http://127.0.0.1:5000. Upload an image, pick a model,
and view the predicted label and confidence.

## Models

| Key | Description | Expected Input |
| --- | ----------- | -------------- |
| `mnist_digits` | CNN trained on MNIST digits | 28×28 grayscale digit |
| `fashion_mnist` | Fashion-MNIST classifier | 28×28 grayscale fashion item |
| `resnet18_imagenet` | Torchvision ResNet18 | RGB image; resized to 224×224 |

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
├── requirements.txt
├── README.md
└── .gitignore
```

## Disclaimer

This project is a demo only and must not be used for clinical decision making.
