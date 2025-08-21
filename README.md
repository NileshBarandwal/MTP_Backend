# MTP

Minimal model testing playground. This project shows how to download a
variety of machine learning models from different sources and run a
quick smoke test on each of them.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Downloading models

```bash
python scripts/download_models.py
```
Models are saved under `models/`. Downloads are skipped if the file
already exists.

## Running tests

```bash
python scripts/test_models.py
```
Each test prints a short message indicating success (✅) or failure (❌).

## Repository structure

```
MTP/
├── models/
├── scripts/
│   ├── download_models.py
│   └── test_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Example

```
$ python scripts/test_models.py
✅ Fashion MNIST output shape: (1, 10)
✅ MNIST digits output shape: (1, 10)
✅ GPT-2 sample: Hello ...
✅ DistilBERT sentiment label: 1
```
