# MNIST TensorFlow Mini-Project

## Project Overview
A lightweight example demonstrating an end-to-end machine learning workflow on the MNIST dataset using TensorFlow/Keras. The project covers dataset handling, model training, evaluation, and a simple FastAPI service for real-time predictions.

## Features
- CNN model built with TensorFlow/Keras
- Configurable training with reproducible seeds
- Optional MLflow experiment logging
- FastAPI server with JSON and file upload prediction endpoints
- Frontend page with model/version selection and image upload interface

## Installation & Requirements
- Python 3.10+
- Recommended to use a virtual environment

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage Guide
### Train
```bash
python -m src.train --config ./configs/train.yaml
# or
scripts/run_train.sh
```

### Serve
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
# or
scripts/run_serve.sh
```
After the server starts, visit <http://localhost:8000> for the upload interface with model selection.

### Make Predictions
- **Base64 JSON**
```bash
curl -X POST "http://localhost:8000/predict?model_name=<MODEL>&version=<VERSION>" \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"<BASE64_OF_28x28_GRAYSCALE>"}'
```
- **File upload**
```bash
curl -X POST "http://localhost:8000/predict?model_name=<MODEL>&version=<VERSION>" \
  -F "file=@path/to/image.png"
```

## API Documentation
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/healthz` | Health check and model version |
| GET | `/models` | List available models and versions |
| POST | `/predict` | Predict digit from image (JSON base64 or multipart file; optional `model_name` and `version` query params) |

## Configuration
Training parameters are stored in `configs/train.yaml`:

| Key | Description |
|-----|-------------|
| `seed` | Random seed for reproducibility |
| `epochs` | Training epochs |
| `batch_size` | Batch size for datasets |
| `lr` | Learning rate |
| `model_name` | Name for saved model directory |
| `registry_root` | Root path for model registry |
| `data_root` | Root data directory |
| `save_every` | Checkpoint frequency (unused but kept for symmetry) |
| `log_mlflow` | Toggle MLflow experiment logging |

## Outputs & Model Registry
Artifacts are saved under `registry/models/<model_name>/<timestamp>/`:
- `model.h5`
- `params.json`
- `metrics.json`
- `eval_summary.json`
- `confusion_matrix.json`
- `code_sha.txt` (if git is available)

The latest model version is recorded in `registry/models/<model_name>/latest.txt` and preprocessing params in `data/processed/params.json`.

## Project Structure
```
├── configs/        # Training configuration
├── data/           # Data and processed params
├── registry/       # Model registry with versions
├── scripts/        # Helper shell scripts
├── src/            # Model, utils, training and serving code
├── templates/      # Frontend HTML page
├── static/         # CSS assets for frontend
└── README.md
```

## Contributing / Future Work
Pull requests are welcome! Possible extensions include dataset augmentation, model improvements, and Docker packaging.

## License
TBD

