# Minimal MNIST Keras Project

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .\\.venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python -m src.train --config ./configs/train.yaml
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

To enable MLflow logging, set `log_mlflow: true` in `configs/train.yaml` and ensure an MLflow tracking server (e.g., `mlflow ui`) is running.

After starting the server, open <http://localhost:8000> in a browser for a simple HTML interface to upload images and view predictions.

## Example `curl`

Base64 JSON:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"<BASE64_OF_28x28_GRAYSCALE>"}'
```

File upload:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@tests/sample_digit.png"
```

## Outputs

Trained models are saved under `registry/models/<model_name>/<timestamp>/` with:
- `model.h5`
- `params.json`
- `metrics.json`
- `eval_summary.json`
- `code_sha.txt` (if available)
- `confusion_matrix.json`

The latest model version is recorded in `registry/models/<model_name>/latest.txt`.
