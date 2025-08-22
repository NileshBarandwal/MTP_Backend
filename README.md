# Minimal MNIST Keras Project

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m src.train --config ./configs/train.yaml
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

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

The latest model version is recorded in `registry/models/<model_name>/latest.txt`.
