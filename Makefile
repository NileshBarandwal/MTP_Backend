.PHONY: setup fetch test serve clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

fetch:
	python scripts/fetch_model_zoo.py

test:
	python scripts/test_models.py

serve:
	uvicorn inference_server:app --reload --port 8000

clean:
	rm -f models/*.h5 models/*.pt models/*.onnx models/imagenet_labels.txt
