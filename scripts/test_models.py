from __future__ import annotations

"""Run smoke tests for all registered models."""

import io
import sys
import subprocess
import importlib

import numpy as np
import requests
from PIL import Image

import model_bootstrap as mb  # type: ignore

ROOT = mb.project_root()
sys.path.append(str(ROOT))
import model_registry as mr

FETCH_SCRIPT = ROOT / "scripts" / "fetch_model_zoo.py"


# ---------------------------------------------------------------------------


def _ensure_models() -> bool:
    missing = [spec.path for spec in mr.MODEL_SPECS.values() if not spec.path.exists()]
    if not missing:
        return True
    names = ", ".join(str(p) for p in missing)
    print(f"Missing models: {names}. Running fetch script...")
    subprocess.run([sys.executable, str(FETCH_SCRIPT)], check=False)
    importlib.reload(mr)
    missing = [spec.path for spec in mr.MODEL_SPECS.values() if not spec.path.exists()]
    if missing:
        print("Still missing models:", ", ".join(str(p) for p in missing))
        return False
    # explicit check for Keras artifacts
    for name in ("mnist_digits.h5", "fashion_mnist.h5"):
        path = mb.resolve_model_path(name)
        if not path.is_file():
            print(f"{path} missing after fetch")
            return False
    return True


def _synthetic_input(spec) -> np.ndarray:
    h, w = spec.input_size
    if spec.mode == "L":
        return np.random.rand(1, h, w, 1).astype("float32")
    else:
        return np.random.rand(1, 3, h, w).astype("float32")


# ---------------------------------------------------------------------------


def run_local_tests() -> bool:
    ok = True
    for key, spec in mr.MODEL_SPECS.items():
        try:
            model = mr.get_model(key)
            dummy = _synthetic_input(spec)
            preds = model.predict(dummy)
            if np.isnan(preds).any() or np.isinf(preds).any():
                raise RuntimeError("non-finite outputs")
            idx = int(np.argmax(preds[0]))
            print(f"✅ {key}: top-1 class {idx}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {key} failed: {exc}")
            ok = False
    return ok


def run_http_test() -> bool:
    url = "http://127.0.0.1:8000"
    try:
        r = requests.get(f"{url}/health", timeout=1)
    except requests.exceptions.RequestException:
        print("⚠️ server not running; skipping HTTP test")
        return True
    if r.status_code != 200:
        print("⚠️ /health returned", r.status_code)
        return False
    spec = next(iter(mr.MODEL_SPECS.values()))
    dummy = (np.random.rand(*spec.input_size) * 255).astype("uint8")
    if spec.mode == "L":
        img = Image.fromarray(dummy, mode="L")
    else:
        img = Image.fromarray(dummy.transpose(1, 2, 0), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("test.png", buf, "image/png")}
    data = {"model_key": spec.key}
    r = requests.post(f"{url}/predict", files=files, data=data, timeout=5)
    if r.status_code != 200:
        print("❌ HTTP inference failed:", r.text)
        return False
    if "prediction_index" not in r.json():
        print("❌ unexpected HTTP response:", r.text)
        return False
    print("✅ HTTP inference ok")
    return True


# ---------------------------------------------------------------------------


def main() -> None:
    if not _ensure_models():
        sys.exit(1)
    ok_local = run_local_tests()
    ok_http = run_http_test()
    sys.exit(0 if ok_local and ok_http else 1)


if __name__ == "__main__":
    main()
