from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open('w') as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def git_sha_or_none() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None
