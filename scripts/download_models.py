"""Backward compatibility wrapper for fetch_model_zoo.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("fetch_model_zoo.py")
    subprocess.run([sys.executable, str(script)], check=False)


if __name__ == "__main__":
    main()
