# scripts/download.py
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
from sklearn.datasets import fetch_openml

DEFAULT_OUT = Path("data/raw/ames_openml.csv")

OPENML_CANDIDATES = [
    # common aliases on OpenML; one of these usually works
    {"name": "ames_housing", "version": 1},
    {"name": "house_prices", "version": 1},
]

def _try_fetch_once() -> pd.DataFrame:
    last_err = None
    for cfg in OPENML_CANDIDATES:
        try:
            return fetch_openml(as_frame=True, **cfg).frame
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not fetch dataset from OpenML: {last_err}")

def run_download(out_path: Path = DEFAULT_OUT, retries: int = 3, pause: float = 2.5) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If we already have a file, prefer using it (helps when OpenML is flaky)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = _try_fetch_once()
            df.to_csv(out_path, index=False)
            # quick integrity check
            chk = pd.read_csv(out_path, nrows=5)
            if len(chk.columns) == 0:
                raise RuntimeError("Downloaded CSV seems empty")
            return out_path
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(pause * attempt)  # backoff
            else:
                raise RuntimeError(f"OpenML download failed after {retries} tries: {last_err}")

if __name__ == "__main__":
    p = run_download()
    print(f"[OK] Saved: {p}")
