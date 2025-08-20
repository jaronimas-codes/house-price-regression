# scripts/download.py
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
from sklearn.datasets import fetch_openml

DEFAULT_OUT = Path("data/raw/ames_openml.csv")

def run_download(out_path: Path = DEFAULT_OUT) -> Path:
    """Download the Ames Housing dataset from OpenML and save as CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try common OpenML names, use first that works
    last_err = None
    for name, kwargs in [
        ("ames_housing", {"version": 1}),
        ("house_prices", {"version": 1}),
    ]:
        try:
            df = fetch_openml(name=name, as_frame=True, **kwargs).frame
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"Could not fetch dataset from OpenML: {last_err}")

    df.to_csv(out_path, index=False)

    # Verify readability
    _chk = pd.read_csv(out_path, low_memory=False)
    if _chk.empty or _chk.shape[1] == 0:
        raise RuntimeError("Saved CSV seems empty/corrupt")

    return out_path

if __name__ == "__main__":
    try:
        dest = run_download(DEFAULT_OUT)
        print(f"[OK] Saved and verified: {dest}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
