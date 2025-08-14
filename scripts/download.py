#!/usr/bin/env python3
# downloads.py

import os
import sys
import pandas as pd
from sklearn.datasets import fetch_openml

RAW_DIR = "data/raw"
RAW_FILE = os.path.join(RAW_DIR, "ames_openml.csv")

# Ensure directory exists
os.makedirs(RAW_DIR, exist_ok=True)

print("[INFO] Downloading Ames Housing dataset from OpenML...")

try:
    # Download from OpenML
    df = fetch_openml("ames_housing", version=1, as_frame=True).frame
    df.to_csv(RAW_FILE, index=False)
    print(f"[OK] Saved dataset to: {RAW_FILE}")
except Exception as e:
    print(f"[ERROR] Failed to download dataset: {e}")
    sys.exit(1)

# Verify the saved file can be read
print("[INFO] Verifying the saved CSV file...")
if not os.path.exists(RAW_FILE):
    print(f"[ERROR] File not found: {RAW_FILE}")
    sys.exit(1)

try:
    df = pd.read_csv(RAW_FILE, low_memory=False)
    print(f"[OK] Loaded: {RAW_FILE}")
    print(f"Shape: {df.shape}")
    print("Columns (first 10):", list(df.columns)[:10])
    print("\nHead:\n", df.head())
except Exception as e:
    print(f"[ERROR] Failed to load CSV: {e}")
    sys.exit(1)
