# scripts/transform.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/ames_openml.csv")
SAVE_PATH = Path("data/processed/ames_cleaned.csv")
POSSIBLE_TARGETS = ["SalePrice", "Sale_Price", "saleprice", "sale_price", "PRICE", "price"]
LOG_TARGET = "Log_SalePrice"
CORR_THRESHOLD = 0.6

drop_missing_cols = ["MiscFeature", "MasVnrType", "MasVnrArea", "MS_SubClass"]
manual_drop_corr_pairs = [
    ("GrLivArea", "TotRmsAbvGrd"),
    ("1stFlrSF", "TotalBsmtSF"),
    ("GarageArea", "GarageCars"),
    ("2ndFlrSF", "GrLivArea"),
    ("TotRmsAbvGrd", "2ndFlrSF"),
    ("2ndFlrSF", "HouseStyle"),
]

def detect_target(df: pd.DataFrame) -> str:
    for c in POSSIBLE_TARGETS:
        if c in df.columns:
            return c
    raise ValueError(f"No target column found among {POSSIBLE_TARGETS}")

def run_transform(raw_path: Path = RAW_PATH, save_path: Path = SAVE_PATH) -> Path:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path, low_memory=False)

    # Drop irrelevant/mostly-empty
    df.drop(columns=drop_missing_cols, inplace=True, errors="ignore")

    # Impute
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols):
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    target_col = detect_target(df)
    df[LOG_TARGET] = np.log1p(df[target_col])

    # Multicollinearity drop, protecting target & log target
    protected = {target_col, LOG_TARGET}
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr().abs()

    to_drop = set()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    for col1 in upper.columns:
        for col2 in upper.index:
            if col1 != col2 and upper.loc[col2, col1] > CORR_THRESHOLD:
                if col1 in protected or col2 in protected: 
                    continue
                c1 = abs(corr.at[col1, target_col]) if col1 in corr.columns else 0
                c2 = abs(corr.at[col2, target_col]) if col2 in corr.columns else 0
                to_drop.add(col2 if c1 >= c2 else col1)

    df.drop(columns=list(to_drop & set(df.columns)), inplace=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return save_path

if __name__ == "__main__":
    out = run_transform()
    print(f"[OK] Cleaned dataset saved to: {out}")
