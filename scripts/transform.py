import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/ames_openml.csv"
SAVE_PATH = "data/processed/ames_cleaned.csv"
TARGET = "Sale_Price"
LOG_TARGET = "Log_SalePrice"
CORR_THRESHOLD = 0.6

# Features to drop explicitly (from EDA)
drop_missing_cols = ["MiscFeature", "MasVnrType", "MasVnrArea", "MS_SubClass"]
manual_drop_corr_pairs = [
    ("GrLivArea", "TotRmsAbvGrd"),
    ("1stFlrSF", "TotalBsmtSF"),
    ("GarageArea", "GarageCars"),
    ("2ndFlrSF", "GrLivArea"),
    ("TotRmsAbvGrd", "2ndFlrSF"),
    ("2ndFlrSF", "HouseStyle")
]

def main():
    if not os.path.exists(RAW_PATH):
        print(f"[ERROR] File not found: {RAW_PATH}")
        return

    df = pd.read_csv(RAW_PATH)

    # Drop irrelevant or mostly-empty columns
    df.drop(columns=drop_missing_cols, inplace=True, errors="ignore")

    # Impute missing values in remaining columns
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Log-transform target
    if TARGET in df.columns:
        df[LOG_TARGET] = np.log1p(df[TARGET])
    else:
        raise ValueError(f"Target column '{TARGET}' not found.")

    # Remove one feature from each manually identified correlated pair
    to_drop = set()
    for f1, f2 in manual_drop_corr_pairs:
        if f1 in df.columns and f2 in df.columns:
            to_drop.add(f2)  # drop second of pair

    # Auto-detect additional multicollinearity (numeric only)
    corr_matrix = df.select_dtypes(include=np.number).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    auto_drop = [
                    column for column in upper.columns 
                    if any(upper[column] > CORR_THRESHOLD) and column not in {TARGET, LOG_TARGET}
                ]
    to_drop.update(auto_drop)

    df.drop(columns=[col for col in to_drop if col in df.columns], inplace=True)

    # Sanity check: No missing values?
    missing = df.isnull().sum().sum()
    if missing == 0:
        print("No missing values remain.")
    else:
        print(f"⚠️ WARNING: {missing} missing values still present.")

    # Save cleaned data
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f"[INFO] Cleaned dataset saved to: {SAVE_PATH}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
