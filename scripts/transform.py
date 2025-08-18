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

    # Prevent dropping these key columns
    protected_features = {"Sale_Price", "Log_SalePrice"}

    # Get absolute correlation matrix for numeric features
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr().abs()

    # Compute feature-target correlations
    target_corrs = corr_matrix["Sale_Price"].drop("Sale_Price")

    # Initialize set for dropping
    to_drop = set()

    # Examine upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col1 in upper.columns:
        for col2 in upper.index:
            if col1 != col2 and upper.loc[col2, col1] > CORR_THRESHOLD:
                if col1 in protected_features or col2 in protected_features:
                    continue  # Skip dropping protected features

                # Check correlation with target
                corr1 = abs(corr_matrix.at[col1, "Sale_Price"]) if col1 in corr_matrix.columns else 0
                corr2 = abs(corr_matrix.at[col2, "Sale_Price"]) if col2 in corr_matrix.columns else 0

                # Drop the one less correlated with target
                if corr1 >= corr2:
                    to_drop.add(col2)
                else:
                    to_drop.add(col1)

    # Drop final list (if still present in df)
    df.drop(columns=list(to_drop & set(df.columns)), inplace=True)
    print(f"[INFO] Dropped {len(to_drop)} multicollinear features.")


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
