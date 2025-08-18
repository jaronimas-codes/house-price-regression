import argparse, json, os, sys, warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

POSSIBLE_TARGETS = ["SalePrice", "Sale_Price", "saleprice", "sale_price", "PRICE", "price"]

# --------------------- IO helpers ---------------------
def _err(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(1)

def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        _err(f"Input not found: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        return pd.read_csv(path, sep=";", low_memory=False)

def detect_target(df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    for c in POSSIBLE_TARGETS:
        if c in df.columns:
            return c
    return None

def load_model_and_features(model_dir: Optional[str]) -> tuple:
    """
    If model_dir is None, load artifacts/models/best_model.pkl and infer feature_names.json
    from the selected model subfolder recorded in metrics_summary.json.
    """
    base = "artifacts/models"
    if model_dir is None:
        model_path = os.path.join(base, "best_model.pkl")
        if not os.path.exists(model_path):
            _err("best_model.pkl not found. Train first.")
        # try to pick feature names from the selected model's folder
        feat_path = os.path.join(base, "feature_names.json")  # fallback
        summary_path = os.path.join(base, "metrics_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            best = summary.get("selected_model")
            if best and os.path.exists(os.path.join(base, best, "feature_names.json")):
                feat_path = os.path.join(base, best, "feature_names.json")
        feats = json.load(open(feat_path, "r", encoding="utf-8")) if os.path.exists(feat_path) else None
        return joblib.load(model_path), feats, {"selected_model": summary.get("selected_model") if os.path.exists(summary_path) else None}
    else:
        model_path = os.path.join(model_dir, "model.pkl")
        feat_path  = os.path.join(model_dir, "feature_names.json")
        if not os.path.exists(model_path):
            _err(f"Model not found at {model_path}")
        feats = json.load(open(feat_path, "r", encoding="utf-8")) if os.path.exists(feat_path) else None
        return joblib.load(model_path), feats, {"selected_model": os.path.basename(model_dir)}

# --------------------- preprocessing ---------------------
def coerce_boolean_like_to_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    """In place: map {True,False,'1','0'} to floats where possible."""
    for c in cols:
        s = df[c]
        vals = set(pd.Series(s).dropna().unique().tolist())
        if vals.issubset({0, 1, "0", "1", True, False}):
            df[c] = s.replace({True: 1, False: 0, "0": 0, "1": 1}).astype(float)

def coerce_and_align(X_df: pd.DataFrame, feature_names: Optional[List[str]]) -> pd.DataFrame:
    # 1) Coerce simple boolean/0-1 text to numeric
    obj_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    coerce_boolean_like_to_numeric(X_df, obj_cols)

    # 2) One-hot any remaining objects
    still_obj = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if still_obj:
        X_df = pd.get_dummies(X_df, columns=still_obj, drop_first=False)

    # 3) Align to training feature order
    if feature_names:
        for col in feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        extra = [c for c in X_df.columns if c not in feature_names]
        if extra:
            X_df = X_df.drop(columns=extra)
        X_df = X_df[feature_names]
    return X_df

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser(description="Predict house prices with a trained model.")
    ap.add_argument("--in_csv", required=True, help="Input CSV (preferably post-transform).")
    ap.add_argument("--out_csv", default="artifacts/predictions/preds.csv", help="Where to save predictions.")
    ap.add_argument("--model_dir", default=None, help="Model folder (e.g., artifacts/models/xgb). Default: best model.")
    ap.add_argument("--target", default=None, help="Optional target column to evaluate on (auto-detected otherwise).")
    ap.add_argument("--id_cols", default="", help="Comma-separated column names to passthrough to output (e.g., Id,PID).")
    ap.add_argument("--include_inputs", action="store_true", help="Include all input columns in output CSV.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Load model + feature list
    model, feature_names, model_meta = load_model_and_features(args.model_dir)

    # Load input
    df_in = load_table(args.in_csv)
    passthrough_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    passthrough_cols = [c for c in passthrough_cols if c in df_in.columns]

    # Separate features / optional target
    tgt = detect_target(df_in, args.target)
    X_df = df_in.drop(columns=[tgt]) if tgt and tgt in df_in.columns else df_in.copy()
    y_true = None
    if tgt and tgt in df_in.columns:
        y_true = pd.to_numeric(df_in[tgt], errors="coerce").to_numpy(dtype=float)

    # Build feature matrix like in training
    X_df = coerce_and_align(X_df, feature_names)
    X = X_df.to_numpy(dtype=float)

    # Predict
    y_pred = model.predict(X).astype(float)

    # Compose output
    if args.include_inputs:
        out = df_in.copy()
    else:
        out = pd.DataFrame(index=df_in.index)
    # add passthrough IDs early for convenience
    for c in passthrough_cols:
        out[c] = df_in[c]
    out["y_pred"] = y_pred

    # Optional evaluation
    if y_true is not None:
        mask = ~np.isnan(y_true)
        if mask.sum() > 0:
            mae  = float(mean_absolute_error(y_true[mask], y_pred[mask]))
            rmse = float(root_mean_squared_error(y_true[mask], y_pred[mask]))
            r2   = float(r2_score(y_true[mask], y_pred[mask]))
            print(f"[EVAL] RMSE={rmse:,.2f}  R²={r2:.4f}  MAE={mae:,.2f}")

            # Save eval metrics next to out_csv
            metrics_path = os.path.join(os.path.dirname(args.out_csv), "metrics_eval.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_meta.get("selected_model"),
                    "in_csv": os.path.abspath(args.in_csv),
                    "target": tgt,
                    "rmse": rmse, "r2": r2, "mae": mae,
                    "n_eval": int(mask.sum())
                }, f, indent=2)

            # optional y_true back for inspection
            out["y_true"] = y_true

    # Save predictions
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved predictions: {os.path.abspath(args.out_csv)}")

if __name__ == "__main__":
    main()