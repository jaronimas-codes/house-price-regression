import argparse, json, os, sys, warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)

# Prefer actual price labels; include Log_SalePrice as fallback for eval/default-row handling
POSSIBLE_TARGETS = [
    "SalePrice", "Sale_Price", "saleprice", "sale_price", "PRICE", "price", "Log_SalePrice"
]

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

def autodetect_input_csv() -> Optional[str]:
    candidates = [
        "data/processed/ames_cleaned.csv",
        "data/processed/ames_clean.csv",
        "data/clean/ames_clean.csv",
        "data/clean/ames_openml.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def detect_target(df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    for c in POSSIBLE_TARGETS:
        if c in df.columns:
            return c
    return None

def safe_read_json(path: str, default=None):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def fmt_money(x: float, currency: str = "$", decimals: int = 0) -> str:
    try:
        return f"{currency}{x:,.{decimals}f}"
    except Exception:
        return f"{currency}{x}"

# --------------------- model artifacts ---------------------
def load_model_and_features(model_dir: Optional[str]) -> tuple:
    """
    Returns (model, feature_names, meta) where meta includes:
      - 'selected_model'
      - 'metrics'  (training metrics dict)
    """
    base = "artifacts/models"
    if model_dir is None:
        model_path = os.path.join(base, "best_model.pkl")
        if not os.path.exists(model_path):
            _err("best_model.pkl not found. Train first.")
        summary_path = os.path.join(base, "metrics_summary.json")
        summary = safe_read_json(summary_path, default={}) or {}
        best = summary.get("selected_model")

        feat_path = os.path.join(base, "feature_names.json")
        if best and os.path.exists(os.path.join(base, best, "feature_names.json")):
            feat_path = os.path.join(base, best, "feature_names.json")
        feats = safe_read_json(feat_path, default=None)

        meta = {"selected_model": best, "metrics": summary.get("metrics", {})}
        return joblib.load(model_path), feats, meta
    else:
        model_path = os.path.join(model_dir, "model.pkl")
        feat_path  = os.path.join(model_dir, "feature_names.json")
        if not os.path.exists(model_path):
            _err(f"Model not found at {model_path}")
        feats = safe_read_json(feat_path, default=None)
        meta = {"selected_model": os.path.basename(model_dir), "metrics": {}}
        return joblib.load(model_path), feats, meta

# --------------------- preprocessing ---------------------
def coerce_boolean_like_to_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        s = df[c]
        vals = set(pd.Series(s).dropna().unique().tolist())
        if vals.issubset({0, 1, "0", "1", True, False}):
            df[c] = s.replace({True: 1, False: 0, "0": 0, "1": 1}).astype(float)

def coerce_and_align(X_df: pd.DataFrame, feature_names: Optional[List[str]]) -> pd.DataFrame:
    obj_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    coerce_boolean_like_to_numeric(X_df, obj_cols)
    still_obj = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if still_obj:
        # training used get_dummies; drop_first choice doesn't matter since we align below
        X_df = pd.get_dummies(X_df, columns=still_obj, drop_first=False)

    # keep the ORIGINAL training order for prediction
    if feature_names:
        for col in feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        extra = [c for c in X_df.columns if c not in feature_names]
        if extra:
            X_df = X_df.drop(columns=extra)
        X_df = X_df[feature_names]
    return X_df

# --------------------- importance utilities ---------------------
def get_builtin_importance(model) -> Optional[np.ndarray]:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_") and getattr(model, "coef_", None) is not None:
        coef = np.asarray(model.coef_)
        return np.abs(coef.ravel() if coef.ndim > 1 else coef)
    return None

def permutation_importance_quick(model, X: np.ndarray, y: Optional[np.ndarray], n_repeats: int = 3, max_n: int = 300) -> Optional[np.ndarray]:
    try:
        if y is None:
            return None
        n = min(max_n, len(y))
        r = permutation_importance(model, X[:n], y[:n], n_repeats=n_repeats, random_state=42, n_jobs=-1)
        return r.importances_mean
    except Exception:
        return None

def build_importance_table(model, feature_names: Optional[List[str]], X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
    if not feature_names:
        return pd.DataFrame(columns=["feature", "importance", "method"])
    imp = get_builtin_importance(model)
    method = None
    if imp is not None and len(imp) == len(feature_names):
        method = "builtin"
    elif X is not None and y is not None:
        imp = permutation_importance_quick(model, X, y)
        method = "permutation" if imp is not None else None

    if imp is None or len(imp) != len(feature_names):
        imp = np.zeros(len(feature_names), dtype=float)
        method = "none"

    df_imp = pd.DataFrame({"feature": feature_names, "importance": imp})
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp["method"] = method
    return df_imp

# --------------------- default one-row input ---------------------
def build_default_input_heuristic(feature_names: Optional[List[str]]) -> pd.DataFrame:
    """Fallback default row if no dataset is available."""
    if not feature_names:
        return pd.DataFrame([{}])
    defaults = {}
    for f in feature_names:
        f_low = f.lower()
        if any(k in f_low for k in ["grliv", "liv", "area", "sf"]):
            defaults[f] = 1500.0
        elif "lot" in f_low:
            defaults[f] = 8000.0
        elif "year" in f_low:
            defaults[f] = 2000.0
        elif any(k in f_low for k in ["garagecars", "garage_cars"]):
            defaults[f] = 2.0
        elif any(k in f_low for k in ["garagearea", "garage_sf"]):
            defaults[f] = 400.0
        elif any(k in f_low for k in ["overallqual", "overall_qual"]):
            defaults[f] = 6.0
        elif any(k in f_low for k in ["overallcond", "overall_cond"]):
            defaults[f] = 5.0
        elif any(k in f_low for k in ["fullbath", "full_bath"]):
            defaults[f] = 2.0
        elif any(k in f_low for k in ["halfbath", "half_bath"]):
            defaults[f] = 1.0
        elif any(k in f_low for k in ["totrms", "rooms", "bedroom"]):
            defaults[f] = 6.0
        elif any(k in f_low for k in ["kitchen"]):
            defaults[f] = 1.0
        else:
            defaults[f] = 0.0
    return pd.DataFrame([defaults])

def build_default_from_reference(feature_names: Optional[List[str]], ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a ONE-ROW default by taking column-wise medians
    in the model feature space (after encoding + alignment).
    """
    if not feature_names:
        return pd.DataFrame([{}])
    tgt = detect_target(ref_df, None)
    X_ref = ref_df.drop(columns=[tgt], errors="ignore").copy()
    X_ref = coerce_and_align(X_ref, feature_names)
    med = np.nanmedian(X_ref.to_numpy(dtype=float), axis=0)
    one = pd.DataFrame([med], columns=feature_names)
    return one

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser(description="Predict house prices with a trained model.")
    ap.add_argument("--in_csv", default=None, help="Input CSV. If omitted, auto-detects a cleaned dataset; if none, uses a median-based one-row input.")
    ap.add_argument("--out_csv", default="artifacts/predictions/preds.csv", help="Where to save predictions.")
    ap.add_argument("--model_dir", default=None, help="Model folder (e.g., artifacts/models/xgb). Default: best model.")
    ap.add_argument("--target", default=None, help="Optional target column to evaluate on (auto-detected otherwise).")
    ap.add_argument("--id_cols", default="", help="Comma-separated column names to passthrough to output (e.g., Id,PID).")
    ap.add_argument("--include_inputs", action="store_true", help="Include all input columns in output CSV.")
    ap.add_argument("--export-importance", default=None, help="Optional path to save full feature importance ranking as CSV.")
    ap.add_argument("--use-default", action="store_true", help="Ignore --in_csv and predict a single median row from a reference dataset if available.")
    # ---- user-friendly price output ----
    ap.add_argument("--delog", dest="delog", action="store_true", help="Convert log predictions back to price with expm1 (default).")
    ap.add_argument("--no-delog", dest="delog", action="store_false", help="Keep predictions on original scale.")
    ap.set_defaults(delog=True)
    ap.add_argument("--currency", default="$", help="Currency symbol/prefix for formatted prices, e.g. $, CHF, €.")
    ap.add_argument("--decimals", type=int, default=0, help="Decimals for formatted prices.")
    ap.add_argument("--no-range", action="store_true", help="Disable expected price range (± training MAE) columns.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Load model + feature list (ORIGINAL training order)
    model, feature_names, meta = load_model_and_features(args.model_dir)
    feature_names = list(feature_names) if feature_names else None

    # ----- Choose input source -----
    if args.in_csv and not args.use_default:
        df_in = load_table(args.in_csv)
        ref_path_used = args.in_csv
    else:
        ref_path = args.in_csv or autodetect_input_csv()
        if ref_path and os.path.exists(ref_path):
            print(f"[INFO] Building a ONE-ROW median input from: {ref_path}")
            ref_df = load_table(ref_path)
            df_in = build_default_from_reference(feature_names, ref_df)
            ref_path_used = ref_path
        else:
            print("[INFO] No reference CSV found. Using heuristic default one-row input.")
            df_in = build_default_input_heuristic(feature_names)
            ref_path_used = "<heuristic-default>"

    # Passthrough IDs if present (usually absent for the synthetic row)
    passthrough_cols = [c.strip() for c in (args.id_cols or "").split(",") if c.strip()]
    passthrough_cols = [c for c in passthrough_cols if c in df_in.columns]

    # Separate features / optional target
    tgt = detect_target(df_in, args.target)
    X_df = df_in.drop(columns=[tgt], errors="ignore") if tgt else df_in.copy()
    y_true = None
    y_true_scale_is_log = False
    if tgt and tgt in df_in.columns:
        # if the target is log, we can compare on price by expm1 later
        if tgt.lower() in {"log_saleprice", "log_sale_price"}:
            y_true_scale_is_log = True
        y_true = pd.to_numeric(df_in[tgt], errors="coerce").to_numpy(dtype=float)

    # Align to training feature order
    X_df = coerce_and_align(X_df, feature_names)
    X = X_df.to_numpy(dtype=float)

    # Importance table (report only; do NOT reorder X)
    imp_df = build_importance_table(model, feature_names, X=X, y=None if y_true is None else (np.expm1(y_true) if y_true_scale_is_log and args.delog else y_true))
    if not imp_df.empty:
        print("[INFO] Top 10 features by importance:")
        for _, row in imp_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.6f} ({row['method']})")
    if args.export_importance:
        os.makedirs(os.path.dirname(args.export_importance), exist_ok=True)
        imp_df.to_csv(args.export_importance, index=False)
        print(f"[OK] Exported importance ranking → {os.path.abspath(args.export_importance)}")

    # Predict (log-scale raw → price if delog=True)
    y_pred_raw = model.predict(X).astype(float)
    price = np.expm1(y_pred_raw) if args.delog else y_pred_raw

    # Training MAE (assuming your training saved 'real_mae' in currency units)
    sel = meta.get("selected_model")
    train_metrics = meta.get("metrics", {}).get(sel, {}) if sel else {}
    real_mae = train_metrics.get("real_mae", None)
    if real_mae is not None and not np.isfinite(real_mae):
        real_mae = None

    # Compose output
    out = df_in.copy() if args.include_inputs else pd.DataFrame(index=df_in.index)
    for c in passthrough_cols:
        out[c] = df_in[c]
    out["y_pred_raw"] = y_pred_raw
    out["price"] = price

    # Expected price range using training MAE (if available)
    if not args.no_range and real_mae is not None:
        out["price_lo"] = np.maximum(0.0, out["price"] - real_mae)
        out["price_hi"] = out["price"] + real_mae

    # Optional evaluation (compare on same scale as 'price' column the user sees)
    if y_true is not None:
        y_eval = np.expm1(y_true) if (y_true_scale_is_log and args.delog) else y_true
        mask = ~np.isnan(y_eval)
        if mask.sum() > 0:
            mae  = float(mean_absolute_error(y_eval[mask], price[mask]))
            rmse = float(root_mean_squared_error(y_eval[mask], price[mask]))
            r2   = float(r2_score(y_eval[mask], price[mask]))
            print(f"[EVAL] RMSE={rmse:,.2f}  R²={r2:.4f}  MAE={mae:,.2f}")
            metrics_path = os.path.join(os.path.dirname(args.out_csv), "metrics_eval.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": sel,
                    "in_csv": os.path.abspath(ref_path_used),
                    "target": tgt,
                    "rmse": rmse, "r2": r2, "mae": mae,
                    "n_eval": int(mask.sum())
                }, f, indent=2)
            out["y_true"] = y_eval  # store in same scale as price

    # Pretty formatted price columns
    out["price_fmt"] = [fmt_money(x, args.currency, args.decimals) for x in out["price"]]
    if "price_lo" in out and "price_hi" in out:
        out["range_fmt"] = [
            f"{fmt_money(l, args.currency, args.decimals)} – {fmt_money(h, args.currency, args.decimals)}"
            for l, h in zip(out["price_lo"], out["price_hi"])
        ]

    # Save predictions
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved predictions: {os.path.abspath(args.out_csv)}")
    print("[NOTE] Default row was built from medians in model feature space, matching the training columns/order.")
    # Friendly console summary if single row
    if len(out) == 1:
        print(f"[PREDICT] Price = {out.loc[out.index[0],'price_fmt']}"
              + (f"  (range {out.loc[out.index[0],'range_fmt']})" if 'range_fmt' in out.columns else ""))

if __name__ == "__main__":
    main()
