# app.py
from __future__ import annotations

import os, sys, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# ---------- Scripts (must exist and be importable) ----------
from scripts.download import run_download
from scripts.transform import run_transform
from scripts.train import run_train  # ensure it writes artifacts to artifacts/models
from scripts.eda import run_visual_eda

# ===================== Paths (unified) =====================
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PATH     = PROJECT_ROOT / "data" / "raw" / "ames_openml.csv"
CLEAN_PATH   = PROJECT_ROOT / "data" / "processed" / "ames_cleaned.csv"
MODEL_DIR    = PROJECT_ROOT / "artifacts" / "models"

st.set_page_config(page_title="🏡 House Price ML Suite", layout="wide")
st.title("🏡 House Prices — Full ML Workflow")

# =================== Cache helpers ===================
def _summary_mtime(models_dir: Path) -> float:
    p = Path(models_dir) / "metrics_summary.json"
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0

@st.cache_resource(show_spinner=False)
def load_best_model(models_dir: Path, summary_mtime: float, cache_bust: float):
    """
    Cached loader for model + feature names + summary.
    Cache key includes (models_dir, summary_mtime, cache_bust) so it
    invalidates when files change or when we set a manual token.
    """
    models_dir = Path(models_dir)
    summary_path  = models_dir / "metrics_summary.json"
    features_path = models_dir / "feature_names.json"
    model_path    = models_dir / "best_model.pkl"

    if not (summary_path.exists() and features_path.exists() and model_path.exists()):
        return None, [], {}, models_dir

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    try:
        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    except Exception:
        feature_names = []

    try:
        model = joblib.load(model_path)
    except Exception:
        model = None

    return model, feature_names, summary, models_dir

@st.cache_data
def load_local_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# =================== Feature helpers ===================
POSSIBLE_TARGETS_ALL = ["Log_SalePrice","SalePrice","Sale_Price","saleprice","sale_price","PRICE","price"]

_INT_NAME_HINTS = (
    "year","yr","garagecars","garage_cars","fullbath","full_bath","halfbath","half_bath",
    "kitchen","bedroom","bedrooms","totrms","rooms","bsmt_full_bath","bsmt_half_bath",
    "fireplaces","mosold","mo_sold","month","mo","day"
)

def resolve_model_and_feature_names(model, fallback_names):
    """
    Return (final_estimator, feature_names_after_preprocessing).
    Works for bare estimators and sklearn Pipelines with 'preprocessor' and 'model' steps.
    """
    final_est = model
    feature_names = list(fallback_names or [])

    if hasattr(model, "named_steps"):  # Pipeline
        steps = model.named_steps
        final_est = steps.get("model", list(steps.values())[-1])
        pre = steps.get("preprocessor")
        if pre is not None:
            try:
                feature_names = list(pre.get_feature_names_out())
            except Exception:
                pass

    if not feature_names:
        n = getattr(final_est, "n_features_in_", None)
        if isinstance(n, int) and n > 0:
            feature_names = [f"f{i}" for i in range(n)]

    return final_est, feature_names

def get_feature_ranking(model, feature_names, cleaned_sample_path=CLEAN_PATH, top_k=25):
    summary = read_json(MODEL_DIR / "metrics_summary.json", {})
    selected = summary.get("selected_model")
    if selected:
        csv_path = MODEL_DIR / selected / "feature_importances.csv"
        if csv_path.exists():
            imp = pd.read_csv(csv_path)
            if {"feature","importance"}.issubset(imp.columns):
                imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
                return imp["feature"].tolist(), imp

    # Coefficients (e.g., linear)
    if hasattr(model, "coef_") and model.coef_ is not None:
        coef = np.asarray(model.coef_).ravel()
        imp = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef)})
        imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
        return imp["feature"].tolist(), imp

    # Permutation importance fallback
    if cleaned_sample_path and Path(cleaned_sample_path).exists() and feature_names:
        ref = pd.read_csv(cleaned_sample_path)
        X_df = ref.drop(columns=[c for c in POSSIBLE_TARGETS_ALL if c in ref.columns], errors="ignore")
        X_df = pd.get_dummies(X_df, drop_first=False)

        # align columns
        for col in feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        extra = [c for c in X_df.columns if c not in feature_names]
        if extra:
            X_df = X_df.drop(columns=extra)
        X_df = X_df[feature_names].fillna(0)

        tgt = next((c for c in POSSIBLE_TARGETS_ALL if c in ref.columns), None)
        y = pd.to_numeric(ref[tgt], errors="coerce").fillna(0).to_numpy() if tgt else np.zeros(len(X_df))

        n = min(600, len(X_df))
        try:
            r = permutation_importance(model, X_df.iloc[:n].to_numpy(dtype=float), y[:n],
                                       n_repeats=5, random_state=42, n_jobs=-1)
            imp = pd.DataFrame({"feature": feature_names, "importance": r.importances_mean})
            imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
            return imp["feature"].tolist(), imp
        except Exception:
            pass

    imp = pd.DataFrame({"feature": feature_names, "importance": np.nan})
    return list(feature_names), imp

def _infer_kind(col_values: np.ndarray, name: str) -> str:
    v = col_values[np.isfinite(col_values)]
    if v.size == 0:
        return "int" if any(h in name.lower() for h in _INT_NAME_HINTS) else "float"
    uniq = np.unique(np.round(v, 6))
    if set(uniq).issubset({0.0, 1.0}):
        return "binary"
    if np.allclose(v, np.round(v)) or any(h in name.lower() for h in _INT_NAME_HINTS):
        return "int"
    return "float"

def _domain_bounds(name: str) -> tuple[float|None, float|None, str|None]:
    n = name.lower()
    cur_year = datetime.now().year
    if any(k in n for k in [
        "fireplace","fullbath","full_bath","halfbath","half_bath",
        "bsmt_full_bath","bsmt_half_bath","bedroom","bedrooms",
        "kitchen","totrms","rooms","garagecars","garage_cars"
    ]):
        return 0, None, "int"
    if any(k in n for k in [
        "sf","area","porch","deck","wooddeck","openporch","enclosedporch","screenporch",
        "3ssnporch","lotfrontage","lot_frontage","masvnrarea","poolarea","miscval"
    ]):
        return 0.0, None, "float"
    if "year" in n or n.endswith("_yr_blt") or n.endswith("yrblt") or "yr" in n:
        return 1870, cur_year + 1, "int"
    if "mosold" in n or "month" in n or n.startswith("mo_") or n.endswith("_mo"):
        return 1, 12, "int"
    if "day" in n:
        return 1, 31, "int"
    return None, None, None

def _align_like_training_ui(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X = df.drop(columns=[c for c in POSSIBLE_TARGETS_ALL if c in df.columns], errors="ignore").copy()
    obj_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=False)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    extra = [c for c in X.columns if c not in feature_names]
    if extra:
        X = X.drop(columns=extra)
    return X[feature_names]

def build_defaults_and_ui_specs_from_csv(ref_path: Path, feature_names: list[str]) -> tuple[dict, dict]:
    defaults = {c: 0.0 for c in feature_names}
    specs    = {c: {"kind":"float","min":0.0,"median":0.0,"max":1.0} for c in feature_names}
    try:
        ref = pd.read_csv(ref_path, low_memory=False)
    except Exception:
        return defaults, specs

    Xdf = _align_like_training_ui(ref, feature_names)
    X   = Xdf.to_numpy(dtype=float)

    q01 = np.nanquantile(X, 0.01, axis=0)
    med = np.nanmedian(X, axis=0)
    q99 = np.nanquantile(X, 0.99, axis=0)

    for i, c in enumerate(feature_names):
        lo, m, hi = float(q01[i]), float(med[i]), float(q99[i])
        kind = _infer_kind(X[:, i], c)
        mn, mx, kind_hint = _domain_bounds(c)
        if kind_hint == "int" and kind != "binary":
            kind = "int"
        rng = hi - lo
        if not np.isfinite(rng) or rng <= 0:
            lo, hi = m - 1.0, m + 1.0
        else:
            lo = lo - 0.05*rng
            hi = hi + 0.05*rng
        if mn is not None: lo = max(lo, mn)
        if mx is not None: hi = min(hi, mx)
        if kind == "binary":
            defaults[c] = 1.0 if m >= 0.5 else 0.0
            specs[c]    = {"kind":"binary","min":0.0,"median":defaults[c],"max":1.0}
        elif kind == "int":
            lo_i = int(np.floor(lo)); hi_i = int(np.ceil(hi)); 
            if lo_i >= hi_i: hi_i = lo_i + 1
            m_i  = int(np.clip(np.round(m), lo_i, hi_i))
            defaults[c] = float(m_i)
            specs[c]    = {"kind":"int","min":lo_i,"median":m_i,"max":hi_i}
        else:
            m_f = float(np.clip(m, lo, hi))
            defaults[c] = m_f
            specs[c]    = {"kind":"float","min":float(lo),"median":m_f,"max":float(hi)}
    return defaults, specs

# ======================= Tabs =======================
tab_dl, tab_eda, tab_clean, tab_train, tab_status, tab_predict = st.tabs([
    "📥 Download Data", "📊 Run EDA", "🧼 Transform", "🧠 Train", "📈 Model Status", "🎯 Predict"
])

# ---- Download ----
with tab_dl:
    st.subheader("📥 Get Ames Housing dataset")
    use_online = st.checkbox(
        "Use online OpenML download", value=False,
        help="When ON, try fetching from OpenML with retries. When OFF, use the committed CSV or your upload."
    )
    up = st.file_uploader("Upload CSV (optional, overrides local file)", type=["csv"], key="raw_upload")
    if up:
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_up = pd.read_csv(up)
        df_up.to_csv(RAW_PATH, index=False)
        st.success(f"✅ Uploaded and saved to {RAW_PATH}")
        st.dataframe(df_up.head())

    if st.button("Load dataset", type="primary"):
        try:
            if use_online:
                with st.spinner("📡 Downloading from OpenML (with retries)…"):
                    csv_path = run_download(RAW_PATH, retries=4, pause=2.5)
                    st.success(f"✅ Downloaded to {csv_path}")
                    df = load_local_csv(csv_path)
                    st.dataframe(df.head())
            else:
                if not RAW_PATH.exists() or RAW_PATH.stat().st_size == 0:
                    st.error(f"❌ No local file found at {RAW_PATH}. Upload a CSV or enable online download.")
                else:
                    df = load_local_csv(RAW_PATH)
                    st.success(f"✅ Loaded local CSV: {RAW_PATH}")
                    st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Could not load dataset: {e}")

# ---- EDA ----
with tab_eda:
    st.subheader("🔍 Exploratory Data Analysis")
    if not RAW_PATH.exists():
        st.warning("❗ Raw data not found. Please download the dataset first.")
    else:
        df = pd.read_csv(RAW_PATH)
        st.success(f"✅ Loaded raw data with shape {df.shape}")
        if st.button("Run EDA", type="primary"):
            run_visual_eda(df, target="Sale_Price")

# ---- Transform ----
with tab_clean:
    st.subheader("🧼 Transform raw → cleaned")
    if st.button("Run transformation"):
        try:
            csv_path = run_transform(RAW_PATH, CLEAN_PATH)
            st.success(f"✅ Cleaned dataset saved to: {csv_path}")
            st.dataframe(pd.read_csv(csv_path, nrows=200))
        except Exception as e:
            st.error(f"❌ Transform failed: {e}")

# ---- Train ----
with tab_train:
    st.subheader("🏋️ Train models")
    if st.button("Start training", type="primary"):
        with st.spinner("Training models…"):
            try:
                result = run_train(CLEAN_PATH)   # use unified clean path
                st.success(f"✅ Trained. Best model: {result['selected_model']}")

                # force cache bust for status/predict tabs
                st.session_state["model_updated_at"] = time.time()

                # Show metrics nicely
                with open(MODEL_DIR / "metrics_summary.json", "r", encoding="utf-8") as f:
                    summary = json.load(f)

                best = summary.get("selected_model", "")
                metrics = summary.get("metrics", {})
                if isinstance(metrics, dict):
                    df_metrics = (
                        pd.DataFrame(metrics).T.reset_index()
                        .rename(columns={"index": "Model", "mae": "Log MAE", "real_mae": "MAE (real)", "r2": "R²"})
                        .loc[:, ["Model", "R²", "MAE (real)", "Log MAE"]]
                        .sort_values("R²", ascending=False)
                        .reset_index(drop=True)
                    )
                else:
                    df_metrics = pd.DataFrame(columns=["Model", "R²", "MAE (real)", "Log MAE"])

                st.markdown("### 📊 Model Metrics")
                st.caption("ℹ️ **How to read this:** R² → higher is better.  MAE → lower is better.")
                
                # Dynamic formatting map (only apply if col exists)
                fmt_map = {
                    "R²": "{:.3f}",
                    "MAE (real)": "{:.0f}",
                    "Log MAE": "{:.3f}",
                }
                fmt_map = {k: v for k, v in fmt_map.items() if k in df_metrics.columns}

                # Decide which cols to highlight dynamically
                hi_max = [c for c in ["R²"] if c in df_metrics.columns]
                hi_min = [c for c in ["MAE (real)", "Log MAE"] if c in df_metrics.columns]

                sty = df_metrics.style.format(fmt_map)
                if hi_max:
                    sty = sty.highlight_max(subset=hi_max, color="#e6ffe6")
                if hi_min:
                    sty = sty.highlight_min(subset=hi_min, color="#e6ffe6")

                st.dataframe(sty, use_container_width=True)

                # Optional plots
                r2_png  = MODEL_DIR / "plots" / "r2_scores.png"
                mae_png = MODEL_DIR / "plots" / "real_mae_scores.png"
                if r2_png.exists() or mae_png.exists():
                    st.markdown("### 📈 Plots")
                    cols = st.columns(2)
                    if r2_png.exists():
                        cols[0].image(str(r2_png), caption="R² Score by Model", use_container_width=True)
                    if mae_png.exists():
                        cols[1].image(str(mae_png), caption="Real MAE by Model", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

# ---- Status ----
with tab_status:
    # --- Load metrics_summary.json ---
    metrics_path = MODEL_DIR / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}

    best = summary.get("selected_model", "")
    metrics = summary.get("metrics", {})

    if isinstance(metrics, dict) and metrics:
        df_metrics = (
            pd.DataFrame(metrics).T.reset_index()
            .rename(columns={
                "index": "Model",
                "mae": "Log MAE",
                "real_mae": "MAE (real)",
                "r2": "R²"
            })
            .loc[:, ["Model", "R²", "MAE (real)", "Log MAE"]]
            .sort_values("R²", ascending=False)
            .reset_index(drop=True)
        )
    else:
        df_metrics = pd.DataFrame(columns=["Model", "R²", "MAE (real)", "Log MAE"])

    st.markdown("### 📊 Model Metrics")
    st.caption("ℹ️ **How to read this:** R² → higher is better.  MAE → lower is better.")

    # Dynamic formatting map (only apply if col exists)
    fmt_map = {
        "R²": "{:.3f}",
        "MAE (real)": "{:.0f}",
        "Log MAE": "{:.3f}",
    }
    fmt_map = {k: v for k, v in fmt_map.items() if k in df_metrics.columns}

    # Decide which cols to highlight dynamically
    hi_max = [c for c in ["R²"] if c in df_metrics.columns]
    hi_min = [c for c in ["MAE (real)", "Log MAE"] if c in df_metrics.columns]

    sty = df_metrics.style.format(fmt_map)
    if hi_max:
        sty = sty.highlight_max(subset=hi_max, color="#e6ffe6")
    if hi_min:
        sty = sty.highlight_min(subset=hi_min, color="#e6ffe6")

    st.dataframe(sty, use_container_width=True)


# ---- Predict ----
with tab_predict:
    st.subheader("🎯 Predict from top‑K most important features")

    cache_bust = st.session_state.get("model_updated_at", 0.0)
    mtime = _summary_mtime(MODEL_DIR)
    model, feature_names, summary, _ = load_best_model(MODEL_DIR, mtime, cache_bust)

    if model is None or not feature_names:
        st.warning("Train a model first so we know the required features.")
    else:
        rank_list, imp_df = get_feature_ranking(model, feature_names, CLEAN_PATH)
        if not rank_list:
            rank_list = feature_names

        if CLEAN_PATH.exists():
            defaults, specs = build_defaults_and_ui_specs_from_csv(CLEAN_PATH, feature_names)
            st.caption(f"Prefilled from medians in **{CLEAN_PATH}**. Binary features → checkbox; integers → integer slider.")
        else:
            defaults = {c: 0.0 for c in feature_names}
            specs = {c: {"kind":"float","min":0.0,"median":0.0,"max":1.0} for c in feature_names}
            st.caption("Prefill fallback: reference file not found, using 0.0 defaults.")

        k = st.slider(
            "How many top features to edit?",
            min_value=4, max_value=min(48, len(rank_list)), value=min(20, len(rank_list)), step=4
        )
        st.caption("Only the top‑K features are shown below. Others are held at dataset medians.")

        GRID_COLS = 4
        cols = st.columns(GRID_COLS, gap="small")
        user_values = {}

        for i, feat in enumerate(rank_list[:k]):
            s = specs.get(feat, {"kind": "float", "min": 0.0, "median": defaults.get(feat, 0.0), "max": 1.0})
            kind = s["kind"]
            lo, mid, hi = s.get("min", 0.0), s.get("median", defaults.get(feat, 0.0)), s.get("max", 1.0)

            with cols[i % GRID_COLS]:
                st.markdown(
                    f"<div style='font-size:0.85rem; font-weight:500; margin-bottom:0.25rem;'>{feat}</div>",
                    unsafe_allow_html=True
                )

                if kind == "binary":
                    init_bool = bool(int(defaults.get(feat, 0.0)))
                    val = st.toggle(label=feat, value=init_bool, key=f"feat_{feat}", label_visibility="collapsed")
                    user_values[feat] = 1.0 if val else 0.0
                elif kind == "int":
                    val = st.number_input(
                        label=feat, min_value=int(lo),
                        max_value=int(hi) if int(hi) > int(lo) else None,
                        value=int(mid), step=1, key=f"feat_{feat}", label_visibility="collapsed"
                    )
                    user_values[feat] = float(int(val))
                else:
                    span = float(hi) - float(lo)
                    step = 0.1 if span <= 10 else 0.5 if span <= 100 else 1.0
                    val = st.number_input(
                        label=feat, min_value=float(lo),
                        max_value=float(hi) if float(hi) > float(lo) else None,
                        value=float(mid), step=step, key=f"feat_{feat}",
                        label_visibility="collapsed", format="%.3f"
                    )
                    user_values[feat] = float(val)

        full_vec = {c: float(defaults.get(c, 0.0)) for c in feature_names}
        for kcol, v in user_values.items():
            full_vec[kcol] = float(v)

        if st.button("Predict price", type="primary"):
            X_df = pd.DataFrame([full_vec])[feature_names]
            yhat = float(model.predict(X_df.to_numpy(dtype=float))[0])
            try:
                price = float(np.expm1(yhat))
                show_price = price if np.isfinite(price) and price > 0 else yhat
            except Exception:
                show_price = yhat

            st.success(f"Predicted SalePrice: **${show_price:,.0f}**")

            sel = summary.get("selected_model")
            real_mae = summary.get("metrics", {}).get(sel, {}).get("real_mae", None)
            if real_mae is not None and np.isfinite(real_mae):
                lo = max(0.0, show_price - real_mae)
                hi = show_price + real_mae
                st.markdown(
                    f"<p style='font-size:1rem; color:gray;'>Expected range (±MAE): ${lo:,.0f} – ${hi:,.0f}</p>",
                    unsafe_allow_html=True
                )
