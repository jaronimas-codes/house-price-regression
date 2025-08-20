# app.py
import os, sys, json, subprocess
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scripts.eda import run_visual_eda
from scripts.download import run_download
from scripts.transform import run_transform

import time
from pathlib import Path

def file_mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0

st.set_page_config(page_title="🏡 House Price ML Suite", layout="wide")

RAW_DEFAULT   = "data/raw/ames_openml.csv"
CLEAN_DEFAULT = "data/clean/ames_clean.csv"
ARTIFACTS     = "artifacts"
MODELS_DIR    = os.path.join(ARTIFACTS, "models")

# ---------------------- utilities ----------------------
def exists(p): return os.path.exists(p)

@st.cache_data(show_spinner=False)
def load_best_model(model_dir: str, cache_bust: float):
    """
    Load best model + features + metrics. Cache is invalidated when any of the
    key files change (via their mtimes) or when cache_bust token changes.
    """
    best_path   = os.path.join(model_dir, "best_model.pkl")
    metrics_path = os.path.join(model_dir, "metrics_summary.json")
    features_path = os.path.join(model_dir, "feature_names.json")

    # include file mtimes in cache key
    mtimes = (
        file_mtime(best_path),
        file_mtime(metrics_path),
        file_mtime(features_path),
        cache_bust,  # session-driven invalidator
    )

    # read files fresh
    summary = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    if not os.path.exists(best_path):
        return None, [], summary, mtimes

    try:
        model = joblib.load(best_path)
    except Exception:
        model = None

    features = []
    for p in (features_path, os.path.join(model_dir, summary.get("selected_model",""), "feature_names.json")):
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    features = json.load(f)
                break
            except Exception:
                pass

    return model, features, summary, mtimes

def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def call_script(cmd_list):
    with st.spinner("Running..."):
        proc = subprocess.run(cmd_list, capture_output=True, text=True)
        if proc.stdout:
            st.code(proc.stdout)
        if proc.stderr:
            st.error(proc.stderr)
        return proc.returncode == 0

@st.cache_data(show_spinner=False)
def load_best_model(model_dir: str = MODELS_DIR, cache_bust: float = 0.0):
    """Load best model + features + metrics with cache-busting."""
    best_path     = os.path.join(model_dir, "best_model.pkl")
    metrics_path  = os.path.join(model_dir, "metrics_summary.json")
    features_path = os.path.join(model_dir, "feature_names.json")

    # include file mtimes in cache key
    _cache_key = (
        file_mtime(best_path),
        file_mtime(metrics_path),
        file_mtime(features_path),
        cache_bust,
    )

    # read metrics
    summary = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    if not os.path.exists(best_path):
        return None, [], summary, _cache_key

    try:
        model = joblib.load(best_path)
    except Exception:
        model = None

    # feature names
    features = []
    for p in (features_path, os.path.join(model_dir, summary.get("selected_model",""), "feature_names.json")):
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    features = json.load(f)
                break
            except Exception:
                pass

    return model, features, summary, _cache_key



def get_feature_ranking(model, feature_names, cleaned_sample_path=CLEAN_DEFAULT, top_k=25):
    summary = read_json(os.path.join(MODELS_DIR, "metrics_summary.json"), {})
    selected = summary.get("selected_model")
    if selected:
        csv_path = os.path.join(MODELS_DIR, selected, "feature_importances.csv")
        if exists(csv_path):
            imp = pd.read_csv(csv_path)
            if {"feature","importance"}.issubset(imp.columns):
                imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
                return imp["feature"].tolist(), imp
    if hasattr(model, "coef_") and model.coef_ is not None:
        coef = np.asarray(model.coef_).ravel()
        imp = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef)})
        imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
        return imp["feature"].tolist(), imp
    if exists(cleaned_sample_path) and feature_names:
        ref = pd.read_csv(cleaned_sample_path)
        if set(feature_names).issubset(ref.columns):
            X_df = ref[feature_names].copy()
        else:
            X_df = ref.select_dtypes(include=np.number).copy()
            for col in feature_names:
                if col not in X_df.columns:
                    X_df[col] = 0.0
            X_df = X_df[feature_names]
        X_df = X_df.fillna(0)
        tgt_candidates = ["SalePrice","Sale_Price","saleprice","sale_price","PRICE","price"]
        tgt = next((c for c in tgt_candidates if c in ref.columns), None)
        if tgt:
            y = pd.to_numeric(ref[tgt], errors="coerce").fillna(0).to_numpy()
        else:
            y = np.zeros(len(X_df))
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

def resolve_model_and_feature_names(model, fallback_names):
    """
    Return (final_estimator, feature_names_after_preprocessing).
    Works for bare estimators and sklearn Pipelines with 'preprocessor' and 'model' steps.
    """
    final_est = model
    feature_names = list(fallback_names or [])

    # If it's a Pipeline, try to pull final estimator and transformed feature names
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        # Prefer a step literally named 'model'; else take the last step
        if "model" in steps:
            final_est = steps["model"]
        else:
            # last step value
            final_est = list(steps.values())[-1]

        # Get names after preprocessing if available
        pre = steps.get("preprocessor")
        if pre is not None:
            try:
                feature_names = list(pre.get_feature_names_out())
            except Exception:
                pass

    # Fallback to numbered names if still empty
    if not feature_names:
        n = getattr(final_est, "n_features_in_", None)
        if isinstance(n, int) and n > 0:
            feature_names = [f"f{i}" for i in range(n)]

    return final_est, feature_names

# ---------------------- main layout ----------------------
st.title("🏡 House Prices — Full ML Workflow")

tab_dl, tab_eda, tab_clean, tab_train, tab_status, tab_predict = st.tabs([
    "📥 Download Data", "📊 Run EDA", "🧼 Transform", "🧠 Train", "📈 Model Status", "🎯 Predict"])

RAW_PATH = Path("data/raw/ames_openml.csv")

with tab_dl:
    st.subheader("📥 Download Ames Housing Dataset")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Download from OpenML", type="primary"):
            with st.spinner("📡 Downloading from OpenML (with retries)…"):
                try:
                    csv_path = run_download(RAW_PATH, retries=4, pause=2.5)
                    st.success(f"✅ Downloaded to {csv_path}")
                    st.dataframe(pd.read_csv(csv_path).head())
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")

    with col2:
        # Manual fallback: upload a CSV
        up = st.file_uploader("Or upload ames_openml.csv", type=["csv"])
        if up:
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(up)
            df.to_csv(RAW_PATH, index=False)
            st.success(f"✅ Uploaded and saved to {RAW_PATH}")
            st.dataframe(df.head())

    # Optional: user-provided URL fallback
    with st.expander("Paste a direct CSV URL (fallback)"):
        url = st.text_input("Direct CSV URL")
        if st.button("Fetch from URL") and url:
            try:
                df = pd.read_csv(url)
                RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(RAW_PATH, index=False)
                st.success(f"✅ Downloaded from URL and saved to {RAW_PATH}")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Could not load CSV from URL: {e}")



with tab_eda:
    st.subheader("🔍 Exploratory Data Analysis")

    if not os.path.exists(RAW_DEFAULT):
        st.warning("❗ Raw data not found. Please download the dataset first.")
    else:
        df = pd.read_csv(RAW_DEFAULT)
        st.success(f"✅ Loaded raw data with shape {df.shape}")
        
        if st.button("Run EDA", type="primary"):
            run_visual_eda(df, target="Sale_Price")

with tab_clean:
    st.subheader("Transform raw → cleaned")
    if st.button("Run transformation script"):
        try:
            csv_path = run_transform(Path("data/raw/ames_openml.csv"), Path("data/processed/ames_cleaned.csv"))
            st.success(f"✅ Cleaned dataset saved to: {csv_path}")
            st.dataframe(pd.read_csv(csv_path, nrows=200))
        except Exception as e:
            st.error(f"❌ Transform failed: {e}")

with tab_train:
    st.subheader("Train model (Linear Reg, RandomForest, XGB)")
    if st.button("Train models now"):
        ok = call_script(["python", "scripts/train.py"])
        if ok:
            st.success("✅ Training completed.")
            # mark update time and force a rerun so status tab sees new files
            st.session_state["model_updated_at"] = time.time()
            # st.rerun()

with tab_status:
    st.subheader("Model status")

    # token that changes after training (and also if user clicks refresh)
    cache_bust = st.session_state.get("model_updated_at", 0.0)

    colA, colB = st.columns([1,1])
    with colB:
        if st.button("🔄 Refresh status"):
            st.session_state["model_updated_at"] = time.time()
            st.rerun()

    model, feature_names, summary, _ = load_best_model(MODELS_DIR, cache_bust)


    if model is None:
        st.info("No trained model found. Click **Train** in the training tab.")
    else:
        st.success(f"Loaded best model: **{summary.get('selected_model', '(unknown)')}**")

        # Metrics table
        allm = summary.get("metrics", {})
        if allm:
            dfm = pd.DataFrame(allm).T.rename(columns={"rmse": "RMSE", "r2": "R²", "mae": "MAE"})
            st.write("Per-model metrics (lower RMSE is better):")
            st.dataframe(
                dfm.style.format({"RMSE": "{:.0f}", "MAE": "{:.0f}", "R²": "{:.4f}"}),
                use_container_width=True
            )

        # --- Feature Importance (robust) ---
        final_est, post_names = resolve_model_and_feature_names(model, feature_names)

        importances = None
        if hasattr(final_est, "feature_importances_"):
            importances = np.asarray(final_est.feature_importances_)
        elif hasattr(final_est, "coef_") and final_est.coef_ is not None:
            coef = np.asarray(final_est.coef_)
            importances = np.abs(coef.ravel() if coef.ndim > 1 else coef)

        if importances is not None and len(importances) > 0:
            names = list(post_names)
            n = min(len(names), len(importances))
            if n == 0:
                st.info("Model exposes importances, but feature names are unavailable.")
            else:
                fi_df = pd.DataFrame({
                    "Feature": names[:n],
                    "Importance": importances[:n]
                }).sort_values("Importance", ascending=False).head(20)

                st.markdown("### 🔍 Feature Importance")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(
                    data=fi_df,
                    x="Importance",
                    y="Feature",
                    hue="Feature",
                    dodge=False,
                    legend=False,
                    ax=ax
                )
                ax.set_title("Top 20 Feature Importances")
                st.pyplot(fig)
        else:
            st.info("This model doesn’t expose feature importances/coefficients. Try RandomForest/XGB or use permutation importances.")
            


from datetime import datetime

# ---------- Prefill + UI typing from a specific CSV (model feature space) ----------
PREFERRED_REF = "data/processed/ames_cleaned.csv"
POSSIBLE_TARGETS_ALL = ["Log_SalePrice","SalePrice","Sale_Price","saleprice","sale_price","PRICE","price"]

# name hints for integers
_INT_NAME_HINTS = (
    "year","yr","garagecars","garage_cars","fullbath","full_bath","halfbath","half_bath",
    "kitchen","bedroom","bedrooms","totrms","rooms","bsmt_full_bath","bsmt_half_bath",
    "fireplaces","mosold","mo_sold","month","mo","day"
)

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

def _infer_kind(col_values: np.ndarray, name: str) -> str:
    """Return 'binary' | 'int' | 'float' based on values (w/ name hints)."""
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
    """
    Return (min, max, kind_hint) by feature name.
    kind_hint can tighten type to 'int' for year/month/day even if values looked floaty.
    """
    n = name.lower()
    cur_year = datetime.now().year

    # counts (never negative)
    if any(k in n for k in [
        "fireplace","fullbath","full_bath","halfbath","half_bath",
        "bsmt_full_bath","bsmt_half_bath","bedroom","bedrooms",
        "kitchen","totrms","rooms","garagecars","garage_cars"
    ]):
        return 0, None, "int"

    # areas/square footage (never negative)
    if any(k in n for k in [
        "sf","area","porch","deck","wooddeck","openporch","enclosedporch","screenporch",
        "3ssnporch","lotfrontage","lot_frontage","masvnrarea","poolarea","miscval"
    ]):
        return 0.0, None, "float"

    # years
    if "year" in n or n.endswith("_yr_blt") or n.endswith("yrblt") or "yr" in n:
        return 1870, cur_year + 1, "int"

    # months/days
    if "mosold" in n or "month" in n or n.startswith("mo_") or n.endswith("_mo"):
        return 1, 12, "int"
    if "day" in n:
        return 1, 31, "int"

    return None, None, None

def build_defaults_and_ui_specs_from_csv(ref_path: str, feature_names: list[str]) -> tuple[dict, dict]:
    """
    Returns:
      defaults: {feat -> numeric default (median or 0/1)}
      ui_specs: {feat -> {"kind": "binary|int|float", "min": ..., "median": ..., "max": ...}}
    """
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

        # infer type from data + name
        kind = _infer_kind(X[:, i], c)
        mn, mx, kind_hint = _domain_bounds(c)
        if kind_hint == "int" and kind != "binary":
            kind = "int"  # force integers for year/month/day/etc.

        # widen a tiny bit, then clamp with domain rules
        rng = hi - lo
        if not np.isfinite(rng) or rng <= 0:
            lo, hi = m - 1.0, m + 1.0
        else:
            lo = lo - 0.05*rng
            hi = hi + 0.05*rng

        if mn is not None: lo = max(lo, mn)
        if mx is not None: hi = min(hi, mx)

        # finalize per kind
        if kind == "binary":
            defaults[c] = 1.0 if m >= 0.5 else 0.0
            specs[c]    = {"kind":"binary","min":0.0,"median":defaults[c],"max":1.0}
        elif kind == "int":
            lo_i = int(np.floor(lo))
            hi_i = int(np.ceil(hi))
            if lo_i >= hi_i: hi_i = lo_i + 1
            m_i  = int(np.clip(np.round(m), lo_i, hi_i))
            defaults[c] = float(m_i)
            specs[c]    = {"kind":"int","min":lo_i,"median":m_i,"max":hi_i}
        else:  # float
            # ensure median within [lo, hi]
            m_f = float(np.clip(m, lo, hi))
            defaults[c] = m_f
            specs[c]    = {"kind":"float","min":float(lo),"median":m_f,"max":float(hi)}
    return defaults, specs


with tab_predict:
    st.subheader("🎯 Predict from top-K most important features")

    # refresh after training
    cache_bust = st.session_state.get("model_updated_at", 0.0)
    model, feature_names, summary, _ = load_best_model(MODELS_DIR, cache_bust)

    if model is None or not feature_names:
        st.warning("Train a model first so we know the required features.")
    else:
        # Feature ranking (your existing helper)
        rank_list, imp_df = get_feature_ranking(model, feature_names, CLEAN_DEFAULT)
        if not rank_list:
            rank_list = feature_names

        # --------- Prefill from data/processed/ames_cleaned.csv ----------
        if os.path.exists(PREFERRED_REF):
            defaults, specs = build_defaults_and_ui_specs_from_csv(PREFERRED_REF, feature_names)
            st.caption(f"Prefilled from medians in **{PREFERRED_REF}**. Binary features → checkbox; integers → integer slider.")
        else:
            # last-resort fallback
            defaults = {c: 0.0 for c in feature_names}
            specs = {c: {"kind":"float","min":0.0,"median":0.0,"max":1.0} for c in feature_names}
            st.caption("Prefill fallback: reference file not found, using 0.0 defaults.")

        k = st.slider(
                    "How many top features to edit?",
                    min_value=4,
                    max_value=min(48, len(rank_list)),  # keep multiple of 4 so grid fills nicely
                    value=min(20, len(rank_list)),
                    step=4
                    )
                
        
        st.caption("Only the top-K features are shown below. Others are held at dataset medians.")

        # Compact inputs in a 4-column grid (labels collapsed)
        GRID_COLS = 4
        
        # tighter columns
        cols = st.columns(GRID_COLS, gap="small")
        user_values = {}

        for i, feat in enumerate(rank_list[:k]):
            s = specs.get(feat, {"kind": "float", "min": 0.0, "median": defaults.get(feat, 0.0), "max": 1.0})
            kind = s["kind"]
            lo, mid, hi = s.get("min", 0.0), s.get("median", defaults.get(feat, 0.0)), s.get("max", 1.0)

            with cols[i % GRID_COLS]:
                # Visible label (our own, above the widget)
                st.markdown(
                    f"<div style='font-size:0.85rem; font-weight:500; margin-bottom:0.25rem;'>{feat}</div>",
                    unsafe_allow_html=True
                )

                if kind == "binary":
                    init_bool = bool(int(defaults.get(feat, 0.0)))
                    # IMPORTANT: give a real label, but collapse it
                    val = st.toggle(
                        label=feat, value=init_bool, key=f"feat_{feat}",
                        label_visibility="collapsed"
                    )
                    user_values[feat] = 1.0 if val else 0.0

                elif kind == "int":
                    val = st.number_input(
                        label=feat,  # non-empty to satisfy Streamlit
                        min_value=int(lo),
                        max_value=int(hi) if int(hi) > int(lo) else None,
                        value=int(mid),
                        step=1,
                        key=f"feat_{feat}",
                        label_visibility="collapsed"  # keeps UI compact
                    )
                    user_values[feat] = float(int(val))

                else:  # float
                    span = float(hi) - float(lo)
                    step = 0.1 if span <= 10 else 0.5 if span <= 100 else 1.0
                    val = st.number_input(
                        label=feat,  # non-empty label
                        min_value=float(lo),
                        max_value=float(hi) if float(hi) > float(lo) else None,
                        value=float(mid),
                        step=step,
                        key=f"feat_{feat}",
                        label_visibility="collapsed",
                        format="%.3f"
                    )
                    user_values[feat] = float(val)


        # Build full vector: medians for all, override top-K with user inputs
        full_vec = {c: float(defaults.get(c, 0.0)) for c in feature_names}
        for kcol, v in user_values.items():
            full_vec[kcol] = float(v)

        # Predict
        if st.button("Predict price", type="primary"):
            X_df = pd.DataFrame([full_vec])[feature_names]
            yhat = float(model.predict(X_df.to_numpy(dtype=float))[0])
            # If your model predicted log price, convert back
            try:
                price = float(np.expm1(yhat))
                show_price = price if np.isfinite(price) and price > 0 else yhat
            except Exception:
                show_price = yhat

            st.success(f"Predicted SalePrice: **${show_price:,.0f}**")

            # Optional ±MAE range from training summary
            sel = summary.get("selected_model")
            real_mae = summary.get("metrics", {}).get(sel, {}).get("real_mae", None)
            if real_mae is not None and np.isfinite(real_mae):
                lo = max(0.0, show_price - real_mae)
                hi = show_price + real_mae
                st.markdown(
                                f"<p style='font-size:1rem; color:gray;'>"
                                f"Expected range (±MAE): ${lo:,.0f} – ${hi:,.0f}"
                                f"</p>",
                                unsafe_allow_html=True
                            )






