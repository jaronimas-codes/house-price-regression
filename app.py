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

with tab_dl:
    st.subheader("📥 Download Ames Housing Dataset")

    if st.button("Download from OpenML", type="primary"):
        with st.spinner("📡 Downloading and saving dataset..."):
            result = subprocess.run(["python", "scripts/download.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("✅ Dataset downloaded successfully!")
                
                # ✅ Try loading and previewing the saved dataset
                try:
                    df = pd.read_csv("data/raw/ames_openml.csv", low_memory=False)
                    st.markdown("#### 📄 Preview of the dataset:")
                    st.dataframe(df.head())
                except Exception as e:
                    st.warning(f"Download succeeded but couldn't load preview: {e}")
            else:
                st.error("❌ Download failed")
                st.code(result.stderr or "No error message.")



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
        call_script(["python", "scripts/transform.py"])

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

        # if importances is not None and len(importances) > 0:
        #     names = list(post_names)
        #     n = min(len(names), len(importances))
        #     if n == 0:
        #         st.info("Model exposes importances, but feature names are unavailable.")
        #     else:
        #         fi_df = pd.DataFrame({
        #             "Feature": names[:n],
        #             "Importance": importances[:n]
        #         }).sort_values("Importance", ascending=False).head(20)

        #         st.markdown("### 🔍 Feature Importance")
        #         fig, ax = plt.subplots(figsize=(8, 5))
        #         sns.barplot(
        #             data=fi_df,
        #             x="Importance",
        #             y="Feature",
        #             hue="Feature",
        #             dodge=False,
        #             legend=False,
        #             ax=ax
        #         )
        #         ax.set_title("Top 20 Feature Importances")
        #         st.pyplot(fig)
        # else:
        #     st.info("This model doesn’t expose feature importances/coefficients. Try RandomForest/XGB or use permutation importances.")



with tab_predict:
    st.subheader("Predict from top-K most important features")

    cache_bust = st.session_state.get("model_updated_at", 0.0)   # <-- add this
    model, feature_names, summary, _ = load_best_model(MODELS_DIR, cache_bust)  # <-- replace old call

    if model is None or not feature_names:
        st.warning("Train a model first so we know the required features.")
    else:
        rank_list, imp_df = get_feature_ranking(model, feature_names, CLEAN_DEFAULT)
        topN = min(25, len(rank_list))
        show_df = imp_df.head(topN).copy()
        st.markdown("**Top feature importances (sorted):**")
        if "importance" in show_df.columns and show_df["importance"].notna().any():
            st.dataframe(show_df.style.format({"importance": "{:.6f}"}), use_container_width=True)
            st.bar_chart(show_df.set_index("feature")["importance"])
        else:
            st.info("No numeric importances available; using default feature order.")
        st.divider()

        defaults = {}
        if exists(CLEAN_DEFAULT):
            ref = pd.read_csv(CLEAN_DEFAULT, nrows=5000)
            for col in feature_names:
                if col in ref.columns and np.issubdtype(ref[col].dtype, np.number):
                    defaults[col] = float(np.nanmedian(ref[col]))
                else:
                    defaults[col] = 0.0
        else:
            for col in feature_names:
                defaults[col] = 0.0

        k = st.slider("How many top features to edit?", min_value=5, max_value=min(50, len(rank_list)), value=min(20, len(rank_list)))
        st.caption("Only the top‑K important features are shown below. All remaining features use dataset medians.")
        user_values = {}
        cols = st.columns(2)
        for i, col in enumerate(rank_list[:k]):
            with cols[i % 2]:
                user_values[col] = st.number_input(col, value=float(defaults.get(col, 0.0)))

        full_vec = {c: defaults.get(c, 0.0) for c in feature_names}
        for kcol, v in user_values.items():
            full_vec[kcol] = v

        if st.button("Predict price", type="primary"):
            X_df = pd.DataFrame([full_vec])[feature_names]
            pred = float(model.predict(X_df.to_numpy(dtype=float))[0])
            st.success(f"Predicted SalePrice: **${pred:,.0f}**")
