import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- Paths ----------------
DATA_PATH  = "data/processed/ames_cleaned.csv"
TARGET     = "Log_SalePrice"
DROP_COLS  = ["Sale_Price"]
MODEL_DIR  = "artifacts/models"
PLOT_DIR   = os.path.join(MODEL_DIR, "plots")

# ---------------- Utils for animations ----------------
def pick_two_numeric_features(df: pd.DataFrame, target: str, prefer: list[str] | None = None) -> list[str]:
    """
    Choose two strong numeric features for a 2D surface.
    Preference order:
      1) any two from 'prefer' that exist and are numeric
      2) top-2 absolute correlation with target among numeric columns
    """
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c != target]
    if prefer:
        chosen = [c for c in prefer if c in num_cols]
        if len(chosen) >= 2:
            return chosen[:2]
    if target in df.columns and np.issubdtype(df[target].dtype, np.number):
        corr = df[num_cols + [target]].corr(numeric_only=True)[target].abs().drop(labels=[target])
        return corr.sort_values(ascending=False).index[:2].tolist()
    # fallback: first two numeric
    return num_cols[:2]

def make_grid_2d(df_or_X: pd.DataFrame, f1: str, f2: str, steps: int = 110):
    q = df_or_X[[f1, f2]].quantile([0.01, 0.99])
    xs = np.linspace(q.loc[0.01, f1], q.loc[0.99, f1], steps)
    ys = np.linspace(q.loc[0.01, f2], q.loc[0.99, f2], steps)
    XX, YY = np.meshgrid(xs, ys)
    return XX, YY

def build_grid_matrix(feature_order: list[str], fixed_vector: np.ndarray, f1: str, f2: str, XX, YY) -> np.ndarray:
    mat = np.tile(fixed_vector, (XX.size, 1))
    i1, i2 = feature_order.index(f1), feature_order.index(f2)
    mat[:, i1] = XX.ravel()
    mat[:, i2] = YY.ravel()
    return mat

def draw_surface_frame(Z: np.ndarray, XX, YY, f1: str, f2: str, title: str) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(XX, YY, Z.reshape(XX.shape), levels=24, alpha=0.85)
    ax.set_xlabel(f1); ax.set_ylabel(f2); ax.set_title(title)
    fig.tight_layout()

    # --- Backend-agnostic: save to PNG buffer, then read as array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img = imageio.imread(buf)   # shape (H, W, 4) RGBA
    plt.close(fig)

    # Drop alpha if present
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def animate_random_forest(rf_model, X_frame: np.ndarray, XX, YY, f1: str, f2: str,
                          out_path: str, max_frames: int = 60, delog: bool = True):
    imgs = []
    csum = np.zeros(X_frame.shape[0], dtype=float)
    ests = rf_model.estimators_[:max_frames]
    for i, est in enumerate(ests, start=1):
        preds = est.predict(X_frame)
        csum += preds
        avg = csum / i
        if delog:
            avg = np.expm1(avg)
        img = draw_surface_frame(avg, XX, YY, f1, f2, f"Random Forest — {i} trees")
        if i == 1:
            imageio.imwrite(out_path.replace(".gif", "_frame_001.png"), img)
        imgs.append(img)
    imageio.mimsave(out_path, imgs, duration=0.12)

def animate_xgboost(xgb_model, X_frame: np.ndarray, XX, YY, f1: str, f2: str,
                    out_path: str, max_frames: int = 80, delog: bool = True):
    imgs = []
    max_k = min(max_frames, getattr(xgb_model, "n_estimators", 100))
    for k in range(1, max_k + 1):
        # prefer iteration_range; fallback to ntree_limit
        try:
            preds = xgb_model.predict(X_frame, iteration_range=(0, k))
        except TypeError:
            preds = xgb_model.predict(X_frame, ntree_limit=k)
        if delog:
            preds = np.expm1(preds)
        img = draw_surface_frame(preds, XX, YY, f1, f2, f"XGBoost — {k} boosting rounds")
        if k == 1:
            imageio.imwrite(out_path.replace(".gif", "_frame_001.png"), img)
        imgs.append(img)
    imageio.mimsave(out_path, imgs, duration=0.10)

# ---------------- Training Function ----------------
def train_and_evaluate(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    # Inverse log transform
    real_mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds))

    print(f"\n{name} Results:")
    print(f"Log MAE: {mae:.3f}")
    print(f"Real MAE: {real_mae:.0f}")
    print(f"R²:      {r2:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    print(f"Saved: {name}.pkl")

    return model, {"mae": mae, "real_mae": real_mae, "r2": r2}

def plot_model_metrics(results):
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

    # ---- FIX seaborn FutureWarning: use hue and hide legend
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="r2", y="Model", hue="Model", dodge=False, legend=False, ax=ax)
    ax.set_title("R² Score by Model")
    ax.set_xlabel("R²")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "r2_scores.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="real_mae", y="Model", hue="Model", dodge=False, legend=False, ax=ax)
    ax.set_title("Real MAE (De-logged) by Model")
    ax.set_xlabel("MAE")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "real_mae_scores.png"))

def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Cleaned data not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        print(f"[ERROR] Target column '{TARGET}' not found in data")
        return

    # Prepare features and target
    df = df.drop(columns=DROP_COLS, errors="ignore")
    X_orig_numeric = df.select_dtypes(include=np.number).drop(columns=[TARGET], errors="ignore")
    X = df.drop(columns=[TARGET])
    X = pd.get_dummies(X, drop_first=True)  # one-hot encode categoricals
    feature_names = X.columns.tolist()
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",   # fast & CPU-friendly
            eval_metric="rmse"
        ),
    }

    results = {}
    best_r2 = float("-inf")
    best_model = None
    best_name = ""

    for name, model in models.items():
        model_trained, metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
        results[name] = metrics
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_model = model_trained
            best_name = name

    # Save summary (align with app.py expectations)
    summary = {
        "selected_model": best_name,
        "metrics": results,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save feature names for the app
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_model_metrics(results)

    print(f"\nBest model: {best_name}")
    print(f"Saved summary to {os.path.join(MODEL_DIR, 'metrics_summary.json')}")
    print(f"Saved model to {os.path.join(MODEL_DIR, 'best_model.pkl')}")
    print(f"Saved feature names to {os.path.join(MODEL_DIR, 'feature_names.json')}")
    print(f"Plots & animations saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
