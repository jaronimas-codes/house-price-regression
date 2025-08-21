# scripts/train.py
from __future__ import annotations
from pathlib import Path
import os, io, json
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

# ---------------- Paths (resolve from repo root) ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # <repo>/
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "ames_cleaned.csv"
MODEL_DIR    = PROJECT_ROOT / "artifacts" / "models"
PLOT_DIR     = MODEL_DIR / "plots"

TARGET    = "Log_SalePrice"
DROP_COLS = ["Sale_Price"]

# ---------------- Utils (unchanged) ----------------
def pick_two_numeric_features(df: pd.DataFrame, target: str, prefer: list[str] | None = None) -> list[str]:
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c != target]
    if prefer:
        chosen = [c for c in prefer if c in num_cols]
        if len(chosen) >= 2:
            return chosen[:2]
    if target in df.columns and np.issubdtype(df[target].dtype, np.number):
        corr = df[num_cols + [target]].corr(numeric_only=True)[target].abs().drop(labels=[target])
        return corr.sort_values(ascending=False).index[:2].tolist()
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
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img = imageio.imread(buf)
    plt.close(fig)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img

def animate_random_forest(rf_model, X_frame: np.ndarray, XX, YY, f1: str, f2: str,
                          out_path: Path, max_frames: int = 60, delog: bool = True):
    out_path = Path(out_path)
    imgs = []
    csum = np.zeros(X_frame.shape[0], dtype=float)
    for i, est in enumerate(rf_model.estimators_[:max_frames], start=1):
        preds = est.predict(X_frame)
        csum += preds
        avg = np.expm1(csum / i) if delog else (csum / i)
        img = draw_surface_frame(avg, XX, YY, f1, f2, f"Random Forest — {i} trees")
        if i == 1:
            imageio.imwrite(str(out_path).replace(".gif", "_frame_001.png"), img)
        imgs.append(img)
    imageio.mimsave(out_path, imgs, duration=0.12)

def animate_xgboost(xgb_model, X_frame: np.ndarray, XX, YY, f1: str, f2: str,
                    out_path: Path, max_frames: int = 80, delog: bool = True):
    out_path = Path(out_path)
    imgs = []
    max_k = min(max_frames, getattr(xgb_model, "n_estimators", 100))
    for k in range(1, max_k + 1):
        try:
            preds = xgb_model.predict(X_frame, iteration_range=(0, k))
        except TypeError:
            preds = xgb_model.predict(X_frame, ntree_limit=k)
        preds = np.expm1(preds) if delog else preds
        img = draw_surface_frame(preds, XX, YY, f1, f2, f"XGBoost — {k} boosting rounds")
        if k == 1:
            imageio.imwrite(str(out_path).replace(".gif", "_frame_001.png"), img)
        imgs.append(img)
    imageio.mimsave(out_path, imgs, duration=0.10)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    real_mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{name}.pkl")
    return model, {"mae": mae, "real_mae": real_mae, "r2": r2}

def plot_model_metrics(results: dict):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="r2", y="Model", hue="Model", dodge=False, legend=False, ax=ax)
    ax.set_title("R² Score by Model"); ax.set_xlabel("R²"); plt.tight_layout()
    fig.savefig(PLOT_DIR / "r2_scores.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="real_mae", y="Model", hue="Model", dodge=False, legend=False, ax=ax)
    ax.set_title("Real MAE (De-logged) by Model"); ax.set_xlabel("MAE"); plt.tight_layout()
    fig.savefig(PLOT_DIR / "real_mae_scores.png")

def run_train(data_path: Path = DATA_PATH) -> dict:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Cleaned data not found at {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")
    df = df.drop(columns=DROP_COLS, errors="ignore")
    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "xgboost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            tree_method="hist", eval_metric="rmse"
        ),
    }

    results = {}
    best = ("", None, -1.0)
    for name, mdl in models.items():
        mdl_trained, metrics = train_and_evaluate(mdl, X_train, y_train, X_test, y_test, name)
        results[name] = metrics
        if metrics["r2"] > best[2]:
            best = (name, mdl_trained, metrics["r2"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best[1], MODEL_DIR / "best_model.pkl")
    with open(MODEL_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({"selected_model": best[0], "metrics": results}, f, indent=2)
    with open(MODEL_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    plot_model_metrics(results)
    return {"selected_model": best[0], "metrics": results, "model_dir": str(MODEL_DIR)}

if __name__ == "__main__":
    out = run_train()
    print(json.dumps(out, indent=2))
