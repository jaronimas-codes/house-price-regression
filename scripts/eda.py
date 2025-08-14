import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

from typing import Optional

# ---------------------------- Helpers ----------------------------

def save_fig(fig, filename):
    output_path = os.path.join("artifacts", "eda", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return output_path

def show_plot_grid(plots):
    rows = [plots[i:i + 2] for i in range(0, len(plots), 2)]
    for row in rows:
        cols = st.columns(2)
        for col, (title, explanation, fig, filename) in zip(cols, row):
            with col:
                col.markdown(f"#### {title}")
                col.markdown(explanation)
                save_fig(fig, filename)
                col.pyplot(fig)
                plt.close(fig)

# ---------------------------- EDA function ----------------------------

def run_visual_eda(df: pd.DataFrame, target: Optional[str] = None):
    st.subheader("📊 EDA Summary & Visualizations")
    st.markdown("---")
    st.markdown("#### Dataset Overview")
    st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    st.write(f"Memory usage: {round(df.memory_usage(deep=True).sum() / 1024**2, 2)} MB")
    st.dataframe(df.head())

    plots = []

    # --- Missing values ---
    miss_df = df.isnull().sum()
    miss_df = miss_df[miss_df > 0].sort_values(ascending=False)
    if not miss_df.empty:
        miss_data = pd.DataFrame({"feature": miss_df.index, "missing": miss_df.values})
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=miss_data, x="missing", y="feature", hue="feature", palette="rocket", legend=False, ax=ax, dodge=False)
        ax.set_title("Missing Values")
        ax.set_xlabel("Count")
        ax.tick_params(axis='y', labelsize=8)
        plots.append((
            "Missing Values",
            "**Why it matters:** Missing values can cause model errors or bias.\n**Next step:** Impute or drop.",
            fig,
            "missing_values.png"
        ))

    # --- Target distribution ---
    if target and target in df.columns:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df[target].dropna(), bins=50, kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribution: {target}")
        ax.set_xlabel(target)
        ax.tick_params(axis='x', labelsize=8)
        plots.append((
            f"Target: {target}",
            "**Why it matters:** Understanding the target guides modeling and metric choices.\n**Next step:** Consider log-transforming if skewed.",
            fig,
            "target_distribution.png"
        ))

    # --- Skewness ---
    numeric_cols = df.select_dtypes(include=np.number).columns
    skewed = df[numeric_cols].skew(numeric_only=True).sort_values(key=abs, ascending=False)
    skewed = skewed[abs(skewed) > 1]
    if not skewed.empty:
        skew_df = pd.DataFrame({"feature": skewed.index, "skew": skewed.values})
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=skew_df, x="skew", y="feature", hue="feature", palette="mako", legend=False, ax=ax, dodge=False)
        ax.set_title("Skewed Numeric Features")
        ax.set_xlabel("Skewness")
        ax.tick_params(axis='y', labelsize=8)
        plots.append((
            "Skewed Features",
            "**Why it matters:** Skewed features can distort model training.\n**Next step:** Apply transformations like log1p or Box-Cox.",
            fig,
            "skewed_features.png"
        ))

    # --- Correlations with target ---
    corr = df[numeric_cols].corr()
    if target and target in corr.columns:
        corr_target = corr[target].drop(target).sort_values(ascending=False)
        top_corr = pd.DataFrame({"feature": corr_target.index, "corr": corr_target.values})
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=top_corr, x="corr", y="feature", hue="feature", palette="viridis", legend=False, ax=ax, dodge=False)
        ax.set_title(f"Correlation with {target}")
        ax.set_xlabel("Correlation")
        ax.tick_params(axis='y', labelsize=8)
        plots.append((
            "Top Correlated Features",
            "**Why it matters:** Highly predictive features are valuable.\n**Next step:** Monitor for multicollinearity.",
            fig,
            "correlated_features.png"
        ))

    # --- Correlation matrix ---
    top_corr_cols = corr.abs().sum().sort_values(ascending=False).head(15).index
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[top_corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=False)
    ax.set_title("Top Feature Correlations")
    plots.append((
        "Correlation Matrix",
        "**Why it matters:** Highly interrelated features may introduce redundancy.\n**Next step:** Drop, combine or use PCA.",
        fig,
        "correlation_matrix.png"
    ))

    # --- Categorical cardinality ---
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols):
        nunique = df[cat_cols].nunique().sort_values(ascending=False)
        cat_df = pd.DataFrame({"feature": nunique.index, "unique": nunique.values})
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=cat_df, x="unique", y="feature", hue="feature", palette="Set2", legend=False, ax=ax, dodge=False)
        ax.set_title("Categorical Cardinality")
        ax.set_xlabel("Unique Values")
        ax.tick_params(axis='y', labelsize=8)
        plots.append((
            "Categorical Features",
            "**Why it matters:** High-cardinality categoricals can overfit.\n**Next step:** Encode, reduce, or drop.",
            fig,
            "categorical_cardinality.png"
        ))

    # --- Show plots in 2-column grid ---
    show_plot_grid(plots)

    st.markdown("---")
    st.success("✅ EDA completed!")
