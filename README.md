# 🏡 House Price Regression

A machine learning project to predict housing prices using the [Ames
Housing dataset](https://www.openml.org/d/42165).\
Built with **Python**, **scikit-learn**, **XGBoost**, and **Streamlit**
for an interactive web UI.

---

## 📌 Project Overview

This project demonstrates a full ML workflow:

- **Data preprocessing** (cleaning, feature engineering,
  log-transforms)
- **Exploratory Data Analysis (EDA)** with compact visualizations
- **Feature selection** (handling multicollinearity, ranking important
  features)
- **Model training & evaluation**:
  - Linear Regression
  - Random Forest
  - XGBoost
- **Streamlit app** for interactive predictions:
  - Train models directly in the app
  - Predict sale price from **top-K features**
  - Compact UI with sliders, toggles, and number inputs
  - Shows expected error range (±MAE)

---

## 📂 Repository Structure

    hause-price-regression/
    │
    ├── data/               # raw and processed datasets
    ├── artifacts/          # trained models (ignored in git)
    ├── scripts/            # training, prediction, transform logic
    ├── app.py              # Streamlit app entry point
    ├── train.py            # model training script
    ├── transform.py        # preprocessing and feature engineering
    ├── requirements.txt    # dependencies
    └── README.md           # project description

---

## ⚡ Installation

1.  Clone the repo:

    ```bash
    git clone https://github.com/jaronimas-codes/hause-price-regression.git
    cd hause-price-regression
    ```

2.  Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate    # on Linux/Mac
    venv\Scripts\activate       # on Windows
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

````

### Launch Streamlit app

```bash
streamlit run app.py
````

Then open <http://localhost:8501> in your browser and follow the workflow.

---

## 📊 Features in Streamlit App

- 🎯 **Top-K feature editing**: adjust the most important features\
- 🧮 **Prefilled defaults**: dataset medians (binary → checkbox, int →
  number input, float → number input)\
- 📈 **Prediction with uncertainty**: shows predicted price and ±MAE
  range\
- 🖥️ **Compact UI**: 4-column layout for efficient editing

---

## 🛠 Tech Stack

- Python 3.11
- Pandas, NumPy
- scikit-learn
- XGBoost
- Streamlit
- Matplotlib
