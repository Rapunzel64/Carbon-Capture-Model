"""
Carbon CO2 Emission Prediction - Model Training Script
======================================================
Run this script ONCE to train models and save the .pkl file.
Then run the Streamlit app: streamlit run app.py

Dataset features:
  Country, Energy_type, Year, Energy_consumption, Energy_production,
  GDP, Population, Energy_intensity_per_capita, Energy_intensity_by_GDP
Target: CO2_emission
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

# ─────────────────────────── CONFIG ──────────────────────────────────────────
DATA_PATH  = "carbon_.csv"           # put your CSV here
MODEL_PATH = "carbon_capture_model.pkl"
TARGET     = "CO2_emission"
TEST_SIZE  = 0.20
RANDOM_STATE = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_and_preprocess(path: str):
    """Load CSV and return cleaned DataFrame."""
    df = pd.read_csv(path)

    # Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET])

    # Fill numeric NaNs with column median
    numeric_cols = [
        "Energy_consumption", "Energy_production", "GDP",
        "Population", "Energy_intensity_per_capita", "Energy_intensity_by_GDP",
    ]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    print(f"✔ Data loaded  →  {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df, numeric_cols


def encode_categoricals(df: pd.DataFrame):
    """Label-encode Country and Energy_type; return encoders."""
    le_country = LabelEncoder()
    le_energy  = LabelEncoder()
    df = df.copy()
    df["Country_enc"]    = le_country.fit_transform(df["Country"].astype(str))
    df["EnergyType_enc"] = le_energy.fit_transform(df["Energy_type"].astype(str))
    return df, le_country, le_energy


def evaluate(model, X_test, y_test) -> dict:
    """Return a dict of evaluation metrics."""
    preds = model.predict(X_test)
    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    # Accuracy proxy: % samples within 10 % of true value
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(y_test != 0, np.abs(preds - y_test) / np.abs(y_test), 0)
    within_10pct = float(np.mean(rel_err < 0.10) * 100)
    return {
        "R2":           round(r2, 4),
        "RMSE":         round(rmse, 2),
        "MAE":          round(mae, 2),
        "Within_10pct": round(within_10pct, 2),
    }


def train_all_models(X_train, y_train):
    """Define, train, and return all models."""
    models = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Ridge Regression": Ridge(alpha=10.0),
        "Linear Regression": LinearRegression(),
    }
    for name, model in models.items():
        print(f"  ⏳ Training {name} …", end="", flush=True)
        model.fit(X_train, y_train)
        print("  done")
    return models


def print_results_table(results: dict):
    """Pretty-print evaluation metrics for all models."""
    header = f"\n{'Model':<25} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'Within 10%':>12}"
    print(header)
    print("─" * len(header))
    for name, m in results.items():
        print(
            f"{name:<25} {m['R2']:>8.4f} {m['RMSE']:>10.2f} "
            f"{m['MAE']:>10.2f} {m['Within_10pct']:>11.2f}%"
        )


def main():
    print("\n" + "=" * 60)
    print("  Carbon CO2 Emission  –  Model Training")
    print("=" * 60 + "\n")

    # 1. Load & preprocess
    df, num_cols = load_and_preprocess(DATA_PATH)
    df, le_country, le_energy = encode_categoricals(df)

    FEATURES = [
        "Year", "Country_enc", "EnergyType_enc",
        "Energy_consumption", "Energy_production", "GDP",
        "Population", "Energy_intensity_per_capita", "Energy_intensity_by_GDP",
    ]

    X = df[FEATURES]
    y = df[TARGET]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"✔ Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # 3. Train all models
    print("Training models:")
    models = train_all_models(X_train, y_train)

    # 4. Evaluate
    results = {}
    for name, model in models.items():
        results[name] = evaluate(model, X_test, y_test)

    print_results_table(results)

    # 5. Pick best by R²
    best_name = max(results, key=lambda n: results[n]["R2"])
    print(f"\n🏆 Best model: {best_name}  (R² = {results[best_name]['R2']})\n")

    # 6. Feature importance (tree models only)
    best_model = models[best_name]
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("Feature Importances:")
        for feat, imp in fi.items():
            bar = "█" * int(imp * 60)
            print(f"  {feat:<35} {bar}  {imp:.4f}")
        print()

    # 7. Save bundle
    col_stats = {
        c: {
            "min":  float(df[c].min()),
            "max":  float(df[c].max()),
            "mean": float(df[c].mean()),
        }
        for c in num_cols
    }

    bundle = {
        "model":           best_model,
        "all_models":      models,
        "le_country":      le_country,
        "le_energy":       le_energy,
        "features":        FEATURES,
        "results":         results,
        "best_model_name": best_name,
        "countries":       sorted(df["Country"].unique().tolist()),
        "energy_types":    sorted(df["Energy_type"].unique().tolist()),
        "year_range":      (int(df["Year"].min()), int(df["Year"].max())),
        "col_stats":       col_stats,
        "target":          TARGET,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"✔ Model bundle saved to '{MODEL_PATH}'  ({size_kb:.1f} KB)")
    print("\nNow run the app:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
