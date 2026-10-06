"""
Module 2 - Demand Forecasting
=============================
Predicts daily order-count demand per product category.

Approach:
- Baseline: 7-day trailing moving average (naive).
- Model: XGBoost regressor on lag + calendar features.
- Time-aware split: train on data before a cutoff date, test strictly after it
  (no shuffling / no random split -> avoids leakage across time).

Run:
    python3 app/forecasting/train_demand_model.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

N_LAGS = 7
MIN_CATEGORY_DAYS = 60  # keep categories with enough history to make forecasting meaningful


def load_series():
    df = pd.read_csv(os.path.join(PROCESSED, "category_daily_demand.csv"), parse_dates=["purchase_date"])
    return df


def make_full_calendar(df):
    """Fill missing days per category with 0 orders so lag features are correct."""
    out = []
    num_cols = ["order_count", "units", "revenue"]
    for cat, g in df.groupby("category"):
        idx = pd.date_range(g["purchase_date"].min(), g["purchase_date"].max(), freq="D")
        g = g.set_index("purchase_date")[num_cols].reindex(idx, fill_value=0)
        g["category"] = str(cat)
        g.index.name = "purchase_date"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def make_features(df):
    df = df.sort_values(["category", "purchase_date"]).copy()
    for lag in range(1, N_LAGS + 1):
        df[f"lag_{lag}"] = df.groupby("category")["order_count"].shift(lag)
    df["rolling_mean_7"] = df.groupby("category")["order_count"].shift(1).rolling(7).mean().reset_index(0, drop=True)
    df["rolling_std_7"] = df.groupby("category")["order_count"].shift(1).rolling(7).std().reset_index(0, drop=True)
    df["dow"] = df["purchase_date"].dt.dayofweek
    df["month"] = df["purchase_date"].dt.month
    df["cat_code"] = df["category"].astype("category").cat.codes
    return df


def train():
    raw = load_series()
    counts_per_cat = raw.groupby("category")["purchase_date"].nunique()
    keep_cats = counts_per_cat[counts_per_cat >= MIN_CATEGORY_DAYS].index
    raw = raw[raw["category"].isin(keep_cats)]

    full = make_full_calendar(raw)
    feat = make_features(full).dropna().reset_index(drop=True)

    cutoff = feat["purchase_date"].quantile(0.8)  # last 20% of the timeline = test (time-aware, no shuffling)
    train_df = feat[feat["purchase_date"] < cutoff]
    test_df = feat[feat["purchase_date"] >= cutoff]

    feature_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)] + [
        "rolling_mean_7", "rolling_std_7", "dow", "month", "cat_code"
    ]
    X_train, y_train = train_df[feature_cols], train_df["order_count"]
    X_test, y_test = test_df[feature_cols], test_df["order_count"]

    # ---- Baseline: naive = lag_1 (yesterday's count for that category) ----
    baseline_pred = X_test["lag_1"]
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5

    # ---- Model: XGBoost ----
    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.clip(pred, 0, None)
    model_mae = mean_absolute_error(y_test, pred)
    model_rmse = mean_squared_error(y_test, pred) ** 0.5
    nonzero = y_test > 0
    model_mape = float(np.mean(np.abs((y_test[nonzero] - pred[nonzero]) / y_test[nonzero])) * 100) if nonzero.any() else None

    cat_map = dict(enumerate(full["category"].astype("category").cat.categories)) \
        if False else dict(enumerate(full["category"].astype("category").cat.categories))

    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "categories": sorted(full["category"].unique().tolist())},
                os.path.join(MODELS, "demand_model.pkl"))

    metrics = {
        "n_train": len(train_df), "n_test": len(test_df),
        "train_period": [str(train_df["purchase_date"].min().date()), str(train_df["purchase_date"].max().date())],
        "test_period": [str(test_df["purchase_date"].min().date()), str(test_df["purchase_date"].max().date())],
        "baseline_naive_lag1": {"MAE": round(baseline_mae, 3), "RMSE": round(baseline_rmse, 3)},
        "xgboost": {"MAE": round(model_mae, 3), "RMSE": round(model_rmse, 3), "MAPE_pct": round(model_mape, 2) if model_mape else None},
        "categories_modelled": len(keep_cats),
    }
    with open(os.path.join(MODELS, "demand_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
