"""
Shared helpers to load trained models (joblib) and produce predictions used
across the risk engine, what-if scenarios and the Streamlit dashboard.
"""
import os
import joblib
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS = os.path.join(BASE, "models")
PROCESSED = os.path.join(BASE, "data", "processed")

_cache = {}


def _load(name):
    if name not in _cache:
        _cache[name] = joblib.load(os.path.join(MODELS, name))
    return _cache[name]


def load_demand_model():
    return _load("demand_model.pkl")


def load_supplier_model():
    """Legacy (Model A) supplier-risk model — same-period, non-temporal. Kept for comparison."""
    return _load("supplier_risk_model.pkl")


def load_supplier_model_temporal():
    """Time-aware (Model B) supplier-risk model — history window predicts future window.
    This is the model used as the PRIMARY supplier-risk predictor in the dashboard."""
    return _load("supplier_risk_model_temporal.pkl")


def load_delivery_model():
    return _load("delivery_model.pkl")


def load_delay_magnitude_model():
    """E[delay_days | late, features] - conditional delay-magnitude model. See
    app/delivery/train_delay_magnitude_model.py for methodology and leakage guard."""
    return _load("delay_magnitude_model.pkl")


def predict_supplier_risk_proba(seller_row: dict):
    """Legacy (Model A) prediction — kept for backward compatibility / comparison.
    seller_row must contain: n_orders, avg_freight, avg_price, n_categories"""
    bundle = load_supplier_model()
    model, cols = bundle["model"], bundle["feature_cols"]
    X = pd.DataFrame([{c: seller_row.get(c, 0) for c in cols}])
    return float(model.predict_proba(X)[:, 1][0])


def predict_supplier_risk_proba_temporal(seller_hist_row: dict):
    """Time-aware (Model B) prediction — the dashboard's primary supplier-risk predictor.
    seller_hist_row must contain the 8 history-window features: n_orders_hist,
    avg_review_score_hist, pct_late_hist, avg_delay_days_hist, avg_freight_hist,
    avg_price_hist, n_categories_hist, cancellation_rate_hist."""
    bundle = load_supplier_model_temporal()
    model, cols = bundle["model"], bundle["feature_cols"]
    X = pd.DataFrame([{c: seller_hist_row.get(c, 0) for c in cols}])
    return float(model.predict_proba(X)[:, 1][0])


def get_seller_profile_temporal_df():
    return pd.read_csv(os.path.join(PROCESSED, "seller_profile_temporal.csv"))


def predict_delivery_late_proba(order_row: dict):
    """order_row should contain raw (pre-encoding) numeric + categorical fields matching
    NUMERIC_FEATURES / CATEGORICAL_FEATURES used at training time."""
    bundle = load_delivery_model()
    model, cols = bundle["model"], bundle["feature_cols"]
    num_feats, cat_feats = bundle["numeric_features"], bundle["categorical_features"]

    row = {f: order_row.get(f, 0) for f in num_feats}
    for f in cat_feats:
        row[f] = order_row.get(f, "unknown")
    df = pd.DataFrame([row])
    df_enc = pd.get_dummies(df, columns=cat_feats, drop_first=True)
    df_enc = df_enc.reindex(columns=cols, fill_value=0)
    return float(model.predict_proba(df_enc)[:, 1][0])


def predict_conditional_delay_days(order_row: dict):
    """E[delay_days | late, features] for a specific hypothetical order - the
    order-conditional half of the expected_delay_days decomposition used by the
    what-if engine. Uses the SAME raw order_row shape as predict_delivery_late_proba."""
    bundle = load_delay_magnitude_model()
    model, cols = bundle["model"], bundle["feature_cols"]
    num_feats, cat_feats = bundle["numeric_features"], bundle["categorical_features"]

    row = {f: order_row.get(f, 0) for f in num_feats}
    for f in cat_feats:
        row[f] = order_row.get(f, "unknown")
    df = pd.DataFrame([row])
    df_enc = pd.get_dummies(df, columns=cat_feats, drop_first=True)
    df_enc = df_enc.reindex(columns=cols, fill_value=0)
    return float(max(0.0, model.predict(df_enc)[0]))


def get_observed_avg_delay_given_late_days():
    """Real, train-period-only historical average delay (days) among late orders.
    Used by the what-if engine as a fallback E[delay|late] when no concrete order
    features are available (e.g. a bare risk-slider scenario)."""
    bundle = load_delay_magnitude_model()
    return float(bundle["observed_avg_delay_given_late_train_period_days"])


def predict_demand_next(category: str, recent_daily_counts: list):
    """recent_daily_counts: most recent >=7 daily order counts for `category`, oldest->newest.
    Returns a single next-day forecast using the trained XGBoost demand model."""
    bundle = load_demand_model()
    model, cols, categories = bundle["model"], bundle["feature_cols"], bundle["categories"]
    vals = list(recent_daily_counts)[-7:]
    if len(vals) < 7:
        vals = [vals[0] if vals else 0] * (7 - len(vals)) + vals
    lag_feats = {f"lag_{i}": vals[-i] for i in range(1, 8)}
    rolling_mean_7 = float(np.mean(vals))
    rolling_std_7 = float(np.std(vals))
    cat_code = categories.index(category) if category in categories else -1
    row = {**lag_feats, "rolling_mean_7": rolling_mean_7, "rolling_std_7": rolling_std_7,
           "dow": 0, "month": 1, "cat_code": cat_code}
    X = pd.DataFrame([row])[cols]
    pred = float(model.predict(X)[0])
    return max(0.0, pred)


def get_seller_profile_df():
    return pd.read_csv(os.path.join(PROCESSED, "seller_profile.csv"))


def get_category_daily_demand_df():
    return pd.read_csv(os.path.join(PROCESSED, "category_daily_demand.csv"), parse_dates=["purchase_date"])


def get_orders_master_df(nrows=None):
    return pd.read_csv(os.path.join(PROCESSED, "orders_master.csv"), nrows=nrows)
