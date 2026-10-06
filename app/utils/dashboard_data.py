"""
Batch-scoring helpers for the Streamlit dashboard. These reuse the exact
feature-engineering functions from the training scripts (imported, not
duplicated) so the dashboard's charts are guaranteed consistent with how the
models were actually trained/evaluated - no re-implementation drift.
"""
import os
import pandas as pd
import numpy as np

from app.forecasting.train_demand_model import (
    load_series, make_full_calendar, make_features, N_LAGS, MIN_CATEGORY_DAYS,
)
from app.utils.predictors import (
    load_demand_model, load_supplier_model_temporal, load_delivery_model,
    get_seller_profile_temporal_df, get_orders_master_df,
)

BASE = os.path.join(os.path.dirname(__file__), "..", "..")


def demand_actual_vs_predicted(category: str):
    """Rebuild the same time-split features used in training and return actual vs
    predicted order_count for `category` on the held-out test period."""
    bundle = load_demand_model()
    model, cols = bundle["model"], bundle["feature_cols"]

    raw = load_series()
    counts_per_cat = raw.groupby("category")["purchase_date"].nunique()
    keep_cats = counts_per_cat[counts_per_cat >= MIN_CATEGORY_DAYS].index
    raw = raw[raw["category"].isin(keep_cats)]
    full = make_full_calendar(raw)
    feat = make_features(full).dropna().reset_index(drop=True)

    cutoff = feat["purchase_date"].quantile(0.8)
    cat_feat = feat[(feat["category"] == category) & (feat["purchase_date"] >= cutoff)].sort_values("purchase_date")
    if cat_feat.empty:
        return pd.DataFrame(columns=["purchase_date", "actual", "predicted"])
    pred = np.clip(model.predict(cat_feat[cols]), 0, None)
    return pd.DataFrame({
        "purchase_date": cat_feat["purchase_date"].values,
        "actual": cat_feat["order_count"].values,
        "predicted": pred,
    })


def score_all_sellers_temporal():
    """Score every seller in the time-aware supplier-risk table with Model B."""
    bundle = load_supplier_model_temporal()
    model, cols = bundle["model"], bundle["feature_cols"]
    df = get_seller_profile_temporal_df()
    X = df[cols].apply(lambda c: c.fillna(c.median()))
    df = df.copy()
    df["predicted_risk_proba"] = model.predict_proba(X)[:, 1]
    return df


def score_sample_orders_delivery(n=3000, random_state=7):
    """Score a random sample of delivered orders with the delivery model, using the
    exact same pre-purchase feature set as training (no leakage)."""
    bundle = load_delivery_model()
    model, cols, num_feats, cat_feats = (
        bundle["model"], bundle["feature_cols"], bundle["numeric_features"], bundle["categorical_features"]
    )
    df = get_orders_master_df()
    df = df[df["order_status"] == "delivered"].dropna(subset=["is_late"]).copy()
    df = df.sample(min(n, len(df)), random_state=random_state)

    for c in num_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    for c in cat_feats:
        df[c] = df[c].fillna("unknown").astype(str)

    enc = pd.get_dummies(df[num_feats + cat_feats], columns=cat_feats, drop_first=True)
    enc = enc.reindex(columns=cols, fill_value=0)
    df["predicted_late_proba"] = model.predict_proba(enc)[:, 1]
    return df
