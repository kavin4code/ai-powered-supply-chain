"""
Module 4 - Delivery / ETA / Delay Prediction
=============================================
Target: is_late (1 if delivered after the estimated delivery date).

LEAKAGE GUARD: features are restricted to information available at order
PURCHASE time only (estimated delivery window, price, freight, item count,
category, geography, payment plan, calendar). We deliberately EXCLUDE the
actual delivered date / actual days-to-deliver / delivery_delay_days from
the feature set, since those are only known after the outcome occurs.

Run:
    python3 app/delivery/train_delivery_model.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

NUMERIC_FEATURES = [
    "estimated_days_to_deliver", "n_items", "item_price_total", "freight_value_total",
    "avg_item_price", "n_distinct_sellers", "n_payment_installments", "payment_value_total",
    "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
    "purchase_month", "purchase_dow", "cross_state_shipment",
]
CATEGORICAL_FEATURES = ["customer_state", "seller_state", "payment_type_primary"]


def train():
    df = pd.read_csv(os.path.join(PROCESSED, "orders_master.csv"), parse_dates=["order_purchase_timestamp"])
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(subset=["is_late"])
    df["is_late"] = df["is_late"].astype(int)

    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].fillna("unknown").astype(str)

    df_enc = pd.get_dummies(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=True)
    feature_cols = df_enc.columns.tolist()

    df = df.sort_values("order_purchase_timestamp")
    df_enc = df_enc.loc[df.index]
    cutoff = df["order_purchase_timestamp"].quantile(0.8)
    train_mask = df["order_purchase_timestamp"] < cutoff

    X_train, X_test = df_enc[train_mask.values], df_enc[~train_mask.values]
    y_train, y_test = df.loc[train_mask, "is_late"], df.loc[~train_mask, "is_late"]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    def clf_metrics(y_true, y_pred, y_proba=None):
        out = {
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        }
        if y_proba is not None and len(set(y_true)) > 1:
            out["roc_auc"] = round(roc_auc_score(y_true, y_proba), 3)
        return out

    metrics = {
        "n_total_delivered_orders": len(df),
        "n_train": len(X_train), "n_test": len(X_test),
        "train_period": [str(df.loc[train_mask, "order_purchase_timestamp"].min().date()),
                          str(df.loc[train_mask, "order_purchase_timestamp"].max().date())],
        "test_period": [str(df.loc[~train_mask, "order_purchase_timestamp"].min().date()),
                         str(df.loc[~train_mask, "order_purchase_timestamp"].max().date())],
        "positive_rate_late_overall": round(float(df["is_late"].mean()), 3),
        "baseline_majority_class": clf_metrics(y_test, dummy_pred),
        "xgboost": clf_metrics(y_test, pred, proba),
        "n_features": len(feature_cols),
        "leakage_guard": "only pre-purchase/at-purchase-time features used; actual delivery date and "
                          "delivery_delay_days are excluded from predictors",
    }

    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES},
                os.path.join(MODELS, "delivery_model.pkl"))
    with open(os.path.join(MODELS, "delivery_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
