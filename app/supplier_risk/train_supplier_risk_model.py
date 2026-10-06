"""
Module 3 - Supplier / Seller Risk Prediction
=============================================
The Olist dataset has no direct "supplier failed" label, so we construct a
transparent PROXY TARGET from measurable historical seller behaviour:

    seller_high_risk = 1 if (pct_late > 0.15) OR (avg_review_score < 3.5)
                             OR (cancellation_rate > 0.05)
                       else 0

This is documented as a proxy, not an observed ground-truth failure label
(see docs/assumptions.md). Because it is *derived from the same aggregated
features* used to predict it, we do NOT feed pct_late / avg_review_score /
cancellation_rate directly into the model as predictors (that would be
tautological). Instead the model is trained on *independent* seller
descriptors (order volume, pricing, freight, catalogue breadth, geography)
to see whether risk is predictable from a seller's operating profile alone.

Run:
    python3 app/supplier_risk/train_supplier_risk_model.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.dummy import DummyClassifier

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

# Features that are legitimately independent of the label components
FEATURE_COLS = ["n_orders", "avg_freight", "avg_price", "n_categories"]
LABEL_COMPONENTS = ["pct_late", "avg_review_score", "cancellation_rate"]


def make_label(df):
    df = df.copy()
    df["pct_late"] = df["pct_late"].fillna(0)
    df["avg_review_score"] = df["avg_review_score"].fillna(df["avg_review_score"].median())
    df["cancellation_rate"] = df["cancellation_rate"].fillna(0)
    df["seller_high_risk"] = (
        (df["pct_late"] > 0.15) | (df["avg_review_score"] < 3.5) | (df["cancellation_rate"] > 0.05)
    ).astype(int)
    return df


def train():
    sp = pd.read_csv(os.path.join(PROCESSED, "seller_profile.csv"))
    sp = make_label(sp)
    sp[FEATURE_COLS] = sp[FEATURE_COLS].fillna(sp[FEATURE_COLS].median())

    X = sp[FEATURE_COLS]
    y = sp["seller_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ---- baseline: majority-class dummy classifier ----
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    # ---- model: XGBoost classifier ----
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
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
        "n_sellers_total": len(sp),
        "n_train": len(X_train), "n_test": len(X_test),
        "positive_rate_overall": round(float(y.mean()), 3),
        "label_definition": "pct_late>0.15 OR avg_review_score<3.5 OR cancellation_rate>0.05 (PROXY, not observed ground truth)",
        "baseline_majority_class": clf_metrics(y_test, dummy_pred),
        "xgboost": clf_metrics(y_test, pred, proba),
        "features_used": FEATURE_COLS,
    }

    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, os.path.join(MODELS, "supplier_risk_model.pkl"))
    with open(os.path.join(MODELS, "supplier_risk_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
