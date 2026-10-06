"""
Supplier risk - TIME-AWARE model (improvement pass, Model B).
================================================================
Predicts a seller's FUTURE risk (orders placed on/after 2018-04-15, the 70th
percentile purchase date) from their HISTORY-WINDOW behaviour only (orders
strictly before that date). See build_temporal_features.py for how the
history/future windows and the future-window proxy label are constructed.

Because features and label are now built from non-overlapping time periods,
past pct_late / avg_review_score / cancellation_rate are legitimate
predictors here (they are NOT computed from the same orders as the label,
unlike the original same-period Model A) - this is standard "past behaviour
predicts future behaviour" modelling, not tautological.

This script also re-reports Model A's (legacy, same-period) metrics for a
transparent side-by-side comparison in docs/model_evaluation.md. Neither
model is post-hoc tuned to look better; both are reported as obtained.

Run:
    python3 app/supplier_risk/build_temporal_features.py
    python3 app/supplier_risk/train_supplier_risk_model_temporal.py
"""
import os
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")

FEATURE_COLS = [
    "n_orders_hist", "avg_review_score_hist", "pct_late_hist", "avg_delay_days_hist",
    "avg_freight_hist", "avg_price_hist", "n_categories_hist", "cancellation_rate_hist",
]
LABEL_COL = "seller_high_risk_future"


def _clf_metrics(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        out["roc_auc"] = round(roc_auc_score(y_true, y_proba), 3)
    return out


def train():
    df = pd.read_csv(os.path.join(PROCESSED, "seller_profile_temporal.csv"))
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(lambda c: c.fillna(c.median()))

    X, y = df[FEATURE_COLS], df[LABEL_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Model B - time-aware (history window -> future window)",
        "n_sellers_total": len(df),
        "n_train": len(X_train), "n_test": len(X_test),
        "positive_rate_future_high_risk": round(float(y.mean()), 3),
        "label_definition": "seller_high_risk_future: pct_late_future>0.15 OR avg_review_future<3.5 OR "
                             "cancellation_rate_future>0.05, computed ONLY from orders on/after the cutoff date "
                             "(PROXY target, not an observed ground-truth failure label)",
        "features_used": FEATURE_COLS,
        "feature_window": "orders strictly BEFORE the cutoff date only",
        "label_window": "orders on/after the cutoff date only (no overlap with feature window)",
        "baseline_majority_class": _clf_metrics(y_test, dummy_pred),
        "xgboost_time_aware": _clf_metrics(y_test, pred, proba),
    }

    # ---- load Model A's (legacy) metrics if present, for an honest side-by-side comparison ----
    legacy_path = os.path.join(MODELS, "supplier_risk_model_metrics.json")
    if os.path.exists(legacy_path):
        with open(legacy_path) as f:
            legacy = json.load(f)
        metrics["comparison_vs_model_A_legacy"] = {
            "model_A_roc_auc": legacy.get("xgboost", {}).get("roc_auc"),
            "model_A_recall": legacy.get("xgboost", {}).get("recall"),
            "model_B_roc_auc": metrics["xgboost_time_aware"].get("roc_auc"),
            "model_B_recall": metrics["xgboost_time_aware"].get("recall"),
            "note": "Model A is same-period/non-temporal (features and label share the same order history). "
                    "Model B is genuinely forward-looking (features strictly precede the label window). "
                    "Model B is preferred on METHODOLOGICAL grounds (real prediction of the future) "
                    "regardless of which has the higher raw metric.",
        }

    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "cutoff_note": "see build_temporal_features.py"},
                os.path.join(MODELS, "supplier_risk_model_temporal.pkl"))
    with open(os.path.join(MODELS, "supplier_risk_model_temporal_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
