"""
Module 4b - Delay MAGNITUDE model (conditional on lateness) — methodological review pass
============================================================================================
Motivation: the what-if engine previously mapped delivery_risk -> expected_delay_days via
a single SIMULATED constant (delivery_risk * 6.0 "max assumed days"). This script asks:
can that mapping be grounded in real data without leakage or fabricated calibration?

We estimate the standard two-part decomposition of an expected value:

    E[delay_days] = P(late | features) * E[delay_days | late, features]

- P(late | features) is the EXISTING delivery classifier (train_delivery_model.py) — unchanged.
- E[delay_days | late, features] is a NEW regression model trained here, using the exact
  same pre-purchase feature set as the classifier (imported, not duplicated), but fit only
  on orders that were historically late (delivery_delay_days > 0). This answers: "given an
  order's pre-purchase attributes, if it does end up late, how many days late is it
  typically?" It is evaluated with the SAME chronological 80/20 cutoff as the classifier,
  so training never sees future-period orders (late or otherwise).

LEAKAGE GUARD: identical to the classifier — only pre-purchase/at-purchase-time features
are used as predictors. The target (delivery_delay_days) and the subsetting variable
(is_late) are never used as predictors, only as the label / row filter.

IMPORTANT — what this model does NOT claim:
- It does NOT establish that any what-if intervention (switch supplier, expedite) changes
  this conditional delay magnitude. The what-if engine still assumes interventions act
  only on P(late), not on E[delay | late] - a documented simplification (see
  docs/assumptions.md), because we have no data on how an intervention would change delay
  magnitude specifically.
- It is a historical-association model, not a causal one.

Run:
    python3 app/delivery/train_delay_magnitude_model.py
(after app/delivery/train_delivery_model.py, whose feature lists it imports)
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.delivery.train_delivery_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)


def train():
    df = pd.read_csv(os.path.join(PROCESSED, "orders_master.csv"), parse_dates=["order_purchase_timestamp"])
    df = df[df["order_status"] == "delivered"].dropna(subset=["is_late"]).copy()
    df["is_late"] = df["is_late"].astype(int)

    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].fillna("unknown").astype(str)

    # SAME chronological cutoff as the classifier (computed over ALL delivered orders,
    # not just late ones) so both models share one consistent train/test time boundary.
    df = df.sort_values("order_purchase_timestamp")
    cutoff = df["order_purchase_timestamp"].quantile(0.8)
    train_mask_all = df["order_purchase_timestamp"] < cutoff

    # ---- historical fallback constant: mean delay among late orders, TRAIN PERIOD ONLY ----
    # Used by the what-if engine when no concrete order features are available (e.g. a bare
    # risk-slider input) - grounded in real data, but restricted to the train period so it
    # is consistent with what the models were actually fit on, not "peeking" at the test period.
    late_train = df[(df["is_late"] == 1) & train_mask_all]
    observed_avg_delay_given_late_train_days = round(float(late_train["delivery_delay_days"].mean()), 2)

    # ---- regression model: E[delay_days | late, features], late orders only ----
    late = df[df["is_late"] == 1].copy()
    enc = pd.get_dummies(late[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=True)
    feature_cols = enc.columns.tolist()
    late_train_mask = late["order_purchase_timestamp"] < cutoff

    X_train, X_test = enc[late_train_mask.values], enc[~late_train_mask.values]
    y_train = late.loc[late_train_mask, "delivery_delay_days"]
    y_test = late.loc[~late_train_mask, "delivery_delay_days"]

    # ---- baseline: predict the train-period mean delay for every late test order ----
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5

    model = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 1, None)
    model_mae = mean_absolute_error(y_test, pred)
    model_rmse = mean_squared_error(y_test, pred) ** 0.5

    metrics = {
        "target": "delivery_delay_days, conditional on the order being late (delivery_delay_days > 0)",
        "n_late_orders_total": len(late),
        "n_train": len(X_train), "n_test": len(X_test),
        "train_period": [str(late.loc[late_train_mask, "order_purchase_timestamp"].min().date()),
                          str(late.loc[late_train_mask, "order_purchase_timestamp"].max().date())],
        "test_period": [str(late.loc[~late_train_mask, "order_purchase_timestamp"].min().date()),
                         str(late.loc[~late_train_mask, "order_purchase_timestamp"].max().date())],
        "baseline_train_mean": {"MAE": round(baseline_mae, 3), "RMSE": round(baseline_rmse, 3),
                                 "predicted_constant_days": round(float(y_train.mean()), 2)},
        "xgboost_conditional_delay": {"MAE": round(model_mae, 3), "RMSE": round(model_rmse, 3)},
        "observed_avg_delay_given_late_train_period_days": observed_avg_delay_given_late_train_days,
        "leakage_guard": "identical feature set to the delivery-lateness classifier; delivery_delay_days "
                          "and is_late are used only as the label / row filter, never as predictors",
        "not_a_causal_claim": "this model estimates historical association between pre-purchase order "
                               "attributes and delay magnitude GIVEN lateness; it does not estimate how "
                               "any intervention would change that magnitude",
    }

    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
                 "observed_avg_delay_given_late_train_period_days": observed_avg_delay_given_late_train_days},
                os.path.join(MODELS, "delay_magnitude_model.pkl"))
    with open(os.path.join(MODELS, "delay_magnitude_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
