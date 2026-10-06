"""
Supplier risk - TIME-AWARE feature/label construction (improvement pass).
============================================================================
The original supplier-risk model (train_supplier_risk_model.py) computes both
its proxy label AND its features from a seller's ENTIRE order history with no
time separation - it is not a genuine forward-looking prediction, just a
same-period behavioural classification.

This module builds a proper temporal formulation:

    cutoff = 70th percentile of delivered-order purchase dates (2018-04-15)
    HISTORY WINDOW  (purchase_date < cutoff)  -> features (what we know "so far")
    FUTURE WINDOW   (purchase_date >= cutoff) -> label (what actually happened next)

Only sellers with >=5 orders in the history window AND >=3 orders in the
future window are kept (need enough history to build features, and enough
future orders to build a reliable future-behaviour label). This is a
genuine train-on-past / predict-the-future setup: no future information is
used to build the features, and the label is built exclusively from orders
that occur strictly after the feature cutoff.

Output: data/processed/seller_profile_temporal.csv
"""
import os
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED = os.path.join(BASE, "data", "processed")

MIN_HIST_ORDERS = 5
MIN_FUTURE_ORDERS = 3
FUTURE_WINDOW_QUANTILE = 0.70  # cutoff = 70th percentile of purchase dates


def _high_risk_label(pct_late, avg_review_score, cancellation_rate):
    pct_late = 0 if pd.isna(pct_late) else pct_late
    avg_review_score = 5.0 if pd.isna(avg_review_score) else avg_review_score
    cancellation_rate = 0 if pd.isna(cancellation_rate) else cancellation_rate
    return int((pct_late > 0.15) or (avg_review_score < 3.5) or (cancellation_rate > 0.05))


def build():
    df = pd.read_csv(os.path.join(PROCESSED, "orders_master.csv"), parse_dates=["order_purchase_timestamp"])
    delivered = df[df["order_status"] == "delivered"].copy()

    cutoff = delivered["order_purchase_timestamp"].quantile(FUTURE_WINDOW_QUANTILE)

    hist = df[df["order_purchase_timestamp"] < cutoff]  # all statuses, for cancellation rate
    hist_delivered = delivered[delivered["order_purchase_timestamp"] < cutoff]
    future = df[df["order_purchase_timestamp"] >= cutoff]
    future_delivered = delivered[delivered["order_purchase_timestamp"] >= cutoff]

    # ---- HISTORY WINDOW features (known "so far", available to build a predictor) ----
    hist_feat = hist_delivered.groupby("seller_id").agg(
        n_orders_hist=("order_id", "count"),
        avg_review_score_hist=("review_score", "mean"),
        pct_late_hist=("is_late", "mean"),
        avg_delay_days_hist=("delivery_delay_days", "mean"),
        avg_freight_hist=("freight_value_total", "mean"),
        avg_price_hist=("avg_item_price", "mean"),
        n_categories_hist=("product_category_name_english", "nunique"),
    ).reset_index()
    cancel_hist = hist.groupby("seller_id").apply(
        lambda g: (g["order_status"].isin(["canceled", "unavailable"])).mean()
    ).reset_index(name="cancellation_rate_hist")
    hist_feat = hist_feat.merge(cancel_hist, on="seller_id", how="left")

    # ---- FUTURE WINDOW label (what actually happened after the cutoff - ground truth for evaluation) ----
    fut_label = future_delivered.groupby("seller_id").agg(
        n_orders_future=("order_id", "count"),
        avg_review_score_future=("review_score", "mean"),
        pct_late_future=("is_late", "mean"),
    ).reset_index()
    cancel_fut = future.groupby("seller_id").apply(
        lambda g: (g["order_status"].isin(["canceled", "unavailable"])).mean()
    ).reset_index(name="cancellation_rate_future")
    fut_label = fut_label.merge(cancel_fut, on="seller_id", how="left")
    fut_label["seller_high_risk_future"] = fut_label.apply(
        lambda r: _high_risk_label(r["pct_late_future"], r["avg_review_score_future"], r["cancellation_rate_future"]),
        axis=1,
    )

    merged = hist_feat.merge(fut_label, on="seller_id", how="inner")
    merged = merged[
        (merged["n_orders_hist"] >= MIN_HIST_ORDERS) & (merged["n_orders_future"] >= MIN_FUTURE_ORDERS)
    ].reset_index(drop=True)

    out_path = os.path.join(PROCESSED, "seller_profile_temporal.csv")
    merged.to_csv(out_path, index=False)

    print(f"cutoff date: {cutoff}")
    print(f"sellers with >= {MIN_HIST_ORDERS} history orders & >= {MIN_FUTURE_ORDERS} future orders: {len(merged)}")
    print(f"future_high_risk positive rate: {merged['seller_high_risk_future'].mean():.3f}")
    return merged, cutoff


if __name__ == "__main__":
    build()
