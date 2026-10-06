"""
Module 1 - Data Ingestion & Preprocessing
==========================================
Loads the raw Olist Brazilian E-Commerce CSVs, cleans them, joins them into
an order-level master table, and derives three processed artifacts used by
the rest of the pipeline:

    data/processed/orders_master.csv        -> order-level table (1 row/order)
    data/processed/seller_profile.csv        -> seller-level aggregated features
    data/processed/category_daily_demand.csv -> category x day order-count series

Run:
    python3 app/preprocessing/build_dataset.py
"""
import os
import pandas as pd
import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)

DATE_COLS_ORDERS = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def _p(name):
    return os.path.join(RAW, name)


def load_raw():
    orders = pd.read_csv(_p("olist_orders_dataset.csv"))
    orders.columns = [c.strip().lower() for c in orders.columns]  # fix 'Order_id' -> 'order_id'
    customers = pd.read_csv(_p("olist_customers_dataset.csv"))
    items = pd.read_csv(_p("olist_order_items_dataset.csv"))
    payments = pd.read_csv(_p("olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(_p("olist_order_reviews_dataset.csv"))
    products = pd.read_csv(_p("olist_products_dataset.csv"))
    sellers = pd.read_csv(_p("olist_sellers_dataset.csv"))
    cat_translation = pd.read_csv(_p("product_category_name_translation.csv"), encoding="utf-8-sig")
    return orders, customers, items, payments, reviews, products, sellers, cat_translation


def audit(df, name):
    """Return a small dict describing shape / missingness / duplicates for the data audit report."""
    return {
        "table": name,
        "rows": len(df),
        "cols": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_top": df.isna().sum().sort_values(ascending=False).head(5).to_dict(),
    }


def build():
    orders, customers, items, payments, reviews, products, sellers, cat_translation = load_raw()

    audit_report = [
        audit(orders, "orders"), audit(customers, "customers"), audit(items, "order_items"),
        audit(payments, "payments"), audit(reviews, "reviews"), audit(products, "products"),
        audit(sellers, "sellers"),
    ]

    # ---- parse dates (source format is DD-MM-YYYY HH:MM, i.e. day-first) ----
    for c in DATE_COLS_ORDERS:
        orders[c] = pd.to_datetime(orders[c], dayfirst=True, errors="coerce")

    # ---- drop exact duplicate rows ----
    for df in (orders, customers, items, payments, products, sellers):
        df.drop_duplicates(inplace=True)

    # ---- products: translate category to English, fill missing category ----
    products = products.merge(cat_translation, on="product_category_name", how="left")
    products["product_category_name_english"] = (
        products["product_category_name_english"].fillna(products["product_category_name"])
    )
    products["product_category_name_english"] = products["product_category_name_english"].fillna("unknown")
    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        products[col] = products[col].fillna(products[col].median())

    # ---- order items aggregated to one row per order (an order can have many items/sellers) ----
    items_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        item_price_total=("price", "sum"),
        freight_value_total=("freight_value", "sum"),
        avg_item_price=("price", "mean"),
        n_distinct_sellers=("seller_id", "nunique"),
    ).reset_index()
    # primary seller/product per order = first item (used for seller-risk join; documented assumption)
    first_item = items.sort_values("order_item_id").groupby("order_id").first().reset_index()
    first_item = first_item[["order_id", "product_id", "seller_id"]]
    items_agg = items_agg.merge(first_item, on="order_id", how="left")

    # ---- payments aggregated ----
    pay_agg = payments.groupby("order_id").agg(
        payment_value_total=("payment_value", "sum"),
        n_payment_installments=("payment_installments", "max"),
        n_payment_types=("payment_type", "nunique"),
    ).reset_index()
    pay_type_mode = (
        payments.sort_values("payment_value", ascending=False)
        .groupby("order_id")["payment_type"].first().reset_index()
        .rename(columns={"payment_type": "payment_type_primary"})
    )
    pay_agg = pay_agg.merge(pay_type_mode, on="order_id", how="left")

    # ---- reviews: one review per order (take first if duplicated) ----
    rev = reviews.sort_values("review_creation_date").groupby("order_id").first().reset_index()
    rev = rev[["order_id", "review_score"]]

    # ---- master join ----
    master = orders.merge(customers, on="customer_id", how="left")
    master = master.merge(items_agg, on="order_id", how="left")
    master = master.merge(pay_agg, on="order_id", how="left")
    master = master.merge(rev, on="order_id", how="left")
    master = master.merge(products, on="product_id", how="left")
    master = master.merge(sellers, on="seller_id", how="left", suffixes=("", "_seller"))

    # ---- derived time features (available at order-purchase time -> safe for pre-outcome features) ----
    master["purchase_year"] = master["order_purchase_timestamp"].dt.year
    master["purchase_month"] = master["order_purchase_timestamp"].dt.month
    master["purchase_dow"] = master["order_purchase_timestamp"].dt.dayofweek
    master["purchase_date"] = master["order_purchase_timestamp"].dt.date
    master["estimated_days_to_deliver"] = (
        master["order_estimated_delivery_date"] - master["order_purchase_timestamp"]
    ).dt.days

    # ---- derived OUTCOME features (only known after delivery -> targets, never used as predictors) ----
    master["actual_days_to_deliver"] = (
        master["order_delivered_customer_date"] - master["order_purchase_timestamp"]
    ).dt.days
    master["delivery_delay_days"] = (
        master["order_delivered_customer_date"] - master["order_estimated_delivery_date"]
    ).dt.days
    master["is_late"] = (master["delivery_delay_days"] > 0).astype("Int64")
    master.loc[master["order_delivered_customer_date"].isna(), "is_late"] = pd.NA

    # ---- geo mismatch flag (customer state != seller state) - available pre-outcome ----
    master["cross_state_shipment"] = (
        master["customer_state"] != master["seller_state"]
    ).astype("Int64")

    # ---- basic missing-value handling for downstream modelling columns ----
    master["review_score"] = master["review_score"]  # keep NaN -> handled per-module
    master["n_items"] = master["n_items"].fillna(0)
    master["item_price_total"] = master["item_price_total"].fillna(0)
    master["freight_value_total"] = master["freight_value_total"].fillna(0)

    master.to_csv(os.path.join(PROCESSED, "orders_master.csv"), index=False)

    # ---- Module: seller profile table (used by supplier/seller risk model) ----
    delivered = master[master["order_status"] == "delivered"].copy()
    seller_profile = delivered.groupby("seller_id").agg(
        n_orders=("order_id", "count"),
        avg_review_score=("review_score", "mean"),
        pct_late=("is_late", "mean"),
        avg_delay_days=("delivery_delay_days", "mean"),
        avg_freight=("freight_value_total", "mean"),
        avg_price=("avg_item_price", "mean"),
        n_categories=("product_category_name_english", "nunique"),
    ).reset_index()
    cancel_rate = master.groupby("seller_id").apply(
        lambda g: (g["order_status"].isin(["canceled", "unavailable"])).mean()
    ).reset_index(name="cancellation_rate")
    seller_profile = seller_profile.merge(cancel_rate, on="seller_id", how="left")
    seller_profile = seller_profile[seller_profile["n_orders"] >= 5].reset_index(drop=True)  # need enough history
    seller_profile.to_csv(os.path.join(PROCESSED, "seller_profile.csv"), index=False)

    # ---- Module: category-level daily demand series (used by demand forecasting) ----
    demand = delivered.dropna(subset=["product_category_name_english"]).copy()
    demand["purchase_date"] = pd.to_datetime(demand["purchase_date"])
    cat_daily = demand.groupby(["product_category_name_english", "purchase_date"]).agg(
        order_count=("order_id", "count"),
        units=("n_items", "sum"),
        revenue=("item_price_total", "sum"),
    ).reset_index().rename(columns={"product_category_name_english": "category"})
    cat_daily.to_csv(os.path.join(PROCESSED, "category_daily_demand.csv"), index=False)

    # ---- write data audit report ----
    with open(os.path.join(PROCESSED, "data_audit_report.md"), "w") as f:
        f.write("# Data Audit Report\n\nGenerated by app/preprocessing/build_dataset.py\n\n")
        for a in audit_report:
            f.write(f"## {a['table']}\n")
            f.write(f"- rows: {a['rows']}\n- cols: {a['cols']}\n- duplicate_rows: {a['duplicate_rows']}\n")
            f.write(f"- top missing columns: {a['missing_top']}\n\n")
        f.write(f"## orders_master (final joined table)\n- rows: {len(master)}\n- cols: {master.shape[1]}\n")
        f.write(f"- delivered orders: {(master['order_status']=='delivered').sum()}\n")
        f.write(f"- orders missing delivery date: {master['order_delivered_customer_date'].isna().sum()}\n")
        f.write(f"\n## seller_profile\n- sellers with >=5 orders: {len(seller_profile)}\n")
        f.write(f"\n## category_daily_demand\n- rows: {len(cat_daily)}\n- categories: {cat_daily['category'].nunique()}\n")

    print(f"orders_master: {master.shape}")
    print(f"seller_profile: {seller_profile.shape}")
    print(f"category_daily_demand: {cat_daily.shape}")
    return master, seller_profile, cat_daily


if __name__ == "__main__":
    build()
