# Data Dictionary

## Source
**Olist Brazilian E-Commerce Public Dataset** — real, anonymized orders from the Olist
marketplace, 2016-2018 (~99,441 orders). Retrieved as 9 CSVs matching the canonical
Kaggle schema (`olistbr/brazilian-ecommerce`). Row counts were checked against the known
dataset size during preprocessing (see `data/processed/data_audit_report.md`).

## Raw tables (`data/raw/`)
| File | Rows | Key columns |
|---|---|---|
| olist_orders_dataset.csv | 99,441 | order_id, customer_id, order_status, 5 timestamp columns |
| olist_customers_dataset.csv | 99,441 | customer_id, customer_unique_id, customer_zip/city/state |
| olist_order_items_dataset.csv | 112,650 | order_id, product_id, seller_id, price, freight_value |
| olist_order_payments_dataset.csv | 103,886 | order_id, payment_type, installments, payment_value |
| olist_order_reviews_dataset.csv | 104,718 | order_id, review_score, review dates |
| olist_products_dataset.csv | 32,951 | product_id, category, dimensions/weight |
| olist_sellers_dataset.csv | 3,095 | seller_id, seller_zip/city/state |
| olist_geolocation_dataset.csv | 1,000,163 | zip_code_prefix, lat/lng (not used in MVP models) |
| product_category_name_translation.csv | 71 | category name PT -> EN |

## Processed table: `data/processed/orders_master.csv` (1 row / order, 99,441 rows, 45 cols)
Key engineered columns:
- `estimated_days_to_deliver` — days between purchase and Olist's estimated delivery date (**pre-outcome, safe feature**)
- `actual_days_to_deliver`, `delivery_delay_days`, `is_late` — **outcome/target columns**, only known after delivery, never used as model inputs
- `cross_state_shipment` — 1 if customer_state != seller_state
- `n_items`, `item_price_total`, `freight_value_total`, `avg_item_price`, `n_distinct_sellers` — order-item aggregates
- `payment_value_total`, `n_payment_installments`, `payment_type_primary` — payment aggregates
- `review_score` — from first review per order

## Processed table: `data/processed/seller_profile.csv` (1 row / seller, sellers with >=5 delivered orders)
`n_orders`, `avg_review_score`, `pct_late`, `avg_delay_days`, `avg_freight`, `avg_price`,
`n_categories`, `cancellation_rate`, `seller_high_risk` (derived label, see assumptions.md)

## Processed table: `data/processed/category_daily_demand.csv`
`category`, `purchase_date`, `order_count`, `units`, `revenue` — daily demand series per
product category, used by the forecasting module.

## Target variables
| Module | Target | Type | Leakage guard |
|---|---|---|---|
| Demand forecasting | `order_count` (next-day, per category) | regression | time-aware split, only lag/calendar features |
| Supplier risk | `seller_high_risk` (proxy) | binary classification | predictors independent of label components |
| Delivery risk | `is_late` | binary classification | only pre-purchase/at-purchase features used |
