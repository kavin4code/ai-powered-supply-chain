# Methodology

## Overall workflow
`PREDICT -> ASSESS -> WHAT-IF -> RECOMMEND -> EXPLAIN`

1. **Preprocess** the 9 raw Olist CSVs into 3 analysis-ready tables (order-level,
   seller-level, category-daily-demand).
2. **Predict** three independent signals: category demand, seller risk, delivery-lateness
   probability — each with its own baseline vs ML model comparison.
3. **Assess**: combine the three predictions (+ a simulated stockout risk) into one
   transparent, weighted Integrated Risk Score.
4. **What-if**: simulate 4 candidate interventions and their effect on that score.
5. **Recommend**: pick the action with the best risk/cost trade-off, with a traceable
   reason (not a black-box output).
6. **Explain**: SHAP values show which features drove the supplier-risk and
   delivery-risk predictions.

## Time-aware evaluation
For both the demand-forecasting and delivery-lateness models, splits are **chronological**
(train on the earliest ~80% of dates, test on the most recent ~20%) rather than random,
because random splitting on time-series/event data leaks future information into training.

## Supplier risk: Model A (legacy) vs Model B (time-aware) — improvement pass
The original supplier-risk model (**Model A**, `train_supplier_risk_model.py`) computes
both its proxy label and its predictors from a seller's *entire* order history, with no
time separation between them. It is a same-period behavioural classifier, not a genuine
forward prediction.

A revised, genuinely time-aware formulation (**Model B**,
`build_temporal_features.py` + `train_supplier_risk_model_temporal.py`) was implemented
and evaluated:
- **Cutoff:** the 70th-percentile purchase date across delivered orders (2018-04-15).
- **History window** (`< cutoff`): behavioural features (`pct_late_hist`,
  `avg_review_score_hist`, `cancellation_rate_hist`, plus `n_orders_hist`,
  `avg_delay_days_hist`, `avg_freight_hist`, `avg_price_hist`, `n_categories_hist`).
- **Future window** (`>= cutoff`): the proxy label `seller_high_risk_future`, computed
  *exclusively* from orders in this later window.
- Only sellers with ≥5 history orders and ≥3 future orders are kept (752 of 1,756
  sellers) — enough for a genuine train-on-past/predict-the-future evaluation.

Because the feature and label windows no longer overlap, `pct_late_hist` etc. are now
legitimate predictors (past behaviour predicting *different, later* orders) rather than
tautological ones. **Model B is used as the dashboard's primary supplier-risk predictor**
on methodological grounds (real forward-looking prediction), not because it scores
higher — see `docs/model_evaluation.md` for the honest side-by-side comparison, where
Model B's ROC-AUC (0.604) is essentially unchanged from Model A's (0.602). We report this
directly rather than presenting Model B as a performance win.

## Class imbalance
Both classification targets (`seller_high_risk`, `is_late`) are imbalanced (17.1% and
6.8% positive respectively). We therefore report precision/recall/F1/ROC-AUC alongside
accuracy, and use `scale_pos_weight` in XGBoost rather than relying on accuracy alone,
which would be misleadingly high for a majority-class predictor.

## Leakage prevention (delivery model)
The delivery-lateness model is restricted to features known **before** the outcome is
observed: the Olist-estimated delivery window, item/payment/product attributes, calendar
features and geography. `order_delivered_customer_date`, `actual_days_to_deliver` and
`delivery_delay_days` are excluded from the feature set (this is verified by an automated
test in `tests/test_pipeline.py::test_no_target_leakage_columns_used_in_delivery_training`).

## Why XGBoost as the primary model
Gradient-boosted trees (XGBoost) were chosen over deep learning because: (a) the data is
tabular with a mix of numeric/categorical features, where boosted trees are a strong,
well-established default; (b) the ~99K-row dataset is small by deep-learning standards;
(c) the project needs to train and explain (SHAP) models quickly on a laptop within a
3-day timeline; (d) tree-based SHAP (`TreeExplainer`) is fast and exact, unlike
approximate SHAP methods often needed for neural nets.

## Model comparison approach
Every predictive module reports a naive/majority-class **baseline** next to the XGBoost
result, so the added value of the ML model (or its absence) is visible rather than assumed.

## What-if / recommendation engine — improvement pass
The scenario engine (`app/scenarios/whatif.py`) now exposes, for every one of the 4
candidate actions: `overall_risk`, `risk_category`, `supplier_risk`, `delivery_risk`,
**`expected_delay_days`**, `stockout_risk`, **`shortage_units`** (an absolute unit count
rather than only a normalized ratio), `service_level`, `relative_cost`, and
`decision_score`. `scenarios_to_dataframe()` turns the result into one clean, sorted
`pandas.DataFrame` (best `decision_score` first, recommended action flagged) that the
dashboard renders directly, so the table shown in the UI is exactly what the engine
computed — no separate re-derivation in the presentation layer.

## Expected delay magnitude — methodological review pass
`expected_delay_days` was originally a purely simulated mapping
(`delivery_risk × 6.0`, an assumed maximum). A later review asked whether this could be
grounded in real data without leakage. We investigated the standard two-part
decomposition of an expected value:

```
E[delay_days] = P(late | features) × E[delay_days | late, features]
```

- **P(late | features)** is the existing delivery-lateness classifier's output (unchanged).
- **E[delay_days | late, features]** is a NEW regression model
  (`app/delivery/train_delay_magnitude_model.py`), trained only on historically-late
  orders, using the *same* pre-purchase feature set as the classifier (imported, not
  duplicated) and the *same* chronological 80/20 cutoff. On held-out (test-period) late
  orders it achieves **MAE 5.96 days vs a 7.81-day naive baseline** (predicting the
  train-period mean for every test order) — a genuine, verified improvement, not an
  assumed number.

When a concrete order's features are available (e.g. after using the Delivery Prediction
tab), the dashboard passes this model's order-specific estimate into the what-if engine
via `conditional_delay_days`. When only an abstract risk value is available (e.g. a bare
slider in the Integrated Risk tab with no underlying order), the engine falls back to a
**real, train-period-only historical average delay among late orders** (11.08 days —
computed once by the training script, not fabricated at call time, and not the same as
the whole-dataset average of 10.6 days, to avoid quietly mixing in test-period
information for a value used outside of model evaluation).

**What this decomposition does NOT claim:** neither half of it is a causal effect, and the
scenario engine still assumes each intervention (switch supplier, expedite, etc.) changes
only `P(late)` (via the existing `EFFECT` multipliers, a documented simulation assumption)
— **not** the conditional delay magnitude if a delay does occur. We have no data on how an
intervention would change delay magnitude specifically, so we deliberately did not invent
that; it remains a documented simplification (see `docs/assumptions.md`).
