# Assumptions, Simulations & Proxies

This project is built on **real Olist e-commerce data**, but the Olist dataset does not
contain everything a full supply-chain system would need (inventory levels, supplier
failure labels, real switching/expediting costs). Wherever something is not directly
observed in the data, it is clearly labelled below as an **ASSUMPTION** or **SIMULATED**.
This document is the single source of truth for those labels.

## 1. Supplier/Seller risk PROXY target (Module 3)
Olist has no "supplier failed" column. Both Model A and Model B use the same **proxy**
rule:
```
seller_high_risk = 1 if (pct_late > 0.15) OR (avg_review_score < 3.5) OR (cancellation_rate > 0.05)
```
Thresholds (15%, 3.5, 5%) are prototype assumptions chosen to flag sellers with clearly
poor historical behaviour, not statistically derived cut points. This is a **PROXY**, not
an observed ground-truth failure label, for **both** model versions below.

**Model A (legacy, same-period):** label and predictors are both computed over a
seller's *entire* order history. To avoid an outright tautological model, Model A is
trained on features *independent* of the label's own components (order volume, average
freight, average price, catalogue breadth) — a conservative choice that produces weaker
metrics than "cheating" would, but the metrics are real.

**Model B (time-aware, current dashboard default — added in the improvement pass):**
the label (`seller_high_risk_future`) is computed **only** from a seller's orders on/after
a fixed cutoff date (2018-04-15, the 70th-percentile purchase date), while the 8
predictors (`*_hist`, including historical `pct_late_hist` / `avg_review_score_hist` /
`cancellation_rate_hist`) are computed **only** from orders strictly before that cutoff.
Because feature and label windows never overlap, using historical `pct_late_hist` etc. as
predictors here is legitimate "past behaviour predicts future behaviour" modelling, not
tautological — this is the key methodological improvement over Model A. See
`docs/methodology.md` and `docs/model_evaluation.md` for the full comparison. Model B's
window split (70th percentile, ≥5 history orders, ≥3 future orders) is itself a documented
**ASSUMPTION**, chosen to balance sample size against a meaningful future window.

## 2. Integrated risk engine weights (Module 5)
```
Overall Risk = 0.20*Demand + 0.30*Supplier + 0.30*Delivery + 0.20*Stockout
```
These weights are a **DESIGN ASSUMPTION**: supplier and delivery risk are weighted
higher because they were judged the most actionable/interventable in the what-if engine.
They are not learned from data and are not scientifically validated. They are exposed as
a constant (`app/risk_engine/risk_engine.py::WEIGHTS`) so a reviewer can change them.

## 3. Risk category thresholds
```
score < 0.33 -> Low Risk
score < 0.66 -> Medium Risk
else         -> High Risk
```
**PROTOTYPE DECISION THRESHOLDS**, evenly-spaced for demo interpretability — not derived
from a cost-benefit or statistical analysis.

## 4. Inventory (Module 6)
Olist has **no inventory table**. Available inventory is a **SIMULATED** value the user
sets via a dashboard slider (or a scenario default of `+25%` of expected demand for the
"increase inventory" action). It is never presented as historical fact.

## 5. Relative costs of what-if actions (Module 7)
```
do_nothing = 0.00, increase_inventory = 0.35, switch_supplier = 0.55, expedite_delivery = 0.65
```
These are **SIMULATED, NORMALIZED relative costs** (0 = free, 1 = very costly), used only
to demonstrate trade-off reasoning. They are not real market prices — Olist has no
switching, expediting, or carrying-cost data.

## 6. Scenario effect sizes (Module 7)
E.g. `switch_supplier` reduces supplier risk to 35% of its original value; `expedite_delivery`
reduces delivery risk to 40% of its original value. These multipliers are **ASSUMPTIONS**
meant to produce plausible, explainable directional effects — they are not fitted to data
(Olist has no record of suppliers actually being switched). They are **not** causal effects:
they demonstrate decision-support reasoning, not a validated real-world intervention model.

## 6b. Expected delay days per scenario (Module 7) — grounded in real data (methodological review pass)
Originally `expected_delay_days = delivery_risk × 6.0` (a purely assumed maximum). This
was reviewed and replaced with a data-grounded decomposition:
```
E[delay_days] = P(late) × E[delay_days | late]
```
- `P(late)` is the existing, real delivery-lateness classifier output (adjusted per
  scenario by the **ASSUMED** `EFFECT` multipliers below — that part is still simulated).
- `E[delay_days | late]` is either (a) a **REAL, order-specific** prediction from
  `app/delivery/train_delay_magnitude_model.py` — a regression model trained only on
  historically-late orders, which beats a naive train-mean baseline by ~24% MAE on
  held-out data (see `docs/model_evaluation.md` §3b) — when a concrete order's features
  are available (from the Delivery Prediction tab); or (b) a **REAL fallback constant**,
  the train-period-only historical average delay among late orders (11.08 days), when no
  order features are available (e.g. a bare risk-slider scenario).

**What is still an assumption, clearly separated from the real parts above:** each
what-if action is assumed to change only `P(late)` (via `EFFECT`, below), not
`E[delay_days | late]` itself — i.e. we assume interventions reduce the *chance* of being
late but we do not claim any evidence about how much they would shrink the delay *size* if
a delay still happens. This is a documented simplification, not a hidden one, and it is
**not** a causal claim: `E[delay_days | late]` describes historical association between an
order's pre-purchase attributes and its eventual delay size, nothing more.

## 7. Recommendation decision score (Module 8)
```
decision_score = 0.7 * overall_risk + 0.3 * relative_cost   (lower is better)
```
An **ASSUMED** weighting between risk-reduction and cost that is transparent and can be
re-weighted; it is not a learned utility function.

## 8. Assumed relationship between tables
`order_items` can contain multiple sellers per order; for the seller-risk join we take the
**first item's seller** per order as a simplifying assumption (documented in
`app/preprocessing/build_dataset.py`). This slightly understates multi-seller order
complexity but avoids a fabricated allocation rule.

## 9. What is NOT simulated
- All raw data (orders, customers, products, sellers, reviews, payments) is the real,
  unmodified (only cleaned/typed) Olist dataset.
- All model metrics (MAE, RMSE, MAPE, accuracy, precision, recall, F1, ROC-AUC) are
  computed by actually running the training scripts in this repository — none are invented.
- SHAP values are computed on the actual trained models.
- The delay-magnitude model's predictions and the 11.08-day fallback constant (§6b) are
  both real, verified outputs of code run against the actual dataset — not assumed numbers.
