# AI-Powered Predictive Analytics for Resilient and Adaptive Supply Chain Management

A working MVP demonstrating an integrated **Predict → Assess → What-if → Recommend →
Explain** workflow for supply-chain decision support, built on the real **Olist Brazilian
E-Commerce** public dataset (~99,441 orders, 2016-2018).

## Problem
Supply chains face uncertainty in demand, supplier reliability, and delivery timing.
This project connects prediction of each of these into one transparent risk score,
lets you simulate interventions, and recommends the best one with a traceable reason.

## Objectives
1. Predict category-level demand, seller risk, and delivery-lateness from real data.
2. Combine those predictions into one transparent Integrated Risk Score.
3. Simulate "what if" interventions and their effect on risk, cost and service level.
4. Recommend an action with an explainable, non-black-box justification.
5. Explain individual model predictions with SHAP.

## Architecture
See [`docs/architecture.md`](docs/architecture.md) for the full data-flow diagram.

## Modules
| # | Module | Code |
|---|---|---|
| 1 | Data ingestion & preprocessing | `app/preprocessing/build_dataset.py` |
| 2 | Demand forecasting (XGBoost vs naive baseline) | `app/forecasting/train_demand_model.py` |
| 3 | Supplier/seller risk — **Model A** (legacy, same-period) | `app/supplier_risk/train_supplier_risk_model.py` |
| 3b | Supplier/seller risk — **Model B** (time-aware, dashboard default) | `app/supplier_risk/build_temporal_features.py`, `train_supplier_risk_model_temporal.py` |
| 4 | Delivery/ETA delay prediction (leakage-safe) | `app/delivery/train_delivery_model.py` |
| 4b | Delay magnitude conditional on lateness (grounds what-if delay estimates) | `app/delivery/train_delay_magnitude_model.py` |
| 5 | Integrated risk engine | `app/risk_engine/risk_engine.py` |
| 6 | Inventory / stockout (simulated) | part of `risk_engine.py` + dashboard slider |
| 7 | What-if scenario analysis (risk, delay, shortage, cost) | `app/scenarios/whatif.py` |
| 8 | Adaptive recommendation engine | `app/scenarios/whatif.py` |
| 9 | Explainable AI (SHAP) | `app/explainability/shap_utils.py` |
| 10 | Streamlit dashboard (9 tabs, batch-scored visualizations) | `streamlit_app.py`, `app/utils/dashboard_data.py` |

**Improvement pass (this revision):** added the time-aware supplier-risk model (Model B),
exposed delay/shortage impact in the what-if engine, fixed a SHAP model-mismatch bug found
during the upgrade, and rebuilt the dashboard's charts to use real, batch-scored model
output instead of single illustrative numbers. See `docs/methodology.md` and
`docs/model_evaluation.md` for the honest before/after comparison.

**Methodological review pass (this revision):** replaced the what-if engine's simulated
`delivery_risk × 6.0` expected-delay mapping with a real, verified decomposition
`E[delay] = P(late) × E[delay|late]`, backed by a new delay-magnitude regression model
that beats a naive baseline by ~24% MAE on held-out data. See `docs/methodology.md` §
"Expected delay magnitude" for the full investigation and honest result.

## Dataset
Olist Brazilian E-Commerce Public Dataset. See [`docs/data_dictionary.md`](docs/data_dictionary.md)
for full schema and [`docs/assumptions.md`](docs/assumptions.md) for everything that is
simulated or assumed rather than directly observed (inventory levels, action costs,
supplier-risk proxy label, risk weights/thresholds).

## Installation
```bash
cd ai-powered-supply-chain
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## How to train the models (run once, in order)
```bash
python3 app/preprocessing/build_dataset.py                    # -> data/processed/*.csv
python3 app/forecasting/train_demand_model.py                 # -> models/demand_model.pkl
python3 app/supplier_risk/train_supplier_risk_model.py        # -> models/supplier_risk_model.pkl (Model A, legacy)
python3 app/supplier_risk/build_temporal_features.py          # -> data/processed/seller_profile_temporal.csv
python3 app/supplier_risk/train_supplier_risk_model_temporal.py  # -> models/supplier_risk_model_temporal.pkl (Model B, dashboard default)
python3 app/delivery/train_delivery_model.py                  # -> models/delivery_model.pkl
python3 app/delivery/train_delay_magnitude_model.py            # -> models/delay_magnitude_model.pkl (run after train_delivery_model.py)
```

## How to run the dashboard
```bash
streamlit run streamlit_app.py
```
Then open the URL Streamlit prints (usually http://localhost:8501).

## Example workflow in the dashboard
1. **Demand Forecast** tab — pick a category, see its trend and next-day forecast.
2. **Supplier Risk** tab — score a seller by its operating profile.
3. **Delivery Prediction** tab — score a hypothetical order's late-delivery probability.
4. **Integrated Risk** tab — combine those three signals (+ a simulated inventory level)
   into one Overall Risk score and category.
5. **What-If Analysis** tab — compare 4 candidate actions.
6. **Recommendation** tab — see the recommended action and why.
7. **Explainability** tab — see SHAP-based global/local explanations.

## Evaluation results (from actually running the code — see `models/*_metrics.json`)
| Model | Metric | Baseline | XGBoost |
|---|---|---|---|
| Demand forecasting | MAE | 1.866 | **1.580** |
| Supplier risk — Model A (legacy) | ROC-AUC | — | 0.602 |
| Supplier risk — Model B (time-aware, dashboard default) | ROC-AUC | — | 0.604 |
| Delivery lateness | ROC-AUC | — | 0.684 |

Model B is used by the dashboard because it is a genuinely time-separated (features
strictly precede the label window) prediction, not because its ROC-AUC is meaningfully
higher — it isn't. See `docs/model_evaluation.md` for the full honest comparison.

Full breakdown, including precision/recall/F1 and honest discussion of imbalanced-class
baselines, is in [`docs/model_evaluation.md`](docs/model_evaluation.md).

## Tests
```bash
python3 -m pytest tests/ -v
```
14 tests cover: processed-data integrity, leakage-guard verification, model
load/predict sanity checks, risk-engine math, and the scenario/recommendation engine.

## Limitations
- Supplier-risk label is a documented proxy, not ground truth.
- Inventory levels and action costs are simulated (Olist has no inventory/cost tables).
- Model discriminative power is modest but honestly reported (no leakage, no tautological features).
- No production hardening (auth, live data feeds, monitoring, DB).

See [`docs/assumptions.md`](docs/assumptions.md) and [`docs/model_evaluation.md`](docs/model_evaluation.md)
for full detail, and [`docs/viva_guide.md`](docs/viva_guide.md) to prepare for a project review.

## Future work
See Q20 in [`docs/viva_guide.md`](docs/viva_guide.md).
