# Architecture

```
ai-powered-supply-chain/
├── app/
│   ├── preprocessing/build_dataset.py     # Module 1: raw CSVs -> processed tables
│   ├── forecasting/train_demand_model.py  # Module 2: demand forecasting (XGBoost)
│   ├── supplier_risk/train_supplier_risk_model.py  # Module 3: seller risk (XGBoost)
│   ├── delivery/train_delivery_model.py   # Module 4: delivery lateness (XGBoost)
│   ├── risk_engine/risk_engine.py         # Module 5: integrated weighted risk score
│   ├── scenarios/whatif.py                # Modules 7 & 8: what-if + recommendation
│   ├── explainability/shap_utils.py       # Module 9: SHAP explanations
│   └── utils/predictors.py                # shared model-loading / prediction helpers
├── data/
│   ├── raw/            # original Olist CSVs (9 files)
│   └── processed/      # orders_master.csv, seller_profile.csv, category_daily_demand.csv
├── models/              # trained .pkl models + *_metrics.json (evaluation results)
├── tests/test_pipeline.py
├── docs/                 # this file + data_dictionary, methodology, model_evaluation,
│                          # literature_review, assumptions, viva_guide
├── streamlit_app.py       # Module 10: dashboard (9 tabs)
└── requirements.txt
```

## Data flow
```
data/raw/*.csv
   │  (app/preprocessing/build_dataset.py)
   ▼
data/processed/{orders_master, seller_profile, category_daily_demand}.csv
   │
   ├─► app/forecasting/train_demand_model.py     ─► models/demand_model.pkl
   ├─► app/supplier_risk/train_supplier_risk_model.py ─► models/supplier_risk_model.pkl
   └─► app/delivery/train_delivery_model.py      ─► models/delivery_model.pkl
                                                        │
                     app/utils/predictors.py  (loads models, exposes predict_* functions)
                                                        │
        ┌───────────────────────────────────────────────┤
        ▼                                                ▼
app/risk_engine/risk_engine.py            app/explainability/shap_utils.py
        │ (Overall Risk score)
        ▼
app/scenarios/whatif.py  (4 scenarios + recommendation)
        │
        ▼
streamlit_app.py  (9-tab dashboard: Overview → Demand → Supplier → Delivery →
                    Integrated Risk → What-If → Recommendation → Explainability → Data Info)
```

## Design choices
- **Models trained offline, loaded at runtime** — `streamlit_app.py` never retrains; it
  loads `models/*.pkl` via `joblib`, so page refreshes are fast.
- **No FastAPI/Postgres/microservices** — per the project's MVP constraint, everything
  runs as plain Python modules + one Streamlit process, locally, with `python3` and
  `streamlit run`.
- **Every risk/cost number that isn't learned from data is a named constant** in
  `risk_engine.py` / `whatif.py`, so the "no hidden arbitrary mathematics" requirement is
  satisfiable by reading two short files.
