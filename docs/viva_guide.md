# Viva Guide — 20 Questions, Explained Simply

**1. What problem does the project solve?**
Supply chains have many moving, uncertain parts: how much will customers buy, will a
supplier deliver on time, will the package arrive late, will we run out of stock. This
project builds one connected system that predicts each of these, combines them into a
single risk picture, lets you simulate "what if we did X", and recommends an action —
explaining *why*, not just *what*.

**2. Why is supply-chain resilience important?**
A single late delivery or stockout can cascade into lost sales, unhappy customers and
higher costs. Being able to *foresee* risk (rather than just react to it after the fact)
lets a business intervene early — switch a supplier, expedite a shipment, add stock —
before the problem actually happens.

**3. Why did we choose this dataset?**
The Olist Brazilian E-Commerce dataset is real, public, well-documented, and — uniquely —
has enough related tables (orders, items, payments, reviews, products, sellers) to support
*all* the project's modules (demand, supplier behaviour, delivery timing) from **one**
consistent source, avoiding fragile, fake joins across unrelated datasets.

**4. What is demand forecasting?**
Predicting how many orders/units a product category will get in the near future, based on
its recent history (e.g. the last 7 days) and calendar patterns (day of week, month).

**5. Why XGBoost?**
XGBoost is a gradient-boosted decision tree model — it's the standard, reliable choice for
structured/tabular data (rows and columns, like a spreadsheet), trains fast even on a
laptop, handles mixed numeric/categorical features well, and pairs efficiently with SHAP
for explanations. Deep learning needs much more data and compute to beat it here.

**6. What is supplier risk?**
The chance a seller will behave in a way that hurts the customer experience — late
deliveries, poor reviews, high cancellation rate. Olist doesn't label sellers as
"failed" directly, so we built a transparent **proxy label** from those measurable
behaviours (see Q8).

**7. How is delivery delay predicted?**
A classifier estimates the probability an order will arrive *after* the estimated delivery
date, using only information known **at the time of purchase** (estimated delivery window,
price, freight, product weight, payment plan, geography) — never information from after
the order was actually delivered.

**8. What is data leakage?**
When a model is trained using information it wouldn't actually have at prediction time —
for example, using the *actual* delivery date to predict *whether* delivery will be late.
This makes a model look artificially accurate in testing but useless in the real world,
because that information doesn't exist yet when you'd actually want the prediction.

**9. How did we prevent leakage?**
Two ways: (a) for delivery prediction, we explicitly excluded the actual delivered date
and delay columns from the feature list — verified by an automated test; (b) for both
demand and delivery models, we split train/test **by date** (train on earlier dates, test
on later dates) instead of randomly shuffling, so the model is never tested on data that
overlaps in time with information used to derive its features.

**9b. Why did supplier risk need a second, time-aware version?**
The original supplier-risk model built its label *and* its features from a seller's whole
order history at once — not a real forward prediction, just a same-period behavioural
score. In the improvement pass we built **Model B**: features come only from orders
*before* a fixed cutoff date, and the label comes only from orders *after* it. This is a
genuine "does past behaviour predict future behaviour" test. Its raw accuracy (ROC-AUC
0.604) barely changed from the original (0.602) — we kept it anyway because it is
methodologically sound, and said so honestly rather than pretending it was a performance
win.

**10. What is the integrated risk score?**
A single number from 0 to 1 combining four component risks — demand volatility, supplier
risk, delivery risk, stockout risk — using fixed, documented weights
(`0.2 / 0.3 / 0.3 / 0.2`). It's then bucketed into Low / Medium / High using two simple
threshold cut-points. Both the weights and thresholds are labelled as *design assumptions*,
not learned or scientifically proven values.

**11. How does what-if analysis work?**
Starting from the current risk situation, we simulate four possible actions (do nothing,
switch supplier, expedite delivery, increase inventory) by adjusting the relevant risk
components using documented effect multipliers, then recompute the integrated risk score
and a simulated relative cost for each.

**12. How does the recommendation engine work?**
Each scenario gets a "decision score" = `0.7 × overall_risk + 0.3 × relative_cost` (lower
is better). The engine picks the scenario with the lowest score and explains *why* using
the actual numbers from that scenario (e.g. "lower risk than doing nothing, acceptable
cost") — so the recommendation is traceable, not a mysterious black box.

**13. What is SHAP?**
SHapley Additive exPlanations. For a trained tree model, SHAP assigns each input feature a
number showing how much it pushed a specific prediction up or down, based on cooperative
game theory (borrowed from economics). We use it to show, e.g., "this seller's low order
volume pushed their predicted risk up."

**14. Does SHAP prove causality?**
No. SHAP explains what the *model* used to make its decision — it tells you about the
model's learned pattern, not about real-world cause and effect. A feature can have a high
SHAP value because of correlation, not because changing it would actually change the
outcome.

**15. Why not deep learning?**
The dataset is moderate-sized tabular data (tens of thousands of rows), where gradient-
boosted trees like XGBoost typically match or beat deep learning while training in seconds
on a laptop and staying explainable with SHAP. Deep learning would add complexity and
training time without a clear benefit here, and conflicts with the project's "no
overengineering" MVP constraint.

**15b. What did the visualization/UX upgrade pass add, and why?**
The dashboard originally showed mostly single numbers and one chart per tab. The upgrade
added charts that use *real, batch-computed* model outputs — e.g. scoring every seller
(752) and a 3,000-order sample with the actual trained models, rather than one
hand-picked example — plus an actual-vs-predicted line chart on the held-out demand test
period, late-delivery-rate-by-state from real data, and a scenario comparison table/charts
that visually highlight the recommended action. The goal was to make the risk situation
visually obvious to a review panel, not to add decorative charts.

**15c. How was "expected delay" made more defensible?**
Originally it was `delivery_risk × 6.0` — an assumed maximum with no basis in the data.
We reviewed whether this could be grounded in real data instead, using the standard
identity `E[delay] = P(late) × E[delay | late]`. `P(late)` was already a real model output.
We then trained a *new* regression model, only on historically-late orders, to estimate
`E[delay | late]` from the same leakage-safe pre-purchase features — and verified on
held-out data that it actually beats a naive constant (MAE 5.96 vs 7.81 days) before using
it. When no concrete order is available, we fall back to a real historical average (not
another guess). What we deliberately did **not** do: claim that any intervention changes
the delay *size* if a delay happens — we only model its effect on the delay *probability*,
because we have no data to support the stronger claim.

**16. What are the limitations?**
- Supplier-risk and delivery-lateness models have *modest* (not strong) discriminative
  power (ROC-AUC 0.60 and 0.68) — a realistic result once the models are kept
  leakage-free and non-tautological.
- Supplier risk is a proxy label, not ground truth.
- Inventory and action costs are simulated, not real.
- Geolocation data is loaded but not deeply used in the MVP models.
- No production deployment, authentication, or live data feed.

**17. What is the novelty?**
Not a new algorithm — the novelty is the **integration**: one coherent workflow that
connects forecasting, two risk models, a transparent scoring engine, scenario simulation,
a traceable recommendation, and explainability, all on one real dataset, with every
assumption explicitly labelled.

**18. What happens if the dataset changes?**
The pipeline is modular: `build_dataset.py` would need column-mapping updates if schema
changes, but the three model-training scripts and the risk/scenario engine would keep
working as long as the processed tables keep the same column names — that's why
preprocessing is separated from modelling.

**19. How would this become production-ready?**
Add: authentication and access control; a real inventory/ERP data feed instead of
simulation; scheduled retraining; a proper backend/API (e.g. FastAPI) instead of loading
CSVs directly in Streamlit; monitoring for model drift; validated (not assumed) cost data;
and a database instead of flat files.

**20. What would you improve with more time?**
Tune classification thresholds to the actual business cost of false positives/negatives;
try graph-based or time-series-specific models (e.g. as in the GCN literature reviewed);
incorporate geolocation distance as a delivery-time feature; validate the supplier-risk
proxy against any real complaint/return data if available; run a proper hyperparameter
search instead of fixed XGBoost settings.
