# Literature Review

## Project base paper
> Md Raisul Islam Khan, "AI-powered predictive analytics for enhancing supply chain
> resilience and delivery outcomes," *Discover Analytics*, 2026. DOI: 10.1007/s44257-026-00065-6

**Note on verification:** web searches run while building this project (Sept 2026) could
not independently locate or confirm this specific DOI/article. This does not mean the
paper does not exist — it may simply not yet be indexed by the search tools available —
but **the student should independently verify this citation directly on the publisher's
site (Springer/Discover Analytics) before including it in submitted academic work**, per
the academic-honesty requirement of this project (never fabricate citations).

## Other relevant, independently-verified literature (2022-2026)

1. **Liu, Hu, Zhang & Wu (2026)**, *"Predictive analytics for supply chain resilience in
   urban infrastructure networks using graph convolutional networks,"* PLOS ONE,
   DOI: 10.1371/journal.pone.0345444.
   - **Method:** three-stage framework combining network analysis, a GCN predictive model
     and resilience stress-testing, validated on a large NYC infrastructure dataset.
   - **Finding:** the GCN approach outperformed ARIMA, SVR, LSTM and XGBoost baselines on
     disruption prediction, and identified temporal/historical-flow features as the
     dominant predictors.
   - **Relevance:** supports this project's choice to give heavy weight to
     historical/behavioural features (e.g. seller delivery history) and confirms
     XGBoost is a standard, credible baseline in this domain.
   - **Gap vs our project:** graph-based modelling is out of scope for a 3-day tabular MVP;
     noted as future work.

2. **arXiv:2601.09680**, *"Automating Supply Chain Disruption Monitoring via an Agentic
   AI Approach."*
   - **Problem:** traditional control towers and digital twins only cover Tier-1/internal
     data and miss upstream disruption signals.
   - **Relevance:** motivates why this project frames its contribution as an *integrated,
     decision-support workflow* rather than a single disconnected model — echoing the
     paper's critique of siloed tools.

3. **arXiv:2604.01775**, *"Bridging Deep Learning and Integer Linear Programming: A
   Predictive-to-Prescriptive Framework for Supply Chain Analytics."*
   - **Method:** N-BEATS/N-HiTS demand forecasting feeding an integer-linear-programming
     shipment-allocation stage.
   - **Relevance:** validates the general "predict-then-decide" pattern this project
     follows (forecast -> risk engine -> scenario optimisation), though we use simpler,
     more explainable tree-based models and rule-based scenario scoring instead of deep
     learning + ILP, appropriate for the MVP's scope and explainability goals.

4. **arXiv:2509.05325**, *"SynDelay: A Synthetic Dataset for Delivery Delay Prediction."*
   - **Problem:** notes that real logistics datasets for delivery-delay prediction are
     scarce or poorly documented, hindering reproducible benchmarking.
   - **Relevance:** reinforces the value of using a well-documented real dataset (Olist)
     with transparent preprocessing, as this project does, rather than an undocumented or
     synthetic one.

## Explainable AI
General XAI literature (SHAP — Lundberg & Lee, *"A Unified Approach to Interpreting Model
Predictions,"* NeurIPS 2017) establishes SHAP as a standard model-attribution method for
tree ensembles, which this project uses via `shap.TreeExplainer`. As emphasized throughout
this project's documentation, SHAP explains what a trained model attributed a prediction
to; it is **not** evidence of real-world causality.

## Positioning of this project's contribution
Consistent with the base-paper framing, this project does **not** claim a new algorithm or
"first-ever" system. Its contribution is the **integration** of demand forecasting,
supplier risk, delivery risk, inventory impact, what-if intervention analysis, adaptive
recommendation and explainability into one coherent, auditable decision-support workflow
on real e-commerce data — with every simulated/assumed element explicitly labelled.
