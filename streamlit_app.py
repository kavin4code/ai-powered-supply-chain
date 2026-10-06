"""
Module 10 - Streamlit Dashboard (v2 - visualization/UX upgrade pass)
======================================================================
Run with:  streamlit run streamlit_app.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from app.utils.predictors import (
    predict_supplier_risk_proba_temporal, predict_delivery_late_proba, predict_demand_next,
    predict_conditional_delay_days, get_seller_profile_temporal_df, get_category_daily_demand_df,
    load_supplier_model_temporal, load_delivery_model,
)
from app.utils.dashboard_data import demand_actual_vs_predicted, score_all_sellers_temporal, score_sample_orders_delivery
from app.risk_engine.risk_engine import (
    demand_risk_from_volatility, stockout_risk, integrated_risk, WEIGHTS, RISK_THRESHOLDS,
)
from app.scenarios.whatif import run_scenarios, scenarios_to_dataframe
from app.explainability.shap_utils import supplier_risk_shap, delivery_risk_shap, global_importance

BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")
RISK_COLOR = {"Low Risk": "#2ca02c", "Medium Risk": "#ff7f0e", "High Risk": "#d62728"}

st.set_page_config(page_title="AI Supply Chain Risk Platform", layout="wide")


@st.cache_data
def load_metrics():
    out = {}
    for name in ["demand_model_metrics.json", "supplier_risk_model_metrics.json",
                 "supplier_risk_model_temporal_metrics.json", "delivery_model_metrics.json"]:
        path = os.path.join(MODELS, name)
        if os.path.exists(path):
            with open(path) as f:
                out[name] = json.load(f)
    return out


@st.cache_data
def load_category_demand():
    return get_category_daily_demand_df()


@st.cache_data
def load_scored_sellers():
    return score_all_sellers_temporal()


@st.cache_data
def load_scored_orders(n=3000):
    return score_sample_orders_delivery(n)


@st.cache_data
def load_demand_backtest(category):
    return demand_actual_vs_predicted(category)


metrics = load_metrics()
demand_df = load_category_demand()
scored_sellers = load_scored_sellers()
scored_orders = load_scored_orders()

st.sidebar.title("Supply Chain AI")
st.sidebar.markdown("**PREDICT** \n↓ \n**ASSESS** \n↓ \n**WHAT-IF** \n↓ \n**RECOMMEND** \n↓ \n**EXPLAIN**")
st.sidebar.caption("Real Olist data. Simulated values are always labelled SIMULATED.")
page = st.sidebar.radio("Navigate", [
    "1. Overview", "2. Demand Forecast", "3. Supplier Risk", "4. Delivery Prediction",
    "5. Integrated Risk", "6. What-If Analysis", "7. Recommendation", "8. Explainability",
    "9. Data / Model Info",
])


def note(text):
    st.caption(f"ℹ️ {text}")


# ============================================================ 1. OVERVIEW
if page.startswith("1"):
    st.title("AI-Powered Predictive Analytics for Resilient Supply Chain Management")
    st.markdown("""
`PREDICT` demand, supplier risk & delivery delay → `ASSESS` an integrated risk score →
`WHAT-IF` simulate interventions → `RECOMMEND` the best action → `EXPLAIN` with SHAP.
Built on the real **Olist Brazilian E-Commerce** dataset (~99,441 orders, 2016-2018).
""")
    om_path = os.path.join(BASE, "data", "processed", "orders_master.csv")
    n_orders = (sum(1 for _ in open(om_path)) - 1) if os.path.exists(om_path) else "N/A"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders modelled", f"{n_orders:,}" if isinstance(n_orders, int) else n_orders)
    c2.metric("Sellers risk-scored", f"{len(scored_sellers):,}")
    c3.metric("Categories forecasted", f"{demand_df['category'].nunique():,}")
    dm = metrics.get("delivery_model_metrics.json", {})
    c4.metric("Delivery model ROC-AUC", dm.get("xgboost", {}).get("roc_auc", "N/A"))

    st.subheader("Risk signal distributions (from actually-run model predictions)")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.histogram(scored_sellers, x="predicted_risk_proba", nbins=30,
                            title="Supplier risk (all scored sellers)", color_discrete_sequence=["#1f77b4"])
        fig.update_layout(xaxis_title="Predicted high-risk probability")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(scored_orders, x="predicted_late_proba", nbins=30,
                            title="Delivery lateness risk (sampled orders)", color_discrete_sequence=["#ff7f0e"])
        fig.update_layout(xaxis_title="Predicted late-delivery probability")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        vol = demand_df.groupby("category")["order_count"].apply(lambda s: demand_risk_from_volatility(s.tail(30).tolist()))
        fig = px.histogram(vol, nbins=30, title="Demand volatility risk (per category)", color_discrete_sequence=["#2ca02c"])
        fig.update_layout(xaxis_title="Volatility-based demand risk (0-1)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    note("These are real model outputs at scale (all sellers, a 3,000-order sample, all categories) - "
         "not single illustrative numbers.")

# ============================================================ 2. DEMAND FORECAST
elif page.startswith("2"):
    st.title("Demand Forecasting")
    dm = metrics.get("demand_model_metrics.json", {})
    if dm:
        c1, c2, c3 = st.columns(3)
        c1.metric("XGBoost MAE", dm["xgboost"]["MAE"],
                   delta=f"-{round(dm['baseline_naive_lag1']['MAE']-dm['xgboost']['MAE'],2)} vs baseline", delta_color="normal")
        c2.metric("XGBoost RMSE", dm["xgboost"]["RMSE"])
        c3.metric("Baseline (naive lag-1) MAE", dm["baseline_naive_lag1"]["MAE"])
        bdf = pd.DataFrame({
            "model": ["Baseline (naive lag-1)", "XGBoost"],
            "MAE": [dm["baseline_naive_lag1"]["MAE"], dm["xgboost"]["MAE"]],
            "RMSE": [dm["baseline_naive_lag1"]["RMSE"], dm["xgboost"]["RMSE"]],
        })
        fig = px.bar(bdf.melt(id_vars="model"), x="model", y="value", color="variable", barmode="group",
                     title="Model vs baseline (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
        note(f"Time-aware split (no shuffling): train {dm['train_period'][0]}–{dm['train_period'][1]}, "
             f"test {dm['test_period'][0]}–{dm['test_period'][1]}.")

    st.subheader("Top categories by total demand")
    top_cats = demand_df.groupby("category")["order_count"].sum().sort_values(ascending=False).head(15).reset_index()
    st.plotly_chart(px.bar(top_cats, x="order_count", y="category", orientation="h",
                            title="Top 15 categories by total orders (real data)").update_layout(yaxis=dict(autorange="reversed")),
                     use_container_width=True)

    st.subheader("Actual vs predicted demand (held-out test period)")
    categories = sorted(demand_df["category"].unique().tolist())
    default_idx = categories.index("bed_bath_table") if "bed_bath_table" in categories else 0
    cat = st.selectbox("Category", categories, index=default_idx)
    bt = load_demand_backtest(cat)
    if not bt.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt["purchase_date"], y=bt["actual"], name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(x=bt["purchase_date"], y=bt["predicted"], name="Predicted (XGBoost)", mode="lines"))
        fig.update_layout(title=f"Actual vs predicted daily orders — {cat} (test period only, never seen in training)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("This category has too little history to appear in the held-out test period.")

    st.subheader("Next-day forecast")
    cat_series = demand_df[demand_df["category"] == cat].sort_values("purchase_date")
    recent = cat_series["order_count"].tail(7).tolist()
    if recent:
        c1, c2 = st.columns(2)
        c1.metric("Next-day forecast (orders)", round(predict_demand_next(cat, recent), 1))
        c2.metric("Demand volatility risk (0-1)", round(demand_risk_from_volatility(cat_series["order_count"].tail(30).tolist()), 3))
    note("The forecast horizon shown is one day ahead, matching how the model was trained and evaluated "
         "(lag/rolling features assume daily granularity).")

# ============================================================ 3. SUPPLIER RISK
elif page.startswith("3"):
    st.title("Supplier / Seller Risk Prediction")
    note("Supplier risk is a proxy-based screening score derived from historical seller behaviour, "
         "not an observed ground-truth failure label.")
    sm = metrics.get("supplier_risk_model_temporal_metrics.json", {})
    if sm:
        st.markdown(f"**Model B (time-aware, primary):** trained on history-window seller behaviour to predict "
                     f"**future**-window risk — a genuine forward prediction, not a same-period classification.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", sm["xgboost_time_aware"]["accuracy"])
        c2.metric("Precision", sm["xgboost_time_aware"]["precision"])
        c3.metric("Recall", sm["xgboost_time_aware"]["recall"])
        c4.metric("ROC-AUC", sm["xgboost_time_aware"].get("roc_auc", "N/A"))
        with st.expander("Model A (legacy, same-period) vs Model B (time-aware) comparison"):
            comp = sm.get("comparison_vs_model_A_legacy", {})
            st.json(comp)
            st.caption("Model B is preferred for methodological soundness (genuine out-of-time prediction), "
                       "not because its raw AUC is higher — the two are comparable.")

    st.subheader("Seller risk distribution (Model B, all scored sellers)")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(scored_sellers, x="predicted_risk_proba", nbins=30,
                            title="Predicted future high-risk probability across sellers")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cat_counts = pd.cut(scored_sellers["predicted_risk_proba"], [0, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"]).value_counts()
        fig = px.pie(values=cat_counts.values, names=cat_counts.index, title="Sellers by risk band")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 highest-risk sellers")
    top_risky = scored_sellers.sort_values("predicted_risk_proba", ascending=False).head(15)
    st.dataframe(top_risky[["seller_id", "predicted_risk_proba", "n_orders_hist", "pct_late_hist",
                             "avg_review_score_hist", "cancellation_rate_hist"]], use_container_width=True)

    st.subheader("Important supplier-risk features (SHAP global importance)")
    bundle = load_supplier_model_temporal()
    cols = bundle["feature_cols"]
    sample = get_seller_profile_temporal_df()[cols].apply(lambda c: c.fillna(c.median())).sample(min(300, len(scored_sellers)), random_state=1)
    sv = supplier_risk_shap(sample)
    imp = global_importance(sv, cols)
    st.plotly_chart(px.bar(imp, x="mean_abs_shap", y="feature", orientation="h",
                            title="Global feature importance (mean |SHAP value|)"), use_container_width=True)

    st.subheader("Score a seller (using history-window features)")
    c1, c2 = st.columns(2)
    default = get_seller_profile_temporal_df().iloc[0]
    n_orders_hist = c1.number_input("n_orders_hist", min_value=1, value=int(default["n_orders_hist"]))
    avg_review_hist = c1.number_input("avg_review_score_hist", min_value=1.0, max_value=5.0, value=float(default["avg_review_score_hist"]))
    pct_late_hist = c1.number_input("pct_late_hist", min_value=0.0, max_value=1.0, value=float(default["pct_late_hist"]))
    avg_delay_hist = c1.number_input("avg_delay_days_hist", value=float(default["avg_delay_days_hist"]))
    avg_freight_hist = c2.number_input("avg_freight_hist (R$)", min_value=0.0, value=float(default["avg_freight_hist"]))
    avg_price_hist = c2.number_input("avg_price_hist (R$)", min_value=0.0, value=float(default["avg_price_hist"]))
    n_categories_hist = c2.number_input("n_categories_hist", min_value=1, value=int(default["n_categories_hist"]))
    cancel_hist = c2.number_input("cancellation_rate_hist", min_value=0.0, max_value=1.0, value=float(default["cancellation_rate_hist"]))

    if st.button("Predict future supplier risk"):
        proba = predict_supplier_risk_proba_temporal({
            "n_orders_hist": n_orders_hist, "avg_review_score_hist": avg_review_hist, "pct_late_hist": pct_late_hist,
            "avg_delay_days_hist": avg_delay_hist, "avg_freight_hist": avg_freight_hist, "avg_price_hist": avg_price_hist,
            "n_categories_hist": n_categories_hist, "cancellation_rate_hist": cancel_hist,
        })
        st.metric("Predicted future high-risk probability", f"{proba*100:.1f}%")
        st.session_state["supplier_risk_proba"] = proba

# ============================================================ 4. DELIVERY PREDICTION
elif page.startswith("4"):
    st.title("Delivery / ETA / Delay Prediction")
    dm = metrics.get("delivery_model_metrics.json", {})
    if dm:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", dm["xgboost"]["accuracy"])
        c2.metric("Precision", dm["xgboost"]["precision"])
        c3.metric("Recall", dm["xgboost"]["recall"])
        c4.metric("ROC-AUC", dm["xgboost"].get("roc_auc", "N/A"))
        note(dm["leakage_guard"])

    st.subheader("On-time vs late deliveries (real data)")
    om = pd.read_csv(os.path.join(BASE, "data", "processed", "orders_master.csv"), usecols=["is_late", "order_status", "customer_state"])
    late_counts = om.dropna(subset=["is_late"])["is_late"].map({0: "On time", 1: "Late"}).value_counts()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(values=late_counts.values, names=late_counts.index,
                                title="On-time vs late (all delivered orders)",
                                color=late_counts.index, color_discrete_map={"On time": "#2ca02c", "Late": "#d62728"}),
                         use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(scored_orders, x="predicted_late_proba", nbins=30,
                                      title="Predicted late-delivery probability (sampled orders)"),
                         use_container_width=True)

    st.subheader("Late-delivery rate by customer state (real data)")
    state_late = om.dropna(subset=["is_late"]).groupby("customer_state")["is_late"].mean().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(state_late, x="customer_state", y="is_late", title="Late-delivery rate by customer state")
                     .update_layout(yaxis_title="Late-delivery rate"), use_container_width=True)

    st.subheader("Predict delay risk for a hypothetical order")
    c1, c2, c3 = st.columns(3)
    est_days = c1.number_input("Estimated days to deliver", min_value=1, value=20)
    n_items = c1.number_input("Number of items", min_value=1, value=1)
    freight = c1.number_input("Freight value total (R$)", min_value=0.0, value=20.0)
    price = c2.number_input("Item price total (R$)", min_value=0.0, value=100.0)
    installments = c2.number_input("Payment installments", min_value=1, value=2)
    weight = c2.number_input("Product weight (g)", min_value=0.0, value=500.0)
    cust_state = c3.selectbox("Customer state", sorted(om["customer_state"].dropna().unique().tolist()))
    seller_state = c3.selectbox("Seller state", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF"], index=0)
    pay_type = c3.selectbox("Payment type", ["credit_card", "boleto", "voucher", "debit_card"])

    if st.button("Predict delay risk"):
        order = {
            "estimated_days_to_deliver": est_days, "n_items": n_items, "item_price_total": price,
            "freight_value_total": freight, "avg_item_price": price / max(n_items, 1),
            "n_distinct_sellers": 1, "n_payment_installments": installments, "payment_value_total": price + freight,
            "product_weight_g": weight, "product_length_cm": 20, "product_height_cm": 10, "product_width_cm": 15,
            "purchase_month": 6, "purchase_dow": 2, "cross_state_shipment": int(cust_state != seller_state),
            "customer_state": cust_state, "seller_state": seller_state, "payment_type_primary": pay_type,
        }
        proba = predict_delivery_late_proba(order)
        cond_delay = predict_conditional_delay_days(order)
        c1, c2 = st.columns(2)
        c1.metric("Predicted probability of late delivery", f"{proba*100:.1f}%")
        c2.metric("Predicted delay IF late (days)", round(cond_delay, 1))
        note("The right-hand figure is E[delay | late] for THIS order's attributes — from a regression "
             "model trained only on historically-late orders (MAE 5.96d vs a 7.81d naive baseline on held-out data). "
             "It is a historical association, not a promise about this specific future order.")
        st.session_state["delivery_risk_proba"] = proba
        st.session_state["last_order_input"] = order
        st.session_state["conditional_delay_days"] = cond_delay

# ============================================================ 5. INTEGRATED RISK
elif page.startswith("5"):
    st.title("Integrated Supply-Chain Risk Engine")
    st.markdown(f"""
**Overall Risk = {WEIGHTS['demand_risk']}·Demand + {WEIGHTS['supplier_risk']}·Supplier +
{WEIGHTS['delivery_risk']}·Delivery + {WEIGHTS['stockout_risk']}·Stockout**
""")
    note("Weights and thresholds are a documented design assumption, not learned — see docs/assumptions.md.")
    c1, c2, c3, c4 = st.columns(4)
    demand_r = c1.slider("Demand risk", 0.0, 1.0, 0.3)
    supplier_r = c2.slider("Supplier risk", 0.0, 1.0, float(st.session_state.get("supplier_risk_proba", 0.3)))
    delivery_r = c3.slider("Delivery risk", 0.0, 1.0, float(st.session_state.get("delivery_risk_proba", 0.3)))
    expected_demand = c4.number_input("Expected demand (units)", min_value=1, value=100)
    inventory = st.slider("Available inventory (units) [SIMULATED]", 0, 300, 80)

    stock_r = stockout_risk(expected_demand, inventory)
    result = integrated_risk(demand_r, supplier_r, delivery_r, stock_r)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Overall risk score", result["overall_score"])
        color = RISK_COLOR[result["category"]]
        st.markdown(f"### <span style='color:{color}'>{result['category']}</span>", unsafe_allow_html=True)
        st.caption(f"Thresholds (prototype): Low < {RISK_THRESHOLDS['low']}, "
                   f"Medium < {RISK_THRESHOLDS['medium']}, else High")
    with c2:
        comp_df = pd.DataFrame({
            "component": list(result["components"].keys()),
            "value": list(result["components"].values()),
        })
        fig = px.bar(comp_df, x="value", y="component", orientation="h", range_x=[0, 1],
                     title="Demand vs Supplier vs Delivery vs Stockout risk", color="value",
                     color_continuous_scale=["#2ca02c", "#ff7f0e", "#d62728"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Overall risk breakdown (weighted contribution)")
    contrib_df = pd.DataFrame({
        "component": list(result["components"].keys()),
        "weighted_contribution": [result["components"][k] * WEIGHTS[k] for k in result["components"]],
    })
    st.plotly_chart(px.pie(contrib_df, values="weighted_contribution", names="component",
                            title="Share of overall risk score by component"), use_container_width=True)

    st.session_state.update({"demand_r": demand_r, "supplier_r": supplier_r, "delivery_r": delivery_r,
                              "expected_demand": expected_demand, "inventory": inventory})

# ============================================================ 6. WHAT-IF
elif page.startswith("6"):
    st.title("What-If Scenario Analysis")
    note("Effects and relative costs of each intervention are documented SIMULATION ASSUMPTIONS "
         "(docs/assumptions.md) — not learned or causally validated effects.")
    demand_r = st.session_state.get("demand_r", 0.3)
    supplier_r = st.session_state.get("supplier_r", 0.3)
    delivery_r = st.session_state.get("delivery_r", 0.3)
    expected_demand = st.session_state.get("expected_demand", 100)
    inventory = st.session_state.get("inventory", 80)
    st.write(f"Using the current situation from the **Integrated Risk** tab: demand_risk={demand_r:.2f}, "
             f"supplier_risk={supplier_r:.2f}, delivery_risk={delivery_r:.2f}, "
             f"expected_demand={expected_demand}, inventory={inventory}")

    cond_delay = st.session_state.get("conditional_delay_days")
    if cond_delay is not None:
        note(f"expected_delay_days below uses a REAL, order-specific E[delay|late] estimate "
             f"({cond_delay:.1f}d) from the Delivery Prediction tab's last order, combined with "
             f"each scenario's P(late).")
    else:
        note("expected_delay_days below uses the real train-period historical average delay among "
             "late orders as a fallback (no specific order scored yet — visit Delivery Prediction to "
             "get an order-specific estimate instead).")

    res = run_scenarios(demand_r, supplier_r, delivery_r, expected_demand, inventory,
                         conditional_delay_days=cond_delay)
    st.session_state["scenario_result"] = res
    sdf = scenarios_to_dataframe(res)

    st.dataframe(
        sdf.style.apply(lambda row: ["background-color: #e6f4ea" if row["is_recommended"] else "" for _ in row], axis=1),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(sdf, x="scenario", y="overall_risk", color="is_recommended",
                     color_discrete_map={True: "#2ca02c", False: "#7f7f7f"},
                     title="Overall risk by scenario (recommended in green)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(sdf, x="relative_cost", y="overall_risk", text="scenario", size=[20]*len(sdf),
                          color="is_recommended", color_discrete_map={True: "#2ca02c", False: "#7f7f7f"},
                          title="Risk vs simulated relative cost (bottom-left is best)")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(sdf, x="scenario", y="expected_delay_days", color="is_recommended",
                     color_discrete_map={True: "#2ca02c", False: "#7f7f7f"},
                     title="Simulated expected delay (days) by scenario")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(sdf, x="scenario", y="shortage_units", color="is_recommended",
                     color_discrete_map={True: "#2ca02c", False: "#7f7f7f"},
                     title="Simulated shortage (units) by scenario")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================ 7. RECOMMENDATION
elif page.startswith("7"):
    st.title("Adaptive Recommendation")
    res = st.session_state.get("scenario_result")
    if not res:
        st.warning("Run the What-If Analysis tab first.")
    else:
        rec = res["recommendation"]
        sdf = scenarios_to_dataframe(res)
        st.success(f"### Recommended action: **{rec['recommended_action'].replace('_',' ').title()}**")
        st.write("**Reasons (traceable to the scenario metrics computed in the What-If tab):**")
        for r in rec["reason"]:
            st.write(f"- {r}")
        note(rec["decision_score_formula"])

        st.subheader("Why this scenario has the lowest decision score")
        fig = px.bar(sdf, x="scenario", y="decision_score", color="is_recommended",
                     color_discrete_map={True: "#2ca02c", False: "#7f7f7f"},
                     title="Decision score by scenario (lower = better)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("All alternatives considered, ranked")
        st.dataframe(sdf, use_container_width=True)

# ============================================================ 8. EXPLAINABILITY
elif page.startswith("8"):
    st.title("Explainable AI (SHAP)")
    note("SHAP explains what the trained model attributed a prediction to. It does NOT prove real-world causality.")

    model_choice = st.radio("Model to explain", ["Supplier Risk (time-aware)", "Delivery Risk"], horizontal=True)

    if model_choice.startswith("Supplier"):
        bundle = load_supplier_model_temporal()
        cols = bundle["feature_cols"]
        sample = get_seller_profile_temporal_df()[cols].apply(lambda c: c.fillna(c.median())).sample(
            min(200, len(scored_sellers)), random_state=1)
        sv = supplier_risk_shap(sample)
        imp = global_importance(sv, cols)
        st.subheader("Global feature importance (mean |SHAP value|)")
        st.plotly_chart(px.bar(imp, x="mean_abs_shap", y="feature", orientation="h"), use_container_width=True)

        idx = st.slider("Pick a sampled seller row to explain locally", 0, len(sample) - 1, 0)
        st.subheader(f"Local explanation for row {idx}")
        row_shap = pd.DataFrame({"feature": cols, "shap_value": sv.values[idx], "value": sample.iloc[idx].values})
        st.plotly_chart(px.bar(row_shap.sort_values("shap_value", key=abs, ascending=False),
                                x="shap_value", y="feature", orientation="h",
                                title="Local SHAP contribution to this seller's predicted risk"), use_container_width=True)
    else:
        order = st.session_state.get("last_order_input")
        if not order:
            st.warning("Go to the Delivery Prediction tab and run a prediction first, so we have an order to explain.")
        else:
            bundle = load_delivery_model()
            cols, num_feats, cat_feats = bundle["feature_cols"], bundle["numeric_features"], bundle["categorical_features"]
            df = pd.DataFrame([order])[num_feats + cat_feats]
            df_enc = pd.get_dummies(df, columns=cat_feats, drop_first=True).reindex(columns=cols, fill_value=0)
            sv = delivery_risk_shap(df_enc)
            row_shap = pd.DataFrame({"feature": cols, "shap_value": sv.values[0]})
            row_shap = row_shap[row_shap["shap_value"] != 0].sort_values("shap_value", key=abs, ascending=False).head(15)
            st.subheader("Top contributing factors for this order's delay prediction")
            st.plotly_chart(px.bar(row_shap, x="shap_value", y="feature", orientation="h"), use_container_width=True)

# ============================================================ 9. DATA / MODEL INFO
elif page.startswith("9"):
    st.title("Data & Model Information")
    st.subheader("Dataset")
    st.markdown("""
- **Source:** Olist Brazilian E-Commerce Public Dataset (real, anonymized commercial data; ~99,441 orders, 2016-2018).
- Retrieved via a public GitHub mirror of the canonical Kaggle dataset (row counts verified against the known
  Olist schema during preprocessing).
""")
    st.subheader("Model metrics (from actually-run training scripts)")
    st.json(metrics)
    st.subheader("Risk engine configuration")
    st.json({"weights": WEIGHTS, "thresholds": RISK_THRESHOLDS})
