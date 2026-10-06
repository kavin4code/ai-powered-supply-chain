"""
Test suite covering: preprocessing outputs, saved models load & predict,
risk engine math, and the scenario/recommendation engine.

Run:  python3 -m pytest tests/ -v
(assumes app/preprocessing/build_dataset.py and the three train_*.py scripts
have already been run once, so data/processed/*.csv and models/*.pkl exist)
"""
import os
import sys
import json
import pandas as pd
import pytest

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE)

from app.risk_engine.risk_engine import (
    integrated_risk, stockout_risk, demand_risk_from_volatility, WEIGHTS,
)
from app.scenarios.whatif import run_scenarios, scenarios_to_dataframe
from app.utils.predictors import (
    predict_supplier_risk_proba, predict_supplier_risk_proba_temporal, predict_delivery_late_proba,
    predict_demand_next, predict_conditional_delay_days, get_observed_avg_delay_given_late_days,
    get_seller_profile_df, get_seller_profile_temporal_df,
    get_category_daily_demand_df, get_orders_master_df,
)


# ---------------------------------------------------------------- preprocessing
def test_processed_files_exist():
    for f in ["orders_master.csv", "seller_profile.csv", "category_daily_demand.csv", "data_audit_report.md"]:
        assert os.path.exists(os.path.join(BASE, "data", "processed", f)), f"missing {f}"


def test_orders_master_shape_and_no_full_duplicates():
    df = get_orders_master_df()
    assert len(df) > 90000, "expected ~99k Olist orders"
    assert df["order_id"].duplicated().sum() == 0, "order_id should be unique per row"


def test_no_target_leakage_columns_used_in_delivery_training():
    import inspect
    from app.delivery import train_delivery_model as tdm
    src = inspect.getsource(tdm)
    assert '"order_delivered_customer_date"' not in tdm.NUMERIC_FEATURES if hasattr(tdm, "NUMERIC_FEATURES") else True
    assert "delivery_delay_days" not in tdm.NUMERIC_FEATURES
    assert "order_delivered_customer_date" not in tdm.NUMERIC_FEATURES


# ---------------------------------------------------------------- models load & predict
def test_models_exist():
    for f in ["demand_model.pkl", "supplier_risk_model.pkl", "delivery_model.pkl"]:
        assert os.path.exists(os.path.join(BASE, "models", f)), f"missing trained model {f}"


def test_supplier_risk_prediction_in_range():
    proba = predict_supplier_risk_proba({"n_orders": 50, "avg_freight": 20.0, "avg_price": 90.0, "n_categories": 3})
    assert 0.0 <= proba <= 1.0


def test_delivery_risk_prediction_in_range():
    order = {
        "estimated_days_to_deliver": 15, "n_items": 1, "item_price_total": 100.0, "freight_value_total": 20.0,
        "avg_item_price": 100.0, "n_distinct_sellers": 1, "n_payment_installments": 2, "payment_value_total": 120.0,
        "product_weight_g": 500, "product_length_cm": 20, "product_height_cm": 10, "product_width_cm": 15,
        "purchase_month": 6, "purchase_dow": 2, "cross_state_shipment": 1,
        "customer_state": "RJ", "seller_state": "SP", "payment_type_primary": "credit_card",
    }
    proba = predict_delivery_late_proba(order)
    assert 0.0 <= proba <= 1.0


def test_temporal_supplier_files_and_model_exist():
    assert os.path.exists(os.path.join(BASE, "data", "processed", "seller_profile_temporal.csv"))
    assert os.path.exists(os.path.join(BASE, "models", "supplier_risk_model_temporal.pkl"))
    assert os.path.exists(os.path.join(BASE, "models", "supplier_risk_model_temporal_metrics.json"))


def test_temporal_supplier_features_and_label_windows_do_not_overlap():
    """The time-aware label (seller_high_risk_future) must be built strictly from a later
    order window than the features (*_hist) - the core methodological claim of Model B."""
    import inspect
    from app.supplier_risk import build_temporal_features as btf
    src = inspect.getsource(btf.build)
    assert "hist = df[df[\"order_purchase_timestamp\"] < cutoff]" in src
    assert "future = df[df[\"order_purchase_timestamp\"] >= cutoff]" in src


def test_temporal_supplier_risk_prediction_in_range():
    df = get_seller_profile_temporal_df()
    row = df.iloc[0].to_dict()
    proba = predict_supplier_risk_proba_temporal(row)
    assert 0.0 <= proba <= 1.0


def test_demand_forecast_nonnegative():
    demand_df = get_category_daily_demand_df()
    cat = demand_df["category"].value_counts().idxmax()
    recent = demand_df[demand_df["category"] == cat].sort_values("purchase_date")["order_count"].tail(7).tolist()
    fc = predict_demand_next(cat, recent)
    assert fc >= 0


# ---------------------------------------------------------------- risk engine
def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_stockout_risk_bounds():
    assert stockout_risk(100, 200) == 0.0  # surplus inventory -> no shortage risk
    assert stockout_risk(100, 0) == 1.0    # zero inventory -> maximum shortage risk
    assert 0.0 < stockout_risk(100, 50) < 1.0


def test_integrated_risk_categorization():
    low = integrated_risk(0.0, 0.0, 0.0, 0.0)
    high = integrated_risk(1.0, 1.0, 1.0, 1.0)
    assert low["category"] == "Low Risk"
    assert high["category"] == "High Risk"
    assert low["overall_score"] < high["overall_score"]


def test_demand_volatility_risk_bounds():
    r = demand_risk_from_volatility([1, 2, 1, 2, 1, 2, 1])
    assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------- scenarios & recommendation
def test_scenario_engine_produces_all_four_actions():
    res = run_scenarios(0.3, 0.6, 0.5, expected_demand_units=100, available_inventory_units=40)
    assert set(res["scenarios"].keys()) == {"do_nothing", "switch_supplier", "expedite_delivery", "increase_inventory"}


def test_recommendation_is_traceable_and_valid_action():
    res = run_scenarios(0.3, 0.8, 0.7, expected_demand_units=100, available_inventory_units=10)
    rec = res["recommendation"]
    assert rec["recommended_action"] in res["scenarios"]
    assert len(rec["reason"]) >= 1


def test_increase_inventory_reduces_stockout_risk_vs_do_nothing():
    res = run_scenarios(0.3, 0.5, 0.5, expected_demand_units=100, available_inventory_units=20)
    assert res["scenarios"]["increase_inventory"]["stockout_risk"] <= res["scenarios"]["do_nothing"]["stockout_risk"]


def test_scenario_exposes_delay_and_shortage_impact():
    res = run_scenarios(0.3, 0.6, 0.5, expected_demand_units=100, available_inventory_units=40)
    fallback = get_observed_avg_delay_given_late_days()
    for action, m in res["scenarios"].items():
        assert "expected_delay_days" in m and 0 <= m["expected_delay_days"] <= fallback
        assert "shortage_units" in m and m["shortage_units"] >= 0


def test_expedite_delivery_reduces_expected_delay_vs_do_nothing():
    res = run_scenarios(0.3, 0.5, 0.7, expected_demand_units=100, available_inventory_units=50)
    assert res["scenarios"]["expedite_delivery"]["expected_delay_days"] < res["scenarios"]["do_nothing"]["expected_delay_days"]


def test_switch_supplier_reduces_supplier_risk_vs_do_nothing():
    res = run_scenarios(0.3, 0.8, 0.4, expected_demand_units=100, available_inventory_units=50)
    assert res["scenarios"]["switch_supplier"]["supplier_risk"] < res["scenarios"]["do_nothing"]["supplier_risk"]


# ---------------------------------------------------------------- delay-magnitude model (methodological review pass)
def test_delay_magnitude_model_and_metrics_exist():
    for f in ["delay_magnitude_model.pkl", "delay_magnitude_model_metrics.json"]:
        assert os.path.exists(os.path.join(BASE, "models", f)), f"missing {f}"


def test_delay_magnitude_model_beats_naive_baseline_on_held_out_data():
    """The core justification for using this model at all: it must actually beat a
    naive train-mean baseline on held-out (test-period) late orders, or the
    what-if engine should not treat it as better than a plain constant."""
    with open(os.path.join(BASE, "models", "delay_magnitude_model_metrics.json")) as f:
        m = json.load(f)
    assert m["xgboost_conditional_delay"]["MAE"] < m["baseline_train_mean"]["MAE"]


def test_delay_magnitude_model_uses_no_outcome_columns_as_predictors():
    """Leakage guard: delivery_delay_days and is_late must only be used as the
    label / row filter, never as predictors, in the delay-magnitude training script."""
    from app.delivery.train_delivery_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    assert "delivery_delay_days" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "is_late" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "order_delivered_customer_date" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_predict_conditional_delay_days_is_nonnegative():
    order = {
        "estimated_days_to_deliver": 15, "n_items": 1, "item_price_total": 100.0, "freight_value_total": 20.0,
        "avg_item_price": 100.0, "n_distinct_sellers": 1, "n_payment_installments": 2, "payment_value_total": 120.0,
        "product_weight_g": 500, "product_length_cm": 20, "product_height_cm": 10, "product_width_cm": 15,
        "purchase_month": 6, "purchase_dow": 2, "cross_state_shipment": 1,
        "customer_state": "RJ", "seller_state": "SP", "payment_type_primary": "credit_card",
    }
    days = predict_conditional_delay_days(order)
    assert days >= 0


def test_observed_avg_delay_given_late_is_a_real_train_period_statistic():
    """The fallback constant must be a real number derived from the train period (not an
    arbitrary guess like the old MAX_ASSUMED_DELAY_DAYS=6.0), and should be in a
    plausible range for this dataset's known delay distribution (single-digit-to-teens days)."""
    val = get_observed_avg_delay_given_late_days()
    assert 5.0 < val < 20.0


def test_scenario_engine_uses_real_order_specific_delay_when_provided():
    """When a real, order-specific conditional_delay_days is supplied, it must actually be
    used in the expected_delay_days calculation (not silently ignored in favour of the
    fallback), verified via the exact expected arithmetic for 'do_nothing'."""
    res = run_scenarios(0.3, 0.5, 0.6, expected_demand_units=100, available_inventory_units=50,
                         conditional_delay_days=9.0)
    assert res["delay_given_late_source"] == "order-specific model"
    assert res["delay_given_late_days_used"] == 9.0
    assert res["scenarios"]["do_nothing"]["expected_delay_days"] == round(0.6 * 9.0, 2)


def test_scenario_engine_falls_back_to_real_historical_average_when_no_order_given():
    res = run_scenarios(0.3, 0.5, 0.6, expected_demand_units=100, available_inventory_units=50)
    assert res["delay_given_late_source"] == "train-period historical average"
    assert res["delay_given_late_days_used"] == get_observed_avg_delay_given_late_days()


def test_scenarios_to_dataframe_shape_and_recommended_flag():
    res = run_scenarios(0.3, 0.6, 0.5, expected_demand_units=100, available_inventory_units=40)
    df = scenarios_to_dataframe(res)
    assert len(df) == 4
    assert df["is_recommended"].sum() == 1
    assert list(df.columns) == [
        "scenario", "is_recommended", "overall_risk", "risk_category", "supplier_risk", "delivery_risk",
        "expected_delay_days", "stockout_risk", "shortage_units", "service_level", "relative_cost", "decision_score",
    ]
    # dataframe should be sorted best (lowest decision_score) first
    assert df["decision_score"].is_monotonic_increasing


# ---------------------------------------------------------------- explainability (SHAP model-mismatch bugfix)
def test_supplier_shap_uses_matching_temporal_model_by_default():
    """Regression test for a bug found during the visualization upgrade pass: SHAP was
    hardcoded to the legacy 4-feature model even when given an 8-feature (temporal)
    sample, causing an XGBoost column-count crash. supplier_risk_shap() must default to
    the time-aware model and match its feature count."""
    from app.explainability.shap_utils import supplier_risk_shap
    bundle = load_supplier_model_temporal_bundle = __import__(
        "app.utils.predictors", fromlist=["load_supplier_model_temporal"]
    ).load_supplier_model_temporal()
    cols = bundle["feature_cols"]
    df = get_seller_profile_temporal_df()[cols].apply(lambda c: c.fillna(c.median())).head(10)
    sv = supplier_risk_shap(df)  # default temporal=True
    assert sv.values.shape == (10, len(cols))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
