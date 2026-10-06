"""
Module 5 - Integrated Supply-Chain Risk Engine
================================================
Combines four component risk scores (each normalized to 0-1) into one
transparent, weighted "Overall Risk" score.

    Overall Risk = w_demand   * Demand_Risk
                 + w_supplier * Supplier_Risk
                 + w_delivery * Delivery_Risk
                 + w_stockout * Stockout_Risk

WEIGHTS ARE A DESIGN ASSUMPTION (see docs/assumptions.md), not something
learned or scientifically validated. They are exposed as constants below so
they can be inspected/adjusted, and the dashboard labels them as such.

Category thresholds (Low / Medium / High) are PROTOTYPE DECISION
THRESHOLDS chosen for demo interpretability, not statistically derived.
"""

# ---- ASSUMPTION: component weights (must sum to 1.0) ----
WEIGHTS = {
    "demand_risk": 0.20,
    "supplier_risk": 0.30,
    "delivery_risk": 0.30,
    "stockout_risk": 0.20,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# ---- ASSUMPTION: prototype risk-category thresholds ----
RISK_THRESHOLDS = {"low": 0.33, "medium": 0.66}  # score < low -> Low, < medium -> Medium, else High


def demand_risk_from_volatility(recent_values):
    """Higher day-to-day volatility (coefficient of variation) relative to the mean => higher demand risk.
    recent_values: list/array of recent daily demand counts for a category."""
    import numpy as np
    v = np.asarray(recent_values, dtype=float)
    if len(v) < 2 or v.mean() == 0:
        return 0.5  # insufficient history -> neutral/medium-ish, documented as a fallback assumption
    cv = v.std() / (v.mean() + 1e-6)
    return float(min(cv / 1.5, 1.0))  # cv of 1.5+ treated as maximum risk (assumption)


def normalize_supplier_risk(proba_high_risk):
    """Supplier risk model already outputs a 0-1 probability of high risk -> used directly."""
    return float(max(0.0, min(1.0, proba_high_risk)))


def normalize_delivery_risk(proba_late):
    """Delivery model already outputs a 0-1 probability of lateness -> used directly."""
    return float(max(0.0, min(1.0, proba_late)))


def stockout_risk(expected_demand_units, available_inventory_units):
    """SIMULATED inventory comparison. available_inventory_units may come from a user-set
    scenario slider, not historical ground truth (Olist has no inventory table)."""
    if expected_demand_units <= 0:
        return 0.0
    shortage = max(0, expected_demand_units - available_inventory_units)
    return float(min(shortage / expected_demand_units, 1.0))


def integrated_risk(demand_risk, supplier_risk, delivery_risk, stockout_risk_value):
    score = (
        WEIGHTS["demand_risk"] * demand_risk
        + WEIGHTS["supplier_risk"] * supplier_risk
        + WEIGHTS["delivery_risk"] * delivery_risk
        + WEIGHTS["stockout_risk"] * stockout_risk_value
    )
    if score < RISK_THRESHOLDS["low"]:
        category = "Low Risk"
    elif score < RISK_THRESHOLDS["medium"]:
        category = "Medium Risk"
    else:
        category = "High Risk"
    return {
        "overall_score": round(float(score), 4),
        "category": category,
        "components": {
            "demand_risk": round(demand_risk, 4),
            "supplier_risk": round(supplier_risk, 4),
            "delivery_risk": round(delivery_risk, 4),
            "stockout_risk": round(stockout_risk_value, 4),
        },
        "weights": WEIGHTS,
        "thresholds": RISK_THRESHOLDS,
    }
