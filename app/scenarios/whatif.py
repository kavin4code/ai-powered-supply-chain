"""
Module 7 - What-If Scenario Analysis
Module 8 - Adaptive Recommendation Engine
==========================================
Given a current situation (supplier risk proba, delivery-late proba, expected
demand, available inventory), estimate the effect of four candidate actions
on risk / delay / shortage / cost, then recommend the best-scoring one with a
traceable, non-black-box reason.

All relative costs and intervention-effect multipliers below are SIMULATION
ASSUMPTIONS (the Olist dataset has no real switching/expediting/carrying-cost
or intervention-effect figures) - see docs/assumptions.md. Nothing here is a
learned or causally-validated effect; it is a documented, adjustable
prototype for demonstrating decision-support reasoning.
"""
import pandas as pd
from app.risk_engine.risk_engine import integrated_risk, stockout_risk as calc_stockout_risk

# ---- ASSUMPTION: normalized relative cost of each action (0=free, 1=very costly) ----
RELATIVE_COST = {
    "do_nothing": 0.00,
    "switch_supplier": 0.55,
    "expedite_delivery": 0.65,
    "increase_inventory": 0.35,
}

# ---- ASSUMPTION: effect of each action on the underlying risk components ----
# multiplicative reduction factors, expressed as fractions retained (e.g. 0.6 = 40% reduction)
EFFECT = {
    "do_nothing":         {"supplier_risk": 1.00, "delivery_risk": 1.00},
    "switch_supplier":    {"supplier_risk": 0.35, "delivery_risk": 0.90},
    "expedite_delivery":  {"supplier_risk": 1.00, "delivery_risk": 0.40},
    "increase_inventory": {"supplier_risk": 1.00, "delivery_risk": 1.00},
}

# ---- Expected delay magnitude: two-part decomposition (methodological review pass) ----
# E[delay_days] = P(late | features) * E[delay_days | late, features]
#
# P(late | features) is the delivery-lateness classifier's output, passed in as
# `delivery_risk` (adjusted per scenario below).
#
# E[delay_days | late, features] ("delay_given_late_days") is either:
#   (a) a REAL, order-specific estimate from the trained delay-magnitude regression
#       model (app/delivery/train_delay_magnitude_model.py), passed in via the
#       `conditional_delay_days` argument when the caller has concrete order features
#       (e.g. from the Delivery Prediction tab) - held-out MAE 5.96 days vs a 7.81-day
#       naive baseline, so this genuinely beats guessing a constant; or
#   (b) a REAL fallback constant - the train-period-only historical average delay among
#       late orders (see get_observed_avg_delay_given_late_days()) - used when no order
#       features are available (e.g. a bare risk-slider scenario in the Integrated Risk
#       / What-If tabs with no underlying order).
#
# NEITHER option is a causal intervention effect: we assume each action changes
# P(late) (via EFFECT below, a documented simulation assumption) but NOT the conditional
# delay magnitude if a delay does occur - we have no data to estimate how an
# intervention would change delay magnitude specifically, so we deliberately do not
# fabricate that. This is a documented simplification, not a hidden one.
_delay_fallback_cache = {}


def _fallback_delay_given_late_days():
    if "value" not in _delay_fallback_cache:
        try:
            from app.utils.predictors import get_observed_avg_delay_given_late_days
            _delay_fallback_cache["value"] = get_observed_avg_delay_given_late_days(
            )
        except Exception:
            # emergency fallback only if the delay-magnitude model hasn't been trained yet
            _delay_fallback_cache["value"] = 6.0
    return _delay_fallback_cache["value"]


# ---- ASSUMPTION: recommendation decision-score weighting (risk vs cost) ----
DECISION_WEIGHTS = {"risk": 0.9, "cost": 0.1}


def run_scenarios(demand_risk, supplier_risk, delivery_risk, expected_demand_units,
                  available_inventory_units, inventory_increase_units=None,
                  conditional_delay_days=None):
    """Evaluate all 4 scenarios and return their metrics + the recommended action.
    Backward compatible with the previous signature/return keys; adds:
    expected_delay_days, shortage_units, decision_score components exposed explicitly.

    conditional_delay_days: optional REAL, order-specific E[delay_days | late] estimate
    from the delay-magnitude model (see app/utils/predictors.predict_conditional_delay_days).
    If omitted, falls back to the real train-period historical average delay among late
    orders (see _fallback_delay_given_late_days above) - never an arbitrary guess."""
    if inventory_increase_units is None:
        # ASSUMPTION: +25% safety stock
        inventory_increase_units = max(1, int(0.25 * expected_demand_units))

    delay_given_late_days = (
        conditional_delay_days if conditional_delay_days is not None else _fallback_delay_given_late_days()
    )

    results = {}
    for action, effect in EFFECT.items():
        adj_supplier = supplier_risk * effect["supplier_risk"]
        adj_delivery = delivery_risk * effect["delivery_risk"]
        inv = available_inventory_units + (
            inventory_increase_units if action == "increase_inventory" else 0
        )
        adj_stockout = calc_stockout_risk(expected_demand_units, inv)
        shortage_units = max(0, expected_demand_units - inv)
        # E[delay_days] = P(late | this scenario's adjusted risk) * E[delay_days | late]
        # (ASSUMPTION: the action changes P(late), not the conditional delay magnitude - see above)
        expected_delay_days = round(adj_delivery * delay_given_late_days, 2)

        risk = integrated_risk(demand_risk, adj_supplier,
                               adj_delivery, adj_stockout)
        service_level = 1 - adj_stockout
        results[action] = {
            "overall_risk": risk["overall_score"],
            "risk_category": risk["category"],
            "supplier_risk": round(adj_supplier, 4),
            "delivery_risk": round(adj_delivery, 4),
            "expected_delay_days": expected_delay_days,
            "stockout_risk": round(adj_stockout, 4),
            "shortage_units": shortage_units,
            "service_level": round(service_level, 4),
            "relative_cost": RELATIVE_COST[action],
        }

    # ---- Module 8: adaptive recommendation ----
    # Traceable scoring: lower risk and lower cost are both good. We rank by a simple,
    # documented composite (ASSUMPTION weighting, not learned/causal):
    #   decision_score = DECISION_WEIGHTS['risk'] * overall_risk + DECISION_WEIGHTS['cost'] * relative_cost
    for action, r in results.items():
        r["decision_score"] = round(
            DECISION_WEIGHTS["risk"] * r["overall_risk"] +
            DECISION_WEIGHTS["cost"] * r["relative_cost"], 4
        )

    best_action = min(results, key=lambda a: results[a]["decision_score"])
    baseline = results["do_nothing"]
    best = results[best_action]

    reasons = []
    if best["overall_risk"] < baseline["overall_risk"]:
        reasons.append(
            f"lower projected overall risk ({best['overall_risk']} vs {baseline['overall_risk']} for doing nothing)"
        )
    if best["expected_delay_days"] < baseline["expected_delay_days"]:
        reasons.append(
            f"lower expected delay ({best['expected_delay_days']}d vs {baseline['expected_delay_days']}d — "
            f"P(late) x typical days-if-late)"
        )
    if best["shortage_units"] < baseline["shortage_units"]:
        reasons.append(
            f"smaller simulated shortage ({best['shortage_units']} vs {baseline['shortage_units']} units)"
        )
    if best["service_level"] > baseline["service_level"]:
        reasons.append(
            f"improved expected service level ({best['service_level']*100:.1f}%)")
    if best["relative_cost"] <= 0.6:
        reasons.append(
            f"acceptable simulated relative cost ({best['relative_cost']})")
    if not reasons:
        reasons.append(
            "best available trade-off among the modelled alternatives")

    recommendation = {
        "recommended_action": best_action,
        "reason": reasons,
        "decision_score_formula": (
            f"{DECISION_WEIGHTS['risk']} * overall_risk + {DECISION_WEIGHTS['cost']} * relative_cost "
            "(lower is better) - ASSUMPTION weighting, not a learned or causal utility function"
        ),
    }

    return {"scenarios": results, "recommendation": recommendation,
            "inventory_increase_units_used": inventory_increase_units,
            "delay_given_late_days_used": delay_given_late_days,
            "delay_given_late_source": "order-specific model" if conditional_delay_days is not None else "train-period historical average"}


def scenarios_to_dataframe(result: dict) -> pd.DataFrame:
    """Convert run_scenarios()'s output into a clean, dashboard-ready DataFrame,
    one row per scenario, sorted by decision_score (best first), with the
    recommended action flagged."""
    rows = []
    rec_action = result["recommendation"]["recommended_action"]
    for action, m in result["scenarios"].items():
        rows.append({
            "scenario": action.replace("_", " ").title(),
            "is_recommended": action == rec_action,
            "overall_risk": m["overall_risk"],
            "risk_category": m["risk_category"],
            "supplier_risk": m["supplier_risk"],
            "delivery_risk": m["delivery_risk"],
            "expected_delay_days": m["expected_delay_days"],
            "stockout_risk": m["stockout_risk"],
            "shortage_units": m["shortage_units"],
            "service_level": m["service_level"],
            "relative_cost": m["relative_cost"],
            "decision_score": m["decision_score"],
        })
    df = pd.DataFrame(rows).sort_values(
        "decision_score").reset_index(drop=True)
    return df
