"""
Module 9 - Explainable AI (SHAP)
=================================
Provides global feature-importance and local (single-prediction) SHAP
explanations for the supplier-risk and delivery-risk XGBoost classifiers.

IMPORTANT: SHAP values explain what the trained model attributed a
prediction to (model attribution). They do NOT prove real-world causality -
this caveat is shown in the dashboard wherever SHAP output is displayed.
"""
import shap
import pandas as pd
import numpy as np
from app.utils.predictors import load_supplier_model, load_supplier_model_temporal, load_delivery_model


def _explainer_for(model):
    return shap.TreeExplainer(model)


def supplier_risk_shap(sample_df: pd.DataFrame, temporal: bool = True):
    """sample_df: rows with columns matching the chosen supplier model's feature_cols.
    temporal=True (default) explains Model B (time-aware, the dashboard's primary model,
    8 features). temporal=False explains Model A (legacy, same-period, 4 features).
    IMPORTANT: sample_df's columns must match the chosen model - mixing them raises an
    XGBoost column-count error, which is exactly the bug this parameter fixes."""
    bundle = load_supplier_model_temporal() if temporal else load_supplier_model()
    model = bundle["model"]
    explainer = _explainer_for(model)
    sv = explainer(sample_df)
    return sv


def delivery_risk_shap(sample_df_encoded: pd.DataFrame):
    """sample_df_encoded: rows already one-hot encoded to match delivery model's feature_cols."""
    bundle = load_delivery_model()
    model = bundle["model"]
    explainer = _explainer_for(model)
    sv = explainer(sample_df_encoded)
    return sv


def global_importance(shap_values, feature_names):
    """Mean absolute SHAP value per feature -> simple global importance ranking."""
    vals = np.abs(shap_values.values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": vals})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
