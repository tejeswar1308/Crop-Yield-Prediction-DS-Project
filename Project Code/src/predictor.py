"""
Predictor Module
Dynamically validates inference input values against training statistics,
flags out-of-distribution values / extrapolation risks,
and executes pipeline predictions with plain-English contextual explanations.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class PredictionValidationError(Exception):
    """Raised when user input for inference fails basic validation."""
    pass


def validate_inference_inputs(
    inputs: Dict[str, Any],
    feature_stats: Dict[str, Dict[str, Any]],
    numeric_features: List[str],
    categorical_features: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates input values against training distributions.
    Flags any values that extrapolate outside the historical training bounds.
    Returns:
        (input_df, warnings_list)
    """
    row_data: Dict[str, Any] = {}
    warnings: List[str] = []

    for col in numeric_features:
        if col not in inputs:
            raise PredictionValidationError(f"Missing required numeric input for feature '{col}'.")
        try:
            val = float(inputs[col])
        except (ValueError, TypeError):
            raise PredictionValidationError(f"Invalid numeric value '{inputs[col]}' provided for '{col}'.")

        row_data[col] = val

        # Range bounds check
        stats = feature_stats.get(col, {})
        col_min = stats.get("min")
        col_max = stats.get("max")
        if col_min is not None and col_max is not None:
            if val < col_min:
                warnings.append(
                    f"'{col}' value ({val:.2f}) is below historical training minimum ({col_min:.2f}). "
                    "Model prediction is extrapolating into uncharted lower range."
                )
            elif val > col_max:
                warnings.append(
                    f"'{col}' value ({val:.2f}) exceeds historical training maximum ({col_max:.2f}). "
                    "Model prediction is extrapolating into uncharted upper range."
                )

    for col in categorical_features:
        if col not in inputs:
            raise PredictionValidationError(f"Missing required categorical input for feature '{col}'.")
        cat_val = str(inputs[col]).strip()
        row_data[col] = cat_val

        # Category check
        stats = feature_stats.get(col, {})
        known_cats = stats.get("categories", [])
        if known_cats and cat_val not in known_cats:
            warnings.append(
                f"'{col}' category '{cat_val}' was not observed during training. "
                "The pipeline will impute or treat it as an unknown category."
            )

    input_df = pd.DataFrame([row_data])
    return input_df, warnings


def predict_single_instance(
    pipeline: Pipeline,
    inputs: Dict[str, Any],
    feature_stats: Dict[str, Dict[str, Any]],
    numeric_features: List[str],
    categorical_features: List[str],
    yield_unit: str = "tonnes/hectare"
) -> Dict[str, Any]:
    """
    Validates inputs, executes inference through the fitted pipeline,
    and returns predicted yield, confidence warnings, and formatted explanation.
    """
    input_df, warnings = validate_inference_inputs(
        inputs, feature_stats, numeric_features, categorical_features
    )

    pred_raw = pipeline.predict(input_df)[0]
    # Physical constraint: crop yield cannot be strictly negative in real-world agriculture
    predicted_yield = max(0.0, float(pred_raw))

    return {
        "predicted_yield": round(predicted_yield, 3),
        "raw_prediction": float(pred_raw),
        "yield_unit": yield_unit,
        "warnings": warnings,
        "input_features_used": inputs,
    }
