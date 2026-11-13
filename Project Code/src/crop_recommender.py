"""
Crop Recommender Module
Compares all supported candidate crops under uniform environmental, soil, and regional conditions.
Ranks candidates from highest to lowest expected yield, produces interactive comparisons,
and enforces agronomic & economic caveats.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline


class CropRecommendationError(Exception):
    """Raised when crop recommendation cannot be executed."""
    pass


def can_support_crop_recommendation(
    bundle: Dict[str, Any],
    crop_col: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Checks whether the trained pipeline bundle includes a valid Crop feature.
    Returns: (is_supported, explanation_message, detected_crop_col)
    """
    feature_cols = bundle.get("feature_cols", [])
    cat_features = bundle.get("categorical_features", [])

    if crop_col and crop_col in feature_cols:
        return True, "Crop column identified.", crop_col

    # Look for candidate crop column in categorical features
    for col in cat_features:
        if "crop" in col.lower() or "commodity" in col.lower() or col.lower() in ["crop", "crop_name", "crop_type"]:
            return True, f"Found crop feature '{col}'.", col

    return (
        False,
        "The trained model does not contain a 'Crop' identity feature among its input columns. "
        "To enable crop comparison and ranking, ensure your uploaded dataset has a crop name column "
        "(e.g. 'Crop', 'Crop_Type') and include it in Model Training features.",
        None
    )


def compare_and_rank_crops(
    pipeline: Pipeline,
    base_conditions: Dict[str, Any],
    crop_col: str,
    feature_stats: Dict[str, Dict[str, Any]],
    numeric_features: List[str],
    categorical_features: List[str],
    candidate_crops: Optional[List[str]] = None,
    yield_unit: str = "tonnes/hectare"
) -> Dict[str, Any]:
    """
    Evaluates every candidate crop under identical base conditions.
    Ranks candidates by predicted yield descending.
    """
    if candidate_crops is None or len(candidate_crops) == 0:
        crop_stats = feature_stats.get(crop_col, {})
        candidate_crops = crop_stats.get("categories", [])

    if not candidate_crops:
        raise CropRecommendationError(
            f"No candidate crops found in the training data for feature '{crop_col}'."
        )

    results: List[Dict[str, Any]] = []

    for crop in candidate_crops:
        test_input = dict(base_conditions)
        test_input[crop_col] = str(crop)

        # Build single row DataFrame
        row_dict: Dict[str, Any] = {}
        for col in numeric_features:
            row_dict[col] = float(test_input.get(col, feature_stats.get(col, {}).get("median", 0.0)))
        for col in categorical_features:
            row_dict[col] = str(test_input.get(col, "Unknown"))

        df_row = pd.DataFrame([row_dict])
        pred_val = max(0.0, float(pipeline.predict(df_row)[0]))

        results.append({
            "Crop": str(crop),
            "Predicted Yield": round(pred_val, 3),
            "Unit": yield_unit,
        })

    results_df = pd.DataFrame(results).sort_values("Predicted Yield", ascending=False).reset_index(drop=True)
    results_df["Rank"] = results_df.index + 1

    top_crop = results_df.iloc[0]["Crop"]
    top_yield = results_df.iloc[0]["Predicted Yield"]

    return {
        "ranking_df": results_df,
        "top_crop": top_crop,
        "top_yield": top_yield,
        "yield_unit": yield_unit,
        "total_crops_compared": len(results_df),
        "base_conditions": base_conditions,
        "crop_column": crop_col,
    }


def plot_crop_ranking_chart(
    ranking_df: pd.DataFrame,
    yield_unit: str = "tonnes/hectare"
) -> go.Figure:
    """
    Creates a modern bar chart comparing predicted crop yields,
    highlighting the #1 best recommended crop.
    """
    df = ranking_df.copy()
    top_crop = df.iloc[0]["Crop"]

    # Assign distinct colors: gold/emerald for top crop, cool blue/green for others
    colors = ["#2ca02c" if c == top_crop else "#4575b4" for c in df["Crop"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Crop"],
        y=df["Predicted Yield"],
        text=df["Predicted Yield"].apply(lambda y: f"{y:.2f}"),
        textposition="outside",
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Rank: %{customdata}<br>Predicted Yield: %{y:.2f} " + yield_unit + "<extra></extra>",
        customdata=df["Rank"],
    ))

    fig.update_layout(
        title=f"Crop Yield Comparison & Ranking (Recommended: {top_crop})",
        xaxis_title="Candidate Crop",
        yaxis_title=f"Expected Yield ({yield_unit})",
        template="plotly_white",
        xaxis_tickangle=-45,
        height=450,
        margin=dict(l=50, r=40, t=60, b=90),
    )
    return fig
