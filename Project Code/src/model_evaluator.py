"""
Model Evaluator Module
Provides comprehensive diagnostics, validation visualizations,
actual vs. predicted comparisons, residual plots, and agricultural domain limitations.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


THEME_TEMPLATE = "plotly_white"


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "Yield",
    model_name: str = "Best Model"
) -> go.Figure:
    """
    Creates an interactive scatter plot of actual vs. predicted values with an ideal 45-degree line.
    """
    df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})

    min_val = min(float(df["Actual"].min()), float(df["Predicted"].min()))
    max_val = max(float(df["Actual"].max()), float(df["Predicted"].max()))
    padding = (max_val - min_val) * 0.05
    line_min = min_val - padding
    line_max = max_val + padding

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=df["Actual"],
        y=df["Predicted"],
        mode="markers",
        name="Observations",
        marker=dict(
            color="#2b83ba",
            size=7,
            opacity=0.65,
            line=dict(width=0.5, color="#1a476f")
        ),
        hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"
    ))

    # Ideal line
    fig.add_trace(go.Scatter(
        x=[line_min, line_max],
        y=[line_min, line_max],
        mode="lines",
        name="Ideal Fit (y = x)",
        line=dict(color="#d7191c", dash="dash", width=2),
    ))

    fig.update_layout(
        title=f"Actual vs. Predicted {target_name} ({model_name})",
        xaxis_title=f"Actual {target_name}",
        yaxis_title=f"Predicted {target_name}",
        template=THEME_TEMPLATE,
        height=450,
        margin=dict(l=50, r=40, t=50, b=50),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
    )

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "Yield"
) -> go.Figure:
    """
    Plots residuals (y_true - y_pred) against predicted values.
    """
    residuals = y_true - y_pred
    df = pd.DataFrame({"Predicted": y_pred, "Residual": residuals})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Predicted"],
        y=df["Residual"],
        mode="markers",
        name="Residuals",
        marker=dict(
            color="#abdda4",
            size=7,
            opacity=0.7,
            line=dict(width=0.5, color="#2b83ba")
        ),
        hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"
    ))

    # Zero residual line
    fig.add_hline(y=0, line_dash="dash", line_color="#d7191c")

    fig.update_layout(
        title=f"Residual Plot ({target_name})",
        xaxis_title=f"Predicted {target_name}",
        yaxis_title="Residual (Actual - Predicted)",
        template=THEME_TEMPLATE,
        height=400,
        margin=dict(l=50, r=40, t=50, b=50)
    )
    return fig


def plot_feature_importance_chart(
    importances: List[Dict[str, Any]],
    top_n: int = 15
) -> Optional[go.Figure]:
    """
    Plots horizontal bar chart of top relative feature importances.
    """
    if not importances:
        return None

    top_imps = importances[:top_n]
    df = pd.DataFrame(top_imps).sort_values("importance", ascending=True)

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {len(df)} Feature Importances",
        labels={"importance": "Relative Importance / Magnitude", "feature": "Feature"},
        template=THEME_TEMPLATE,
        color="importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=max(380, len(df) * 24), margin=dict(l=120, r=40, t=50, b=50))
    return fig


def create_models_comparison_table(models_comparison: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Formats multi-model test metrics into a clean pandas DataFrame.
    """
    records = []
    for name, data in models_comparison.items():
        test_m = data["test_metrics"]
        train_m = data["train_metrics"]
        records.append({
            "Model": name,
            "Test R²": test_m["R2"],
            "Test MAE": test_m["MAE"],
            "Test RMSE": test_m["RMSE"],
            "Train R²": train_m["R2"],
            "Train MAE": train_m["MAE"],
            "Train RMSE": train_m["RMSE"],
        })

    df = pd.DataFrame(records).sort_values("Test R²", ascending=False).reset_index(drop=True)
    return df


def plot_model_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart comparing models across R², MAE, and RMSE.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Test R²",
        x=comparison_df["Model"],
        y=comparison_df["Test R²"],
        marker_color="#2ca02c"
    ))
    fig.add_trace(go.Bar(
        name="Test RMSE",
        x=comparison_df["Model"],
        y=comparison_df["Test RMSE"],
        marker_color="#1f77b4"
    ))
    fig.add_trace(go.Bar(
        name="Test MAE",
        x=comparison_df["Model"],
        y=comparison_df["Test MAE"],
        marker_color="#ff7f0e"
    ))

    fig.update_layout(
        barmode="group",
        title="Multi-Model Performance Comparison (Held-out Test Set)",
        template=THEME_TEMPLATE,
        xaxis_tickangle=-25,
        height=400,
        margin=dict(l=40, r=40, t=50, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def generate_limitations_and_warnings(bundle: Dict[str, Any]) -> List[str]:
    """
    Generates tailored, scientifically accurate limitations and agricultural caveats.
    """
    warnings: List[str] = []
    metadata = bundle["metadata"]
    metrics = bundle["best_metrics"]
    split_info = bundle["split_info"]
    feature_cols = bundle["feature_cols"]

    # 1. Sample size checks
    train_rows = metadata["train_rows"]
    if train_rows < 100:
        warnings.append(
            f"Limited Dataset Size: Trained on only {train_rows} records. Small sample sizes increase variance and the risk of overfitting."
        )

    # 2. Predictive fit checks
    if metrics["R2"] < 0.50:
        warnings.append(
            f"Moderate Explanatory Power (R² = {metrics['R2']:.2f}): A significant portion of yield variability is influenced by unmeasured factors (e.g. soil microbiome, pest pressures, extreme weather events)."
        )

    # 3. Train vs Test gap (Overfitting)
    best_name = bundle["best_model_name"]
    comp = bundle["models_comparison"][best_name]
    train_r2 = comp["train_metrics"]["R2"]
    test_r2 = comp["test_metrics"]["R2"]
    if (train_r2 - test_r2) > 0.25:
        warnings.append(
            f"Potential Overfitting: Training R² ({train_r2:.2f}) is noticeably higher than Test R² ({test_r2:.2f}). Predictions on unseen conditions may exhibit reduced accuracy."
        )

    # 4. Domain / Weather checks
    has_weather = any(c.lower() in ["rainfall", "rain", "temperature", "temp"] for c in feature_cols)
    if not has_weather:
        warnings.append(
            "Missing Meteorological Data: Weather/rainfall is not included in the feature set. Agricultural yields are highly sensitive to in-season precipitation and heat stress."
        )

    # 5. Temporal caveat
    if split_info.get("method") != "Chronological":
        warnings.append(
            "Temporal Generalization: Random train/test split was used. Future climate shifts and technological farming advancements across years may alter yield dynamics."
        )

    # General physical constraint
    warnings.append(
        "Agronomic Feasibility: Predictions are statistical point estimates based strictly on historical data distributions. Local irrigation management, fertilizer application timing, and soil health directly alter realized yields."
    )

    return warnings
