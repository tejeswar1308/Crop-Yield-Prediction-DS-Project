"""
Exploratory Data Analysis (EDA) Module
Generates dynamic, interactive Plotly visualizations based solely on the uploaded dataset.
Handles missing optional columns gracefully without hardcoding.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


THEME_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Plotly


def plot_missing_values_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Plots missing values count and percentage per column.
    """
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": null_counts.values,
        "Missing %": null_pct.values,
    }).sort_values("Missing Count", ascending=False)

    # Filter to columns that actually have missing values or show top 10
    has_missing = missing_df[missing_df["Missing Count"] > 0]
    display_df = has_missing if not has_missing.empty else missing_df.head(10)

    fig = px.bar(
        display_df,
        x="Column",
        y="Missing Count",
        text="Missing %",
        title="Missing Values Analysis per Column" if not has_missing.empty else "Missing Values (0% missing across dataset)",
        color="Missing %",
        color_continuous_scale="Reds",
        template=THEME_TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, height=380, margin=dict(l=40, r=40, t=50, b=80))
    return fig


def plot_crop_distribution(df: pd.DataFrame, crop_col: str, top_n: int = 15) -> Optional[go.Figure]:
    """
    Plots the frequency distribution of crop types in the dataset.
    """
    if crop_col not in df.columns:
        return None

    counts = df[crop_col].value_counts().head(top_n).reset_index()
    counts.columns = [crop_col, "Count"]

    fig = px.bar(
        counts,
        x=crop_col,
        y="Count",
        title=f"Top {len(counts)} Most Frequent Crops",
        color="Count",
        color_continuous_scale="Greens",
        template=THEME_TEMPLATE,
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, height=400, margin=dict(l=40, r=40, t=50, b=80))
    return fig


def plot_crop_yield_comparison(
    df: pd.DataFrame,
    crop_col: str,
    yield_col: str,
    top_n: int = 12
) -> Optional[go.Figure]:
    """
    Generates box plots comparing yield distributions across candidate crops.
    """
    if crop_col not in df.columns or yield_col not in df.columns:
        return None

    # Filter to top crops by frequency to keep chart readable
    top_crops = df[crop_col].value_counts().head(top_n).index.tolist()
    sub_df = df[df[crop_col].isin(top_crops)].dropna(subset=[yield_col])

    if sub_df.empty:
        return None

    fig = px.box(
        sub_df,
        x=crop_col,
        y=yield_col,
        color=crop_col,
        title=f"Yield Distribution Across Top {len(top_crops)} Crops",
        template=THEME_TEMPLATE,
        points="outliers",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=450, margin=dict(l=40, r=40, t=50, b=90))
    return fig


def plot_historical_yield_trends(
    df: pd.DataFrame,
    year_col: str,
    yield_col: str,
    crop_col: Optional[str] = None,
    top_crops_limit: int = 5
) -> Optional[go.Figure]:
    """
    Plots historical crop yield trends over time.
    """
    if year_col not in df.columns or yield_col not in df.columns:
        return None

    valid_df = df.dropna(subset=[year_col, yield_col]).copy()
    valid_df[year_col] = pd.to_numeric(valid_df[year_col], errors="coerce")
    valid_df = valid_df.dropna(subset=[year_col])
    valid_df[year_col] = valid_df[year_col].astype(int)

    if valid_df.empty or valid_df[year_col].nunique() < 2:
        return None

    if crop_col and crop_col in valid_df.columns:
        top_crops = valid_df[crop_col].value_counts().head(top_crops_limit).index.tolist()
        trend_df = (
            valid_df[valid_df[crop_col].isin(top_crops)]
            .groupby([year_col, crop_col])[yield_col]
            .mean()
            .reset_index()
        )
        fig = px.line(
            trend_df,
            x=year_col,
            y=yield_col,
            color=crop_col,
            markers=True,
            title=f"Historical Yield Trends by Crop ({trend_df[year_col].min()} - {trend_df[year_col].max()})",
            template=THEME_TEMPLATE,
        )
    else:
        trend_df = valid_df.groupby(year_col)[yield_col].agg(["mean", "min", "max"]).reset_index()
        fig = px.line(
            trend_df,
            x=year_col,
            y="mean",
            markers=True,
            title=f"Annual Average Crop Yield Trend ({trend_df[year_col].min()} - {trend_df[year_col].max()})",
            template=THEME_TEMPLATE,
            labels={"mean": f"Mean {yield_col}"},
        )

    fig.update_layout(xaxis=dict(tickmode="linear"), height=420, margin=dict(l=40, r=40, t=50, b=60))
    return fig


def plot_region_yield_comparison(
    df: pd.DataFrame,
    region_col: str,
    yield_col: str,
    top_n: int = 15
) -> Optional[go.Figure]:
    """
    Plots mean yield by State, District, or Region.
    """
    if region_col not in df.columns or yield_col not in df.columns:
        return None

    region_stats = (
        df.groupby(region_col)[yield_col]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(top_n)
    )
    region_stats["mean"] = region_stats["mean"].round(2)

    fig = px.bar(
        region_stats,
        x=region_col,
        y="mean",
        text="mean",
        color="mean",
        color_continuous_scale="Viridis",
        title=f"Top {len(region_stats)} Regions by Average Yield",
        template=THEME_TEMPLATE,
        labels={"mean": f"Average {yield_col}"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, height=420, margin=dict(l=40, r=40, t=50, b=90))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> Optional[go.Figure]:
    """
    Calculates and visualizes Pearson correlation matrix for numeric features.
    """
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_cols) < 2:
        return None

    corr_df = df[numeric_cols].corr().round(2)

    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
        template=THEME_TEMPLATE,
    )
    fig.update_layout(height=480, margin=dict(l=60, r=40, t=60, b=60))
    return fig


def plot_weather_analysis(
    df: pd.DataFrame,
    yield_col: str,
    rainfall_col: Optional[str] = None,
    temp_col: Optional[str] = None,
) -> Dict[str, go.Figure]:
    """
    Plots scatter plots with trendlines for rainfall and temperature vs yield.
    """
    figs: Dict[str, go.Figure] = {}

    if rainfall_col and rainfall_col in df.columns and yield_col in df.columns:
        valid_rf = df.dropna(subset=[rainfall_col, yield_col])
        if len(valid_rf) >= 5:
            fig_rf = px.scatter(
                valid_rf,
                x=rainfall_col,
                y=yield_col,
                trendline="ols",
                title=f"Crop Yield vs. Rainfall ({rainfall_col})",
                template=THEME_TEMPLATE,
                opacity=0.65,
                color_discrete_sequence=["#1b9e77"],
            )
            fig_rf.update_layout(height=400, margin=dict(l=40, r=40, t=50, b=60))
            figs["rainfall"] = fig_rf

    if temp_col and temp_col in df.columns and yield_col in df.columns:
        valid_tmp = df.dropna(subset=[temp_col, yield_col])
        if len(valid_tmp) >= 5:
            fig_tmp = px.scatter(
                valid_tmp,
                x=temp_col,
                y=yield_col,
                trendline="ols",
                title=f"Crop Yield vs. Temperature ({temp_col})",
                template=THEME_TEMPLATE,
                opacity=0.65,
                color_discrete_sequence=["#d95f02"],
            )
            fig_tmp.update_layout(height=400, margin=dict(l=40, r=40, t=50, b=60))
            figs["temperature"] = fig_tmp

    return figs


def plot_feature_distribution(df: pd.DataFrame, col_name: str) -> Optional[go.Figure]:
    """
    Plots the distribution (histogram/box) of any numeric or categorical feature.
    """
    if col_name not in df.columns:
        return None

    if pd.api.types.is_numeric_dtype(df[col_name]):
        fig = px.histogram(
            df.dropna(subset=[col_name]),
            x=col_name,
            marginal="box",
            title=f"Distribution of {col_name}",
            template=THEME_TEMPLATE,
            color_discrete_sequence=["#2b83ba"],
        )
    else:
        counts = df[col_name].value_counts().head(20).reset_index()
        counts.columns = [col_name, "Count"]
        fig = px.bar(
            counts,
            x=col_name,
            y="Count",
            title=f"Value Distribution of {col_name}",
            template=THEME_TEMPLATE,
            color="Count",
            color_continuous_scale="Teal",
        )
        fig.update_layout(xaxis_tickangle=-45)

    fig.update_layout(height=380, margin=dict(l=40, r=40, t=50, b=60))
    return fig
