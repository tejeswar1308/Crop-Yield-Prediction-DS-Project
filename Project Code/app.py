"""
Crop Yield Prediction and Recommendation System
Modern Streamlit web application for end-to-end dynamic agricultural data analysis,
automated machine learning, inference, and crop ranking.
"""

from __future__ import annotations
import io
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Import core modules
from src.dataset_loader import (
    load_dataset,
    compute_dataset_stats,
    DatasetLoadError,
    export_dataframe_to_csv,
    export_dataframe_to_excel,
)
from src.schema_detector import (
    detect_schema_mappings,
    can_derive_yield,
    ROLE_DEFINITIONS,
)
from src.data_cleaner import clean_dataset
from src.eda import (
    plot_missing_values_chart,
    plot_crop_distribution,
    plot_crop_yield_comparison,
    plot_historical_yield_trends,
    plot_region_yield_comparison,
    plot_correlation_heatmap,
    plot_weather_analysis,
    plot_feature_distribution,
)
from src.model_trainer import (
    train_and_compare_models,
    save_trained_bundle,
    TrainingError,
)
from src.model_evaluator import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance_chart,
    create_models_comparison_table,
    plot_model_comparison_bar,
    generate_limitations_and_warnings,
)
from src.predictor import predict_single_instance, PredictionValidationError
from src.crop_recommender import (
    can_support_crop_recommendation,
    compare_and_rank_crops,
    plot_crop_ranking_chart,
    CropRecommendationError,
)


# --- PAGE SETUP & CUSTOM CSS ---
st.set_page_config(
    page_title="AgriPredict - Crop Yield AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main container styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a1e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #276749;
        margin-top: 0.2rem;
    }
    .badge-success {
        display: inline-block;
        background: #c6f6d5;
        color: #22543d;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warning {
        display: inline-block;
        background: #feebc8;
        color: #744210;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-info {
        display: inline-block;
        background: #e2e8f0;
        color: #2d3748;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .card-container {
        background-color: #f7fafc;
        border: 1px solid #edf2f7;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- SESSION STATE INITIALIZATION ---
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "dataset_meta" not in st.session_state:
    st.session_state.dataset_meta = None
if "schema_mapping" not in st.session_state:
    st.session_state.schema_mapping = {}
if "cleaning_summary" not in st.session_state:
    st.session_state.cleaning_summary = None
if "trained_bundle" not in st.session_state:
    st.session_state.trained_bundle = None
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []
if "last_recommendation" not in st.session_state:
    st.session_state.last_recommendation = None


# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("## 🌾 AgriPredict AI")
st.sidebar.caption("Dynamic Crop Yield Prediction & Recommendation")

menu_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home Dashboard",
        "📤 Upload Dataset & Schema",
        "🧹 Data Cleaning & Preprocessing",
        "📊 Exploratory Data Analysis (EDA)",
        "🤖 Model Training & Comparison",
        "🌾 Predict Crop Yield",
        "🌟 Recommend Best Crop",
        "📈 Model Evaluation Dashboard",
        "ℹ️ About & Limitations",
    ],
    index=0,
)

# Sidebar System Status Card
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 System Status")

if st.session_state.raw_df is not None:
    n_rows = len(st.session_state.raw_df)
    n_cols = len(st.session_state.raw_df.columns)
    st.sidebar.success(f"Dataset Loaded: {n_rows:,} rows, {n_cols} cols")
    if st.session_state.cleaned_df is not None:
        st.sidebar.caption("🧹 Dataset Cleaned & Ready")
else:
    st.sidebar.info("Dataset: No file loaded")

if st.session_state.trained_bundle is not None:
    b_name = st.session_state.trained_bundle["best_model_name"]
    b_r2 = st.session_state.trained_bundle["best_metrics"]["R2"]
    st.sidebar.success(f"Model: {b_name} (R² = {b_r2:.3f})")
else:
    st.sidebar.caption("Model: Untrained")


# -------------------------------------------------------------
# PAGE 1: 🏠 HOME DASHBOARD
# -------------------------------------------------------------
if menu_choice == "🏠 Home Dashboard":
    st.markdown('<div class="main-title">Crop Yield Prediction & Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A dynamic, machine-learning-powered agronomic decision support platform. '
        'Upload your agricultural data, automatically train predictive models, explore historical patterns, '
        'and discover the highest-yielding crops for your region.</div>',
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Dataset Records</div>'
            f'<div class="metric-value">{len(st.session_state.raw_df) if st.session_state.raw_df is not None else 0:,}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Active Features</div>'
            f'<div class="metric-value">{len(st.session_state.trained_bundle["feature_cols"]) if st.session_state.trained_bundle is not None else 0}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        best_r2_val = (
            f"{st.session_state.trained_bundle['best_metrics']['R2']:.3f}"
            if st.session_state.trained_bundle is not None
            else "N/A"
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Model R² Score</div>'
            f'<div class="metric-value">{best_r2_val}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Predictions Logged</div>'
            f'<div class="metric-value">{len(st.session_state.predictions_history)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 🚀 Guided Workflow")
    st.markdown("""
    1. **Upload Dataset (`📤`)**: Upload your agricultural CSV or Excel file. The system dynamically maps column roles (Crop, Yield, Rainfall, Soil, etc.).
    2. **Data Cleaning (`🧹`)**: Remove duplicates, handle missing values, cap outliers, and optionally derive Yield = Production / Area.
    3. **Exploratory Data Analysis (`📊`)**: Interactive Plotly visualizations for yield distributions, temporal trends, and correlation heatmaps.
    4. **Automated ML Training (`🤖`)**: Automatically trains and benchmarks Ridge Regression, Decision Trees, Random Forests, and Gradient Boosting on held-out test data.
    5. **Predict Yield (`🌾`)**: Enter local field and weather conditions to forecast expected crop yield with extrapolation warnings.
    6. **Recommend Best Crop (`🌟`)**: Compare all supported candidate crops under identical weather and soil conditions to find the optimal yield.
    7. **Evaluation Dashboard (`📈`)**: Diagnostic plots, actual vs. predicted comparisons, residual analysis, and agronomic limitations.
    """)

    st.markdown("---")
    st.info("💡 **Quick Start**: If you don't have an agricultural dataset ready, navigate to the **'Upload Dataset & Schema'** page and click **'Load Sample Dataset'** for an instant demo!")


# -------------------------------------------------------------
# PAGE 2: 📤 UPLOAD DATASET & SCHEMA MAPPING
# -------------------------------------------------------------
elif menu_choice == "📤 Upload Dataset & Schema":
    st.markdown('<div class="main-title">Dataset Upload & Schema Mapping</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload your agricultural dataset in CSV or Excel format. '
        'The application dynamically inspects the file, detects key agricultural features, and lets you verify column roles.</div>',
        unsafe_allow_html=True,
    )

    # Demo Dataset Loader Button
    col_demo1, col_demo2 = st.columns([1, 3])
    with col_demo1:
        if st.button("📁 Load Built-in Sample CSV", use_container_width=True):
            sample_path = "data/agricultural_yield_dataset.csv"
            if os.path.exists(sample_path):
                df_loaded, meta = load_dataset(sample_path)
                st.session_state.raw_df = df_loaded
                st.session_state.dataset_meta = meta
                st.session_state.cleaned_df = None
                st.session_state.schema_mapping = detect_schema_mappings(df_loaded)
                st.success("Loaded built-in agricultural sample dataset!")
                st.rerun()
            else:
                st.error("Sample dataset file not found on disk.")

    with col_demo2:
        if st.button("📊 Load Sample Excel (Derive Yield Demo)", use_container_width=False):
            excel_path = "data/crop_production_derive_yield.xlsx"
            if os.path.exists(excel_path):
                df_loaded, meta = load_dataset(excel_path)
                st.session_state.raw_df = df_loaded
                st.session_state.dataset_meta = meta
                st.session_state.cleaned_df = None
                st.session_state.schema_mapping = detect_schema_mappings(df_loaded)
                st.success("Loaded Excel dataset without explicit Yield (ready for Yield derivation)!")
                st.rerun()

    uploaded_file = st.file_uploader(
        "Upload agricultural dataset (CSV, XLSX, XLS)",
        type=["csv", "xlsx", "xls"],
        help="Upload any agricultural dataset. No hardcoded schemas required.",
    )

    if uploaded_file is not None:
        try:
            df_loaded, meta = load_dataset(uploaded_file, uploaded_file.name)
            # Only update if new file loaded
            if (
                st.session_state.dataset_meta is None
                or st.session_state.dataset_meta.get("file_name") != meta["file_name"]
            ):
                st.session_state.raw_df = df_loaded
                st.session_state.dataset_meta = meta
                st.session_state.cleaned_df = None
                st.session_state.schema_mapping = detect_schema_mappings(df_loaded)
                st.success(f"Successfully loaded '{meta['file_name']}'!")
                st.rerun()
        except DatasetLoadError as dle:
            st.error(f"Dataset Loading Error: {dle}")
            st.info("Ensure your file has valid column headers and delimiter-separated rows.")
        except Exception as ex:
            st.error(f"Unexpected error while reading file: {ex}")

    # Display dataset details if loaded
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        stats = compute_dataset_stats(df)

        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")

        # Metric badges
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Rows", f"{stats['total_rows']:,}")
        m2.metric("Total Columns", stats['total_cols'])
        m3.metric("Missing Cells", f"{stats['total_missing_cells']:,} ({stats['overall_missing_pct']}%)")
        m4.metric("Duplicate Rows", f"{stats['duplicate_rows']} ({stats['duplicate_pct']}%)")
        m5.metric("Numeric Features", len(stats['numeric_cols']))

        # Data preview
        with st.expander("👀 Dataset Preview (First 15 Rows)", expanded=True):
            st.dataframe(df.head(15), use_container_width=True)

        # Detailed column stats table
        with st.expander("📋 Column Details & Missing Values Breakdown", expanded=False):
            col_details_df = pd.DataFrame(stats["column_details"])
            st.dataframe(col_details_df, use_container_width=True)

        # Dynamic Schema Role Mapping Card
        st.markdown("---")
        st.markdown("### 🔍 Semantic Agricultural Role Detection")
        st.markdown(
            "The system automatically analyzed your dataset columns to suggest agricultural roles. "
            "Please confirm or adjust the mappings below if needed."
        )

        detected = st.session_state.schema_mapping
        all_cols_options = ["(None / Unmapped)"] + list(df.columns)

        col_left, col_right = st.columns(2)
        confirmed_mapping: Dict[str, Optional[str]] = {}

        role_keys = list(ROLE_DEFINITIONS.keys())
        half = len(role_keys) // 2

        with col_left:
            for rk in role_keys[:half]:
                meta = detected.get(rk, {})
                auto_col = meta.get("column")
                idx = all_cols_options.index(auto_col) if auto_col in all_cols_options else 0
                chosen = st.selectbox(
                    f"{ROLE_DEFINITIONS[rk]['label']}",
                    options=all_cols_options,
                    index=idx,
                    key=f"schema_sel_{rk}",
                    help=meta.get("reason", ""),
                )
                confirmed_mapping[rk] = chosen if chosen != "(None / Unmapped)" else None

        with col_right:
            for rk in role_keys[half:]:
                meta = detected.get(rk, {})
                auto_col = meta.get("column")
                idx = all_cols_options.index(auto_col) if auto_col in all_cols_options else 0
                chosen = st.selectbox(
                    f"{ROLE_DEFINITIONS[rk]['label']}",
                    options=all_cols_options,
                    index=idx,
                    key=f"schema_sel_{rk}",
                    help=meta.get("reason", ""),
                )
                confirmed_mapping[rk] = chosen if chosen != "(None / Unmapped)" else None

        # Check Yield availability
        yield_col = confirmed_mapping.get("yield")
        if not yield_col:
            st.warning("⚠️ No explicit 'Yield' column mapped.")
            can_derive, derive_msg = can_derive_yield(confirmed_mapping, df)
            if can_derive:
                st.success(f"💡 {derive_msg}. You can compute this in the **Data Cleaning** page!")
            else:
                st.info(f"ℹ️ {derive_msg}")

        if st.button("✅ Confirm Schema Mappings", use_container_width=True):
            # Update detected mapping with confirmed selections
            for rk, col_val in confirmed_mapping.items():
                if rk in st.session_state.schema_mapping:
                    st.session_state.schema_mapping[rk]["column"] = col_val
                else:
                    st.session_state.schema_mapping[rk] = {"column": col_val, "confidence": 1.0}
            st.success("Column schema mappings confirmed! Proceed to Data Cleaning.")


# -------------------------------------------------------------
# PAGE 3: 🧹 DATA CLEANING & PREPROCESSING
# -------------------------------------------------------------
elif menu_choice == "🧹 Data Cleaning & Preprocessing":
    st.markdown('<div class="main-title">Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Safely prepare your agricultural data for exploratory analysis and model training. '
        'The original uploaded dataset is preserved untouched.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first in the **Upload Dataset & Schema** page.")
    else:
        df_raw = st.session_state.raw_df
        mapping = {k: v.get("column") for k, v in st.session_state.schema_mapping.items()}

        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            st.markdown("#### ⚙️ Cleaning Configurations")
            remove_dups = st.checkbox("Remove identical duplicate rows", value=True)
            drop_na_target = st.checkbox(
                "Filter out rows missing the target variable",
                value=True,
                help="Supervised ML cannot train on missing labels. Silently fabricating real-world yield creates label leakage.",
            )

            # Target selection
            all_cols = list(df_raw.columns)
            default_target = mapping.get("yield")
            target_idx = all_cols.index(default_target) if default_target in all_cols else 0
            selected_target = st.selectbox(
                "Target Variable for Model",
                options=all_cols,
                index=target_idx,
            )

        with col_opt2:
            st.markdown("#### 🌾 Dynamic Yield Derivation")
            # Check if user wants to derive yield
            can_derive, derive_msg = can_derive_yield(mapping, df_raw)
            derive_yield_checked = st.checkbox(
                "Calculate Yield = Production / Cultivated Area",
                value=(mapping.get("yield") is None and can_derive),
                disabled=not can_derive,
                help=derive_msg,
            )
            area_col = mapping.get("area")
            prod_col = mapping.get("production")

            if derive_yield_checked:
                st.caption(f"Will compute: `{prod_col}` / `{area_col}` (excluding non-positive area).")

            st.markdown("#### 🔍 Outlier Handling Strategy")
            outlier_strategy = st.selectbox(
                "Numeric Outlier Action",
                options=["inspect", "cap_iqr", "drop_extreme"],
                format_func=lambda x: {
                    "inspect": "Inspect only (Do not alter values)",
                    "cap_iqr": "Cap at IQR bounds (1.5x IQR)",
                    "drop_extreme": "Remove extreme outliers (3.0x IQR)",
                }[x],
            )

        # Trigger Cleaning
        if st.button("🧹 Execute Data Cleaning Pipeline", use_container_width=True):
            with st.spinner("Processing data cleaning operations..."):
                cleaned_df, summary = clean_dataset(
                    df_raw=df_raw,
                    target_col=selected_target,
                    remove_duplicates=remove_dups,
                    drop_missing_target=drop_na_target,
                    derive_yield=derive_yield_checked,
                    area_col=area_col,
                    prod_col=prod_col,
                    yield_col_name="Yield",
                    outlier_action=outlier_strategy,
                )
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_summary = summary
                if derive_yield_checked and "Yield" in cleaned_df.columns:
                    st.session_state.schema_mapping["yield"] = {"column": "Yield", "confidence": 1.0}
                st.success("Cleaning completed successfully!")

        # Display results if cleaned
        active_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df_raw
        summary = st.session_state.cleaning_summary

        if summary is not None:
            st.markdown("---")
            st.markdown("### 📋 Cleaning Audit Log")

            log_records = summary["audit_log"]
            if log_records:
                st.dataframe(pd.DataFrame(log_records), use_container_width=True)

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Initial Rows", f"{summary['initial_rows']:,}")
            kpi2.metric("Cleaned Rows", f"{summary['final_rows']:,}")
            kpi3.metric("Rows Removed", f"{summary['rows_removed']:,}")
            kpi4.metric("Active Columns", summary['final_cols'])

            # Outlier diagnostics
            outlier_data = summary.get("outlier_reports", {})
            if outlier_data:
                with st.expander("📊 Outlier Detection Report (IQR Method)", expanded=False):
                    out_rows = []
                    for col_n, vals in outlier_data.items():
                        out_rows.append({
                            "Feature": col_n,
                            "Outliers Detected": vals["outlier_count"],
                            "Lower Bound (Q1 - 1.5 IQR)": vals["lower_bound"],
                            "Upper Bound (Q3 + 1.5 IQR)": vals["upper_bound"],
                        })
                    st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📥 Export Cleaned Dataset")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            csv_bytes = export_dataframe_to_csv(active_df)
            st.download_button(
                "⬇️ Download Cleaned Data as CSV",
                data=csv_bytes,
                file_name="cleaned_agricultural_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col2:
            excel_bytes = export_dataframe_to_excel(active_df)
            st.download_button(
                "⬇️ Download Cleaned Data as Excel",
                data=excel_bytes,
                file_name="cleaned_agricultural_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )


# -------------------------------------------------------------
# PAGE 4: 📊 EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
elif menu_choice == "📊 Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Visualize agricultural patterns, crop distributions, weather correlations, '
        'and temporal yield trends. All visualizations dynamically reflect your uploaded dataset.</div>',
        unsafe_allow_html=True,
    )

    df_to_use = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df

    if df_to_use is None:
        st.warning("Please upload a dataset first in the **Upload Dataset & Schema** page.")
    else:
        mapping = {k: v.get("column") for k, v in st.session_state.schema_mapping.items()}
        crop_col = mapping.get("crop")
        yield_col = mapping.get("yield")
        year_col = mapping.get("year")
        region_col = mapping.get("region")
        rainfall_col = mapping.get("rainfall")
        temp_col = mapping.get("temperature")

        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
            "🌾 Crop & Yield Distributions",
            "📈 Historical Trends & Regional Analysis",
            "🌦️ Weather & Soil Correlations",
            "🔍 Feature Distribution Explorer",
        ])

        with eda_tab1:
            st.markdown("#### Crop Distribution & Yield Variances")
            c1, c2 = st.columns(2)

            with c1:
                if crop_col and crop_col in df_to_use.columns:
                    fig_crop = plot_crop_distribution(df_to_use, crop_col)
                    if fig_crop:
                        st.plotly_chart(fig_crop, use_container_width=True)
                else:
                    st.info("Crop column not mapped. Map 'Crop' in the Schema page to see crop distribution.")

            with c2:
                if crop_col and yield_col and crop_col in df_to_use.columns and yield_col in df_to_use.columns:
                    fig_box = plot_crop_yield_comparison(df_to_use, crop_col, yield_col)
                    if fig_box:
                        st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Map both 'Crop' and 'Yield' columns to view yield comparisons across crops.")

        with eda_tab2:
            st.markdown("#### Historical Trends & Geographic Performance")
            h1, h2 = st.columns(2)

            with h1:
                if year_col and yield_col and year_col in df_to_use.columns and yield_col in df_to_use.columns:
                    fig_trend = plot_historical_yield_trends(df_to_use, year_col, yield_col, crop_col)
                    if fig_trend:
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("Insufficient distinct historical years to construct a trendline.")
                else:
                    st.info("Historical trends require both 'Year' and 'Yield' columns to be mapped.")

            with h2:
                if region_col and yield_col and region_col in df_to_use.columns and yield_col in df_to_use.columns:
                    fig_reg = plot_region_yield_comparison(df_to_use, region_col, yield_col)
                    if fig_reg:
                        st.plotly_chart(fig_reg, use_container_width=True)
                else:
                    st.info("Regional comparison requires both 'Region/State' and 'Yield' columns.")

        with eda_tab3:
            st.markdown("#### Environmental Features & Correlation Heatmap")
            w1, w2 = st.columns(2)

            with w1:
                fig_corr = plot_correlation_heatmap(df_to_use)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("At least 2 numeric features required for correlation matrix.")

            with w2:
                if yield_col and yield_col in df_to_use.columns:
                    weather_figs = plot_weather_analysis(df_to_use, yield_col, rainfall_col, temp_col)
                    if "rainfall" in weather_figs:
                        st.plotly_chart(weather_figs["rainfall"], use_container_width=True)
                    if "temperature" in weather_figs:
                        st.plotly_chart(weather_figs["temperature"], use_container_width=True)
                    if not weather_figs:
                        st.info("Rainfall or Temperature features not mapped or have insufficient data for trendlines.")
                else:
                    st.info("Yield column required for weather relationship plots.")

        with eda_tab4:
            st.markdown("#### Custom Feature Distribution Viewer")
            selected_f = st.selectbox("Select any column to inspect its distribution:", options=df_to_use.columns)
            if selected_f:
                fig_f = plot_feature_distribution(df_to_use, selected_f)
                if fig_f:
                    st.plotly_chart(fig_f, use_container_width=True)

            # Missing value chart
            fig_miss = plot_missing_values_chart(df_to_use)
            if fig_miss:
                st.plotly_chart(fig_miss, use_container_width=True)


# -------------------------------------------------------------
# PAGE 5: 🤖 MODEL TRAINING & COMPARISON
# -------------------------------------------------------------
elif menu_choice == "🤖 Model Training & Comparison":
    st.markdown('<div class="main-title">Automated Machine Learning Model Training</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Train and benchmark multiple machine learning algorithms. '
        'Evaluates on held-out test data, prevents data leakage via isolated ColumnTransformers, '
        'and selects the best suited model.</div>',
        unsafe_allow_html=True,
    )

    df_for_ml = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df

    if df_for_ml is None:
        st.warning("Please upload a dataset first in the **Upload Dataset & Schema** page.")
    else:
        mapping = {k: v.get("column") for k, v in st.session_state.schema_mapping.items()}
        all_cols = list(df_for_ml.columns)

        col_cfg1, col_cfg2 = st.columns(2)

        with col_cfg1:
            st.markdown("#### 🎯 Target Variable Selection")
            default_target = mapping.get("yield")
            target_idx = all_cols.index(default_target) if default_target in all_cols else 0
            target_column = st.selectbox("Select Target Variable (Continuous Crop Yield)", all_cols, index=target_idx)

            yield_unit = st.text_input("Yield Measurement Unit", value="tonnes/hectare")

            # Features multi-select
            available_features = [c for c in all_cols if c != target_column]
            # Default to mapped features if present
            default_feats = [
                col for rk, col in mapping.items()
                if col and col in available_features and rk not in ["yield", "production"]
            ]
            if not default_feats:
                default_feats = available_features[:min(6, len(available_features))]

            selected_features = st.multiselect(
                "Select Feature Columns for Training",
                options=available_features,
                default=default_feats,
                help="Choose agricultural variables (weather, crop type, location, soil properties). Avoid total production as an input if predicting yield to avoid leakage.",
            )

        with col_cfg2:
            st.markdown("#### ⚙️ Validation & Split Strategy")
            year_col = mapping.get("year")
            can_chrono = year_col is not None and year_col in selected_features

            split_method = st.radio(
                "Train / Test Split Method",
                options=["chronological", "random"] if can_chrono else ["random"],
                format_func=lambda x: "Chronological Split (Train on earlier years, test on future years)" if x == "chronological" else "Random Split (Stratified/Shuffled)",
                help="Chronological splitting is best practice when predicting future harvest seasons.",
            )

            test_ratio = st.slider("Held-out Test Set Size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
            random_seed = st.number_input("Reproducible Random Seed", value=42, min_value=1, max_value=99999)

        st.markdown("---")
        if st.button("🚀 Train & Compare Candidate Models", use_container_width=True):
            if not selected_features:
                st.error("Please select at least one feature column for model training.")
            else:
                with st.spinner("Training Random Forest, Gradient Boosting, Decision Tree, and Ridge Regressors..."):
                    try:
                        bundle = train_and_compare_models(
                            df=df_for_ml,
                            target_col=target_column,
                            feature_cols=selected_features,
                            year_col=year_col if split_method == "chronological" else None,
                            split_method=split_method,
                            test_size=test_ratio,
                            random_state=int(random_seed),
                            dataset_name=st.session_state.dataset_meta.get("file_name", "dataset.csv") if st.session_state.dataset_meta else "dataset.csv",
                            yield_unit=yield_unit,
                        )

                        # Save bundle in session and disk
                        st.session_state.trained_bundle = bundle
                        save_trained_bundle(bundle, "models/best_crop_model.joblib")
                        st.success(f"Training Complete! Best Performing Model: **{bundle['best_model_name']}**")
                    except TrainingError as te:
                        st.error(f"Model Training Error: {te}")
                    except Exception as ex:
                        st.error(f"Unexpected error during training: {ex}")

        # Display results if model trained
        if st.session_state.trained_bundle is not None:
            bundle = st.session_state.trained_bundle
            best_name = bundle["best_model_name"]
            metrics = bundle["best_metrics"]
            comp_table = create_models_comparison_table(bundle["models_comparison"])

            st.markdown("---")
            st.markdown("### 🏆 Model Comparison on Held-out Test Set")

            # Metrics cards
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            kpi_c1.metric("Selected Best Model", best_name)
            kpi_c2.metric("Test R² Score", f"{metrics['R2']:.3f}")
            kpi_c3.metric("Test MAE", f"{metrics['MAE']:.3f} {yield_unit}")
            kpi_c4.metric("Test RMSE", f"{metrics['RMSE']:.3f} {yield_unit}")

            # Plain language explanation
            st.info(f"**Performance Summary**: {bundle['explanation']}")

            # Comparison Table & Bar Chart
            st.dataframe(comp_table, use_container_width=True)

            fig_comp = plot_model_comparison_bar(comp_table)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Feature Importance
            if bundle["best_importances"]:
                st.markdown("### 🌟 Feature Importance Ranking")
                fig_imp = plot_feature_importance_chart(bundle["best_importances"])
                if fig_imp:
                    st.plotly_chart(fig_imp, use_container_width=True)

            # Export Options
            st.markdown("---")
            st.markdown("### 📥 Model Downloads & Reports")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                comp_csv = comp_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Model Comparison Table (CSV)",
                    data=comp_csv,
                    file_name="model_comparison_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_d2:
                if os.path.exists("models/best_crop_model.joblib"):
                    with open("models/best_crop_model.joblib", "rb") as f:
                        bundle_bytes = f.read()
                    st.download_button(
                        "⬇️ Download Trained Model Pipeline (.joblib)",
                        data=bundle_bytes,
                        file_name="best_crop_pipeline.joblib",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )


# -------------------------------------------------------------
# PAGE 6: 🌾 PREDICT CROP YIELD
# -------------------------------------------------------------
elif menu_choice == "🌾 Predict Crop Yield":
    st.markdown('<div class="main-title">Crop Yield Prediction Form</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enter agronomic and environmental parameters to forecast crop yield. '
        'All input fields are dynamically generated from the features used by your trained model.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.trained_bundle is None:
        st.warning("No model has been trained yet. Please train a model in the **Model Training & Comparison** page.")
    else:
        bundle = st.session_state.trained_bundle
        feature_cols = bundle["feature_cols"]
        num_features = bundle["numeric_features"]
        cat_features = bundle["categorical_features"]
        feature_stats = bundle["feature_stats"]
        yield_unit = bundle["metadata"].get("yield_unit", "tonnes/hectare")

        st.markdown(f"**Fitted Model**: `{bundle['best_model_name']}` | **Features**: {len(feature_cols)}")

        # Build dynamic form
        inputs: Dict[str, Any] = {}

        with st.form("yield_prediction_form"):
            st.markdown("#### Input Field Conditions")
            col_f1, col_f2 = st.columns(2)

            for idx, feat in enumerate(feature_cols):
                target_col = col_f1 if idx % 2 == 0 else col_f2

                with target_col:
                    if feat in num_features:
                        stats = feature_stats.get(feat, {})
                        f_min = float(stats.get("min", 0.0))
                        f_max = float(stats.get("max", 100.0))
                        f_med = float(stats.get("median", 0.0))
                        step_val = max(0.1, round((f_max - f_min) / 100.0, 2)) if f_max > f_min else 1.0

                        inputs[feat] = st.number_input(
                            f"{feat} (Historical Range: {f_min:.1f} – {f_max:.1f})",
                            value=f_med,
                            step=step_val,
                            key=f"pred_input_{feat}",
                        )
                    else:
                        stats = feature_stats.get(feat, {})
                        categories = stats.get("categories", ["Unknown"])
                        inputs[feat] = st.selectbox(
                            f"{feat}",
                            options=categories,
                            key=f"pred_input_{feat}",
                        )

            submitted = st.form_submit_button("🔮 Predict Crop Yield", use_container_width=True)

        if submitted:
            try:
                res = predict_single_instance(
                    pipeline=bundle["pipeline"],
                    inputs=inputs,
                    feature_stats=feature_stats,
                    numeric_features=num_features,
                    categorical_features=cat_features,
                    yield_unit=yield_unit,
                )

                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")

                pred_c1, pred_c2 = st.columns([1, 2])
                with pred_c1:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">Estimated Yield</div>'
                        f'<div class="metric-value">{res["predicted_yield"]:.2f} <span style="font-size:1.1rem;color:#718096;">{yield_unit}</span></div></div>',
                        unsafe_allow_html=True,
                    )
                with pred_c2:
                    st.markdown("#### 📋 Parameters Provided")
                    inputs_df = pd.DataFrame([res["input_features_used"]])
                    st.dataframe(inputs_df, use_container_width=True)

                # Show extrapolation warnings if any
                if res["warnings"]:
                    st.markdown("#### ⚠️ Out-of-Distribution Warnings")
                    for w in res["warnings"]:
                        st.warning(w)

                # Append to prediction history
                hist_record = dict(inputs)
                hist_record["Predicted_Yield"] = res["predicted_yield"]
                hist_record["Yield_Unit"] = yield_unit
                st.session_state.predictions_history.append(hist_record)

            except PredictionValidationError as pve:
                st.error(f"Validation Error: {pve}")
            except Exception as ex:
                st.error(f"Prediction Error: {ex}")

        # Prediction History
        if st.session_state.predictions_history:
            st.markdown("---")
            st.markdown("### 🕒 Prediction Session History")
            hist_df = pd.DataFrame(st.session_state.predictions_history)
            st.dataframe(hist_df, use_container_width=True)

            csv_hist = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Prediction History as CSV",
                data=csv_hist,
                file_name="crop_yield_predictions.csv",
                mime="text/csv",
            )


# -------------------------------------------------------------
# PAGE 7: 🌟 RECOMMEND BEST CROP
# -------------------------------------------------------------
elif menu_choice == "🌟 Recommend Best Crop":
    st.markdown('<div class="main-title">Best Crop Recommendation & Ranking</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Compare all supported candidate crops under uniform regional and environmental conditions. '
        'Discover which crop produces the highest expected yield.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.trained_bundle is None:
        st.warning("Please train a model first in the **Model Training & Comparison** page.")
    else:
        bundle = st.session_state.trained_bundle
        can_rec, rec_msg, crop_col = can_support_crop_recommendation(bundle)

        if not can_rec:
            st.error("🚫 Crop Recommendation Unavailable for Current Model")
            st.info(rec_msg)
        else:
            feature_cols = bundle["feature_cols"]
            num_features = bundle["numeric_features"]
            cat_features = bundle["categorical_features"]
            feature_stats = bundle["feature_stats"]
            yield_unit = bundle["metadata"].get("yield_unit", "tonnes/hectare")

            crop_stats = feature_stats.get(crop_col, {})
            all_crops = crop_stats.get("categories", [])

            st.markdown(f"**Identified Crop Feature**: `{crop_col}` ({len(all_crops)} candidate crops supported)")

            # Base condition inputs (everything except the crop feature)
            base_inputs: Dict[str, Any] = {}

            with st.form("crop_recommendation_form"):
                st.markdown("#### Set Uniform Environmental & Regional Conditions")
                col_r1, col_r2 = st.columns(2)

                other_features = [f for f in feature_cols if f != crop_col]

                for idx, feat in enumerate(other_features):
                    target_col = col_r1 if idx % 2 == 0 else col_r2
                    with target_col:
                        if feat in num_features:
                            stats = feature_stats.get(feat, {})
                            f_min = float(stats.get("min", 0.0))
                            f_max = float(stats.get("max", 100.0))
                            f_med = float(stats.get("median", 0.0))
                            step_val = max(0.1, round((f_max - f_min) / 100.0, 2)) if f_max > f_min else 1.0

                            base_inputs[feat] = st.number_input(
                                f"{feat} (Historical Range: {f_min:.1f} – {f_max:.1f})",
                                value=f_med,
                                step=step_val,
                                key=f"rec_base_{feat}",
                            )
                        else:
                            stats = feature_stats.get(feat, {})
                            categories = stats.get("categories", ["Unknown"])
                            base_inputs[feat] = st.selectbox(
                                f"{feat}",
                                options=categories,
                                key=f"rec_base_{feat}",
                            )

                candidate_selection = st.multiselect(
                    "Select candidate crops to compare (Leave empty to compare ALL supported crops)",
                    options=all_crops,
                    default=[],
                )

                rec_submitted = st.form_submit_button("🏆 Compare & Rank Candidate Crops", use_container_width=True)

            if rec_submitted:
                with st.spinner("Simulating yields across candidate crops..."):
                    try:
                        crops_to_run = candidate_selection if candidate_selection else all_crops
                        rec_res = compare_and_rank_crops(
                            pipeline=bundle["pipeline"],
                            base_conditions=base_inputs,
                            crop_col=crop_col,
                            feature_stats=feature_stats,
                            numeric_features=num_features,
                            categorical_features=cat_features,
                            candidate_crops=crops_to_run,
                            yield_unit=yield_unit,
                        )
                        st.session_state.last_recommendation = rec_res
                        st.success(f"Recommendation Generated! Top Recommended Crop: **{rec_res['top_crop']}**")
                    except Exception as ex:
                        st.error(f"Failed to generate recommendations: {ex}")

            # Display recommendation results
            if st.session_state.last_recommendation is not None:
                rec_res = st.session_state.last_recommendation
                ranking_df = rec_res["ranking_df"]

                st.markdown("---")
                st.markdown("### 🥇 Recommendation Summary")

                r1, r2, r3 = st.columns(3)
                r1.metric("Highest Yielding Crop", rec_res["top_crop"])
                r2.metric("Predicted Yield", f"{rec_res['top_yield']:.2f} {yield_unit}")
                r3.metric("Candidate Crops Evaluated", rec_res["total_crops_compared"])

                # Bar chart
                fig_rank = plot_crop_ranking_chart(ranking_df, yield_unit)
                st.plotly_chart(fig_rank, use_container_width=True)

                # Ranking Table
                st.dataframe(ranking_df, use_container_width=True)

                # Economic & Agronomic Disclaimer
                st.warning(
                    "⚠️ **Agronomic & Economic Disclaimer**: "
                    "The top-ranked crop is the one with the highest estimated physical yield volume. "
                    "In real-world farming, **highest yield does not necessarily mean highest economic profit**. "
                    "High-volume crops (e.g. sugarcane or tubers) may carry lower market prices per kilogram or "
                    "higher irrigation and labor costs than high-value oilseeds, pulses, or cash crops."
                )

                csv_rank = ranking_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Crop Ranking Results as CSV",
                    data=csv_rank,
                    file_name="crop_recommendation_ranking.csv",
                    mime="text/csv",
                )


# -------------------------------------------------------------
# PAGE 8: 📈 MODEL EVALUATION DASHBOARD
# -------------------------------------------------------------
elif menu_choice == "📈 Model Evaluation Dashboard":
    st.markdown('<div class="main-title">Model Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">In-depth statistical validation, residual analysis, goodness-of-fit charts, '
        'and scientifically grounded domain limitations.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.trained_bundle is None:
        st.warning("Please train a model first in the **Model Training & Comparison** page.")
    else:
        bundle = st.session_state.trained_bundle
        best_name = bundle["best_model_name"]
        metrics = bundle["best_metrics"]
        metadata = bundle["metadata"]
        y_test = bundle["y_test"]
        y_test_pred = bundle["y_test_pred"]

        # Metadata cards
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Model Architecture", best_name.split()[0])
        k2.metric("Train Samples", f"{metadata['train_rows']:,}")
        k3.metric("Test Samples", f"{metadata['test_rows']:,}")
        k4.metric("Test R²", f"{metrics['R2']:.3f}")
        k5.metric("Test RMSE", f"{metrics['RMSE']:.3f}")

        st.markdown("---")
        # Visual Diagnostic Tabs
        v_tab1, v_tab2, v_tab3 = st.tabs([
            "🎯 Actual vs. Predicted & Residuals",
            "🌟 Feature Importance & Coefficients",
            "⚠️ Domain Caveats & Warnings",
        ])

        with v_tab1:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                fig_avp = plot_actual_vs_predicted(
                    y_test.values,
                    y_test_pred,
                    target_name=bundle["target_col"],
                    model_name=best_name,
                )
                st.plotly_chart(fig_avp, use_container_width=True)

            with col_v2:
                fig_res = plot_residuals(
                    y_test.values,
                    y_test_pred,
                    target_name=bundle["target_col"],
                )
                st.plotly_chart(fig_res, use_container_width=True)

        with v_tab2:
            if bundle["best_importances"]:
                fig_imp = plot_feature_importance_chart(bundle["best_importances"], top_n=20)
                if fig_imp:
                    st.plotly_chart(fig_imp, use_container_width=True)
                imp_df = pd.DataFrame(bundle["best_importances"])
                st.dataframe(imp_df, use_container_width=True)
            else:
                st.info("Feature importance extraction is not supported for this estimator.")

        with v_tab3:
            st.markdown("#### Scientific Rigor & Agricultural Limitations")
            warnings = generate_limitations_and_warnings(bundle)
            for w in warnings:
                st.markdown(f"- ⚠️ {w}")


# -------------------------------------------------------------
# PAGE 9: ℹ️ ABOUT & LIMITATIONS
# -------------------------------------------------------------
elif menu_choice == "ℹ️ About & Limitations":
    st.markdown('<div class="main-title">About & System Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Architectural design, algorithmic methodology, and agricultural domain guidelines.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    ### 🔬 Scientific Methodology
    1. **Data Leakage Prevention**:
       - Transformers (standardizers, median imputers, one-hot encoders) are wrapped inside a Scikit-learn `Pipeline` and `ColumnTransformer`.
       - All parameters are fit strictly on training splits. Test data is never observed during preprocessing.
    2. **Dynamic Schema Adaptation**:
       - The system detects column semantics via regular expressions, keyword analysis, and data value heuristics without hardcoding any crop names or fixed schemas.
    3. **Agricultural Physics & Heuristics**:
       - Physical yield boundaries: Yield cannot be negative.
       - Dynamic calculation: If Yield is absent, it is calculated as `Production / Cultivated Area`, excluding non-positive land area to prevent division by zero.
       - Range Bounding: The inference engine verifies whether prediction inputs extrapolate beyond observed historical training distributions.

    ### 🚜 Practical Considerations in Agriculture
    - **Liebig's Law of the Minimum**: Crop yield is governed not by the total resources available, but by the scarcest nutrient or factor (water, nitrogen, temperature).
    - **Microclimates & Extreme Weather**: Regional aggregate weather statistics may mask localized microclimates, sudden hailstorms, pest infestations, or unseasonal frost.
    - **Profitability vs. Yield**: Yield is purely a physical biomass measure. Farm profitability depends on market prices, transportation costs, and cultivation investments.
    """)

    st.markdown("---")
    st.caption("Developed with Python, Streamlit, Pandas, Scikit-learn, Plotly, Joblib, and Openpyxl.")
