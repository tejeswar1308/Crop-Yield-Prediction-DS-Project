"""
End-to-End Integration Tests
Validates the entire workflow from file upload to data cleaning,
model training, prediction, and crop recommendation using both CSV and Excel sample files.
"""

import os
import pandas as pd
import pytest

from src.dataset_loader import load_dataset, compute_dataset_stats
from src.schema_detector import detect_schema_mappings, can_derive_yield
from src.data_cleaner import clean_dataset
from src.model_trainer import train_and_compare_models
from src.predictor import predict_single_instance
from src.crop_recommender import compare_and_rank_crops, can_support_crop_recommendation


def test_end_to_end_csv_workflow():
    csv_path = "data/agricultural_yield_dataset.csv"
    assert os.path.exists(csv_path)

    # 1. Load Dataset
    df_raw, meta = load_dataset(csv_path)
    assert len(df_raw) > 100
    assert meta["extension"] == ".csv"

    # 2. Stats & Schema
    stats = compute_dataset_stats(df_raw)
    assert stats["total_rows"] == len(df_raw)

    mapping = detect_schema_mappings(df_raw)
    assert mapping["crop"]["column"] == "Crop"
    assert mapping["yield"]["column"] == "Yield_Ton_Ha"
    assert mapping["year"]["column"] == "Crop_Year"

    # 3. Clean Dataset
    df_cleaned, summary = clean_dataset(
        df_raw=df_raw,
        target_col=mapping["yield"]["column"],
        remove_duplicates=True,
        drop_missing_target=True,
        outlier_action="inspect",
    )
    assert len(df_cleaned) > 0
    assert summary["initial_rows"] >= summary["final_rows"]

    # 4. Train Models
    features = ["Crop", "State_Name", "Season", "Annual_Rainfall_mm", "Avg_Temperature_C", "Soil_Type", "Fertilizer_Usage_kg"]
    bundle = train_and_compare_models(
        df=df_cleaned,
        target_col=mapping["yield"]["column"],
        feature_cols=features,
        year_col="Crop_Year",
        split_method="random",
        test_size=0.2,
        random_state=42,
    )
    assert bundle["best_model_name"] in bundle["models_comparison"]
    assert bundle["best_metrics"]["R2"] > -1.0

    # 5. Predict Single Instance
    test_inputs = {
        "Crop": "Wheat",
        "State_Name": "Punjab",
        "Season": "Rabi",
        "Annual_Rainfall_mm": 800.0,
        "Avg_Temperature_C": 24.0,
        "Soil_Type": "Alluvial",
        "Fertilizer_Usage_kg": 130.0,
    }
    pred_res = predict_single_instance(
        pipeline=bundle["pipeline"],
        inputs=test_inputs,
        feature_stats=bundle["feature_stats"],
        numeric_features=bundle["numeric_features"],
        categorical_features=bundle["categorical_features"],
    )
    assert pred_res["predicted_yield"] > 0
    assert isinstance(pred_res["predicted_yield"], float)

    # 6. Crop Recommendation & Ranking
    can_rec, _, crop_col = can_support_crop_recommendation(bundle)
    assert can_rec is True

    base_conds = {
        "State_Name": "Punjab",
        "Season": "Kharif",
        "Annual_Rainfall_mm": 900.0,
        "Avg_Temperature_C": 28.0,
        "Soil_Type": "Alluvial",
        "Fertilizer_Usage_kg": 120.0,
    }
    rec_res = compare_and_rank_crops(
        pipeline=bundle["pipeline"],
        base_conditions=base_conds,
        crop_col="Crop",
        feature_stats=bundle["feature_stats"],
        numeric_features=bundle["numeric_features"],
        categorical_features=bundle["categorical_features"],
    )
    assert len(rec_res["ranking_df"]) >= 5
    assert rec_res["top_crop"] is not None
    assert rec_res["top_yield"] > 0


def test_end_to_end_excel_with_derived_yield():
    excel_path = "data/crop_production_derive_yield.xlsx"
    assert os.path.exists(excel_path)

    # 1. Load Excel
    df_raw, meta = load_dataset(excel_path)
    assert meta["extension"] == ".xlsx"
    assert "Yield_Ton_Ha" not in df_raw.columns

    # 2. Schema check for yield derivation
    mapping = detect_schema_mappings(df_raw)
    can_derive, msg = can_derive_yield(
        {"area": mapping["area"]["column"], "production": mapping["production"]["column"]},
        df_raw
    )
    assert can_derive is True

    # 3. Clean and Derive Yield
    df_cleaned, summary = clean_dataset(
        df_raw=df_raw,
        derive_yield=True,
        area_col=mapping["area"]["column"],
        prod_col=mapping["production"]["column"],
        yield_col_name="Calculated_Yield",
    )
    assert "Calculated_Yield" in df_cleaned.columns
    assert df_cleaned["Calculated_Yield"].notnull().sum() > 300

    # 4. Train Model on Derived Yield
    features = ["Crop", "Annual_Rainfall_mm", "Avg_Temperature_C", "Fertilizer_Usage_kg"]
    bundle = train_and_compare_models(
        df=df_cleaned,
        target_col="Calculated_Yield",
        feature_cols=features,
        split_method="random",
        test_size=0.2,
        random_state=42,
    )
    assert bundle["best_metrics"]["R2"] > 0.0
