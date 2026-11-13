"""
Tests for model training, metrics evaluation, single-instance prediction,
and candidate crop recommendation ranking.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.model_trainer import (
    train_and_compare_models,
    save_trained_bundle,
    load_trained_bundle,
    TrainingError
)
from src.predictor import predict_single_instance, validate_inference_inputs
from src.crop_recommender import (
    can_support_crop_recommendation,
    compare_and_rank_crops
)


@pytest.fixture
def sample_agri_df():
    np.random.seed(42)
    crops = ["Rice", "Wheat", "Maize"]
    states = ["Punjab", "Haryana"]
    data = []
    for i in range(60):
        c = crops[i % 3]
        s = states[i % 2]
        yr = 2015 + (i % 6)
        rf = 700 + (i * 5)
        temp = 25.0 + (i % 5)
        yld = 2.0 + (1.5 if c == "Wheat" else 1.0) + (rf * 0.001) + np.random.normal(0, 0.1)
        data.append({
            "Crop": c,
            "State": s,
            "Year": yr,
            "Rainfall": rf,
            "Temperature": temp,
            "Yield": round(yld, 2)
        })
    return pd.DataFrame(data)


def test_train_and_compare_models(sample_agri_df):
    bundle = train_and_compare_models(
        df=sample_agri_df,
        target_col="Yield",
        feature_cols=["Crop", "State", "Year", "Rainfall", "Temperature"],
        year_col="Year",
        split_method="chronological",
        test_size=0.25,
        random_state=42
    )

    assert bundle["best_model_name"] is not None
    assert "test_metrics" in bundle["models_comparison"][bundle["best_model_name"]]
    metrics = bundle["best_metrics"]
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert bundle["split_info"]["method"] == "Chronological"
    assert len(bundle["best_importances"]) > 0


def test_save_and_load_bundle(sample_agri_df):
    bundle = train_and_compare_models(
        df=sample_agri_df,
        target_col="Yield",
        feature_cols=["Crop", "Rainfall"],
        split_method="random",
        test_size=0.2,
        random_state=42
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.joblib")
        save_trained_bundle(bundle, save_path)
        assert os.path.exists(save_path)

        loaded = load_trained_bundle(save_path)
        assert loaded["best_model_name"] == bundle["best_model_name"]
        assert loaded["target_col"] == "Yield"


def test_predict_single_instance(sample_agri_df):
    bundle = train_and_compare_models(
        df=sample_agri_df,
        target_col="Yield",
        feature_cols=["Crop", "Rainfall", "Temperature"],
        split_method="random",
        random_state=42
    )

    inputs = {"Crop": "Wheat", "Rainfall": 800.0, "Temperature": 27.0}
    res = predict_single_instance(
        pipeline=bundle["pipeline"],
        inputs=inputs,
        feature_stats=bundle["feature_stats"],
        numeric_features=bundle["numeric_features"],
        categorical_features=bundle["categorical_features"],
        yield_unit="tonnes/ha"
    )

    assert res["predicted_yield"] > 0
    assert len(res["warnings"]) == 0

    # Test out of range input warning
    extreme_inputs = {"Crop": "Wheat", "Rainfall": 50000.0, "Temperature": 27.0}
    res_ext = predict_single_instance(
        pipeline=bundle["pipeline"],
        inputs=extreme_inputs,
        feature_stats=bundle["feature_stats"],
        numeric_features=bundle["numeric_features"],
        categorical_features=bundle["categorical_features"],
    )
    assert len(res_ext["warnings"]) > 0
    assert "exceeds historical training maximum" in res_ext["warnings"][0]


def test_crop_recommendation(sample_agri_df):
    bundle = train_and_compare_models(
        df=sample_agri_df,
        target_col="Yield",
        feature_cols=["Crop", "Rainfall", "Temperature"],
        random_state=42
    )

    can_rec, msg, crop_col = can_support_crop_recommendation(bundle)
    assert can_rec is True
    assert crop_col == "Crop"

    base_conditions = {"Rainfall": 850.0, "Temperature": 26.0}
    rec_res = compare_and_rank_crops(
        pipeline=bundle["pipeline"],
        base_conditions=base_conditions,
        crop_col="Crop",
        feature_stats=bundle["feature_stats"],
        numeric_features=bundle["numeric_features"],
        categorical_features=bundle["categorical_features"],
        yield_unit="tonnes/ha"
    )

    ranking_df = rec_res["ranking_df"]
    assert len(ranking_df) == 3  # Rice, Wheat, Maize
    assert ranking_df.iloc[0]["Rank"] == 1
    # Check descending order
    assert ranking_df.iloc[0]["Predicted Yield"] >= ranking_df.iloc[1]["Predicted Yield"]
