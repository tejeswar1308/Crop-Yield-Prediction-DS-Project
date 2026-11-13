"""
Tests for data cleaning, duplicate removal, yield derivation, and outlier detection.
"""

import numpy as np
import pandas as pd
import pytest
from src.data_cleaner import (
    clean_dataset,
    clean_numeric_series,
    clean_categorical_series,
    detect_outliers_iqr
)


def test_clean_numeric_series_with_commas_and_text():
    dirty_series = pd.Series(["1,200", " 3,450.50 ", "invalid", "500"])
    cleaned = clean_numeric_series(dirty_series)
    assert cleaned.iloc[0] == 1200.0
    assert cleaned.iloc[1] == 3450.50
    assert np.isnan(cleaned.iloc[2])
    assert cleaned.iloc[3] == 500.0


def test_clean_categorical_series():
    dirty_series = pd.Series([" Rice ", "", "nan", "Wheat", "NULL"])
    cleaned = clean_categorical_series(dirty_series)
    assert cleaned.iloc[0] == "Rice"
    assert pd.isna(cleaned.iloc[1])
    assert pd.isna(cleaned.iloc[2])
    assert cleaned.iloc[3] == "Wheat"
    assert pd.isna(cleaned.iloc[4])


def test_clean_dataset_removes_duplicates_and_preserves_raw():
    raw_df = pd.DataFrame({
        "Crop": ["Rice", "Rice", "Wheat"],
        "Yield": [3.5, 3.5, 4.0]
    })
    raw_copy = raw_df.copy(deep=True)

    cleaned_df, summary = clean_dataset(raw_df, target_col="Yield", remove_duplicates=True)

    # Check raw_df was not mutated
    pd.testing.assert_frame_equal(raw_df, raw_copy)
    assert len(cleaned_df) == 2
    assert summary["rows_removed"] == 1


def test_derive_yield_calculation():
    raw_df = pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize"],
        "Area": [100.0, 200.0, 0.0],  # 0.0 area should not divide by zero
        "Production": [350.0, 800.0, 50.0]
    })

    cleaned_df, summary = clean_dataset(
        raw_df,
        derive_yield=True,
        area_col="Area",
        prod_col="Production",
        yield_col_name="Derived_Yield"
    )

    assert "Derived_Yield" in cleaned_df.columns
    assert cleaned_df.loc[0, "Derived_Yield"] == 3.5  # 350 / 100
    assert cleaned_df.loc[1, "Derived_Yield"] == 4.0  # 800 / 200
    # Zero area becomes NaN
    assert pd.isna(cleaned_df.loc[2, "Derived_Yield"])


def test_detect_outliers_iqr():
    # Normal distribution with one obvious extreme outlier
    data = pd.Series([10.0, 11.0, 10.5, 12.0, 9.8, 10.2, 11.5, 100.0])
    count, low, high, mask = detect_outliers_iqr(data)
    assert count == 1
    assert mask.iloc[-1] is True or mask.iloc[-1] == True
