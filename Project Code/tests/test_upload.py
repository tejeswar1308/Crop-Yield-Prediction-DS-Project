"""
Tests for dataset loading, validation, statistical summarization, and schema detection.
"""

import io
import pytest
import pandas as pd
from src.dataset_loader import (
    load_dataset,
    compute_dataset_stats,
    DatasetLoadError,
    export_dataframe_to_csv,
    export_dataframe_to_excel
)
from src.schema_detector import detect_schema_mappings, can_derive_yield


def test_load_csv_valid():
    csv_data = "Crop,Year,Yield,Rainfall\nRice,2020,3.5,900\nWheat,2021,4.1,850\n"
    buf = io.BytesIO(csv_data.encode("utf-8"))
    df, meta = load_dataset(buf, "sample.csv")

    assert len(df) == 2
    assert list(df.columns) == ["Crop", "Year", "Yield", "Rainfall"]
    assert meta["raw_rows"] == 2
    assert meta["raw_cols"] == 4


def test_load_excel_valid():
    df_orig = pd.DataFrame({
        "Crop": ["Maize", "Barley"],
        "Area": [120.5, 95.0],
        "Production": [360.0, 280.0]
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_orig.to_excel(writer, index=False)
    buf.seek(0)

    df_loaded, meta = load_dataset(buf, "test_excel.xlsx")
    assert len(df_loaded) == 2
    assert "Crop" in df_loaded.columns
    assert meta["extension"] == ".xlsx"


def test_empty_csv_raises_error():
    buf = io.BytesIO(b"")
    with pytest.raises(DatasetLoadError):
        load_dataset(buf, "empty.csv")


def test_unsupported_file_extension():
    buf = io.BytesIO(b"dummy")
    with pytest.raises(DatasetLoadError):
        load_dataset(buf, "data.pdf")


def test_compute_dataset_stats():
    df = pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Rice"],
        "Yield": [3.5, None, 3.5],
        "Rainfall": [800, 950, 800]
    })
    stats = compute_dataset_stats(df)

    assert stats["total_rows"] == 3
    assert stats["total_cols"] == 3
    assert "Yield" in stats["numeric_cols"]
    assert "Crop" in stats["categorical_cols"]
    assert stats["total_missing_cells"] == 1
    # Check duplicate count
    assert stats["duplicate_rows"] == 1


def test_schema_detection():
    df = pd.DataFrame({
        "Crop_Year": [2018, 2019, 2020],
        "Crop_Name": ["Rice", "Wheat", "Maize"],
        "Yield_Ton_Ha": [3.2, 4.0, 3.1],
        "Annual_Rainfall": [700.0, 850.0, 600.0],
        "State": ["Punjab", "Haryana", "UP"]
    })
    detected = detect_schema_mappings(df)

    assert detected["year"]["column"] == "Crop_Year"
    assert detected["crop"]["column"] == "Crop_Name"
    assert detected["yield"]["column"] == "Yield_Ton_Ha"
    assert detected["rainfall"]["column"] == "Annual_Rainfall"
    assert detected["region"]["column"] == "State"


def test_can_derive_yield():
    df = pd.DataFrame({
        "Area_Hectares": [100.0, 200.0],
        "Production_Tonnes": [400.0, 800.0]
    })
    mapping = {"area": "Area_Hectares", "production": "Production_Tonnes"}
    can_derive, msg = can_derive_yield(mapping, df)
    assert can_derive is True
    assert "Yield can be calculated" in msg
