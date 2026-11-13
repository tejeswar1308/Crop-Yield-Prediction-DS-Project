"""
Dataset Loader Module
Handles loading, format detection, basic validation, and statistical summarization
for user-uploaded CSV and Excel agricultural datasets.
"""

from __future__ import annotations
import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DatasetLoadError(Exception):
    """Custom exception raised when a dataset cannot be loaded or validated."""
    pass


def load_dataset(
    file_source: Union[str, io.BytesIO, Any],
    file_name: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a dataset from a file path or file-like buffer (Streamlit UploadedFile).
    Supports CSV (.csv) and Excel (.xlsx, .xls) files.

    Returns:
        Tuple of (pd.DataFrame, load_metadata_dict)
    """
    if file_name is None:
        if hasattr(file_source, "name"):
            file_name = file_source.name
        elif isinstance(file_source, str):
            file_name = os.path.basename(file_source)
        else:
            file_name = "uploaded_dataset.csv"

    ext = os.path.splitext(file_name)[1].lower()
    df: Optional[pd.DataFrame] = None
    detected_encoding = "utf-8"

    if ext in [".csv", ".txt"]:
        # Try multiple encodings for robust CSV reading
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        last_error = None

        # Reset pointer if file-like
        if hasattr(file_source, "seek"):
            file_source.seek(0)

        for enc in encodings:
            try:
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                # Let pandas auto-detect delimiter via sep=None or standard comma
                try:
                    df = pd.read_csv(file_source, encoding=enc, sep=None, engine="python")
                except Exception:
                    if hasattr(file_source, "seek"):
                        file_source.seek(0)
                    df = pd.read_csv(file_source, encoding=enc)
                detected_encoding = enc
                break
            except Exception as e:
                last_error = e
                continue

        if df is None:
            raise DatasetLoadError(
                f"Failed to read CSV file '{file_name}'. Ensure it is a valid delimiter-separated file. "
                f"Underlying error: {last_error}"
            )

    elif ext in [".xlsx", ".xls"]:
        try:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            df = pd.read_excel(file_source, engine="openpyxl")
            detected_encoding = "excel-binary"
        except Exception as e:
            raise DatasetLoadError(
                f"Failed to read Excel file '{file_name}'. Make sure the workbook is not corrupted "
                f"and openpyxl is installed. Underlying error: {e}"
            )
    else:
        raise DatasetLoadError(
            f"Unsupported file format '{ext}'. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file."
        )

    # Basic validations
    if df is None or df.empty:
        raise DatasetLoadError("The uploaded file is empty or contains no records.")

    if len(df.columns) == 0:
        raise DatasetLoadError("The uploaded file does not contain any recognizable columns.")

    # Clean column headers: strip whitespace, stringify
    df.columns = [str(col).strip() for col in df.columns]

    metadata = {
        "file_name": file_name,
        "extension": ext,
        "encoding": detected_encoding,
        "raw_rows": int(len(df)),
        "raw_cols": int(len(df.columns)),
    }

    return df, metadata


def compute_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comprehensive structural and statistical metrics for the dataset.
    """
    total_rows = int(len(df))
    total_cols = int(len(df.columns))

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    total_cells = total_rows * total_cols
    null_pct = round((total_nulls / total_cells * 100), 2) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []

    column_details: List[Dict[str, Any]] = []

    for col in df.columns:
        col_series = df[col]
        dtype_str = str(col_series.dtype)
        is_num = pd.api.types.is_numeric_dtype(col_series)
        is_dt = pd.api.types.is_datetime64_any_dtype(col_series)

        if is_num:
            numeric_cols.append(col)
            category = "Numeric"
        elif is_dt:
            datetime_cols.append(col)
            category = "Datetime"
        else:
            categorical_cols.append(col)
            category = "Categorical"

        col_nulls = int(null_counts[col])
        col_null_pct = round((col_nulls / total_rows * 100), 2) if total_rows > 0 else 0.0
        unique_cnt = int(col_series.nunique(dropna=True))

        # Sample values as readable strings
        sample_vals = col_series.dropna().head(3).tolist()
        sample_str = ", ".join([str(v) for v in sample_vals]) if sample_vals else "N/A"

        column_details.append({
            "Column": col,
            "Type": dtype_str,
            "Category": category,
            "Non-Null Count": total_rows - col_nulls,
            "Missing Count": col_nulls,
            "Missing %": col_null_pct,
            "Unique Values": unique_cnt,
            "Sample": sample_str,
        })

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "total_missing_cells": total_nulls,
        "overall_missing_pct": null_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "column_details": column_details,
    }


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Converts a pandas DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def export_dataframe_to_excel(df: pd.DataFrame) -> bytes:
    """Converts a pandas DataFrame to Excel bytes for download."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    buffer.seek(0)
    return buffer.getvalue()
