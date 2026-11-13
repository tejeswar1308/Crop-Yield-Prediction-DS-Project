"""
Data Cleaner and Preprocessing Module
Preserves raw data integrity, cleans dirty strings, detects/handles duplicates and outliers,
dynamically computes yield when possible, and records an auditable operations summary.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Cleans a series that should be numeric by removing commas, whitespace, and non-numeric chars.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    # Convert to string and clean
    cleaned = series.astype(str).str.replace(r"[,\s]", "", regex=True)
    # Convert anything that isn't a valid float or int to NaN
    return pd.to_numeric(cleaned, errors="coerce")


def clean_categorical_series(series: pd.Series) -> pd.Series:
    """
    Trims whitespace and standardizes empty strings to NaN for categorical columns.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan, "NULL": np.nan, "N/A": np.nan})
    return cleaned


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> Tuple[int, float, float, pd.Series]:
    """
    Calculates IQR outliers for a numeric series.
    Returns: (outlier_count, lower_bound, upper_bound, boolean_mask)
    """
    valid = series.dropna()
    if len(valid) < 4:
        return 0, -np.inf, np.inf, pd.Series(False, index=series.index)

    q25 = valid.quantile(0.25)
    q75 = valid.quantile(0.75)
    iqr = q75 - q25

    if iqr == 0:
        return 0, float(q25), float(q75), pd.Series(False, index=series.index)

    lower_bound = float(q25 - (multiplier * iqr))
    upper_bound = float(q75 + (multiplier * iqr))

    mask = (series < lower_bound) | (series > upper_bound)
    return int(mask.sum()), lower_bound, upper_bound, mask


def clean_dataset(
    df_raw: pd.DataFrame,
    target_col: Optional[str] = None,
    remove_duplicates: bool = True,
    drop_missing_target: bool = False,
    derive_yield: bool = False,
    area_col: Optional[str] = None,
    prod_col: Optional[str] = None,
    yield_col_name: str = "Yield",
    outlier_action: str = "inspect",  # "inspect", "cap_iqr", "drop_extreme"
    selected_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs comprehensive, leakage-safe dataset cleaning.
    The original raw dataframe is untouched.

    Returns:
        (cleaned_df, cleaning_summary_dict)
    """
    df = df_raw.copy(deep=True)
    initial_rows = len(df)
    initial_cols = len(df.columns)
    audit_log: List[Dict[str, Any]] = []

    # 1. Clean column headers
    df.columns = [str(c).strip() for c in df.columns]

    # 2. Derive Yield if requested
    if derive_yield and area_col and prod_col:
        if area_col in df.columns and prod_col in df.columns:
            # Clean both numeric series
            area_num = clean_numeric_series(df[area_col])
            prod_num = clean_numeric_series(df[prod_col])
            df[area_col] = area_num
            df[prod_col] = prod_num

            # Calculate Yield: Production / Area, invalid where Area <= 0 or NaN
            valid_calc = (area_num > 0) & (prod_num.notnull()) & (area_num.notnull())
            new_yield = pd.Series(np.nan, index=df.index, dtype=float)
            new_yield[valid_calc] = prod_num[valid_calc] / area_num[valid_calc]
            df[yield_col_name] = new_yield

            audit_log.append({
                "step": "Derive Yield Column",
                "status": "Success",
                "details": (
                    f"Computed '{yield_col_name}' = '{prod_col}' / '{area_col}'. "
                    f"Generated {valid_calc.sum()} valid values (excluded non-positive area)."
                ),
                "rows_affected": int(valid_calc.sum()),
            })
            if target_col is None or target_col == "":
                target_col = yield_col_name
        else:
            audit_log.append({
                "step": "Derive Yield Column",
                "status": "Skipped",
                "details": f"Specified Area ('{area_col}') or Production ('{prod_col}') not found.",
                "rows_affected": 0,
            })

    # 3. Clean string whitespace and empty values in categorical columns
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = clean_categorical_series(df[col])

    # 4. Remove duplicate rows if enabled
    if remove_duplicates:
        dups_count = int(df.duplicated().sum())
        if dups_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            audit_log.append({
                "step": "Remove Duplicates",
                "status": "Applied",
                "details": f"Removed {dups_count} identical duplicate rows.",
                "rows_affected": dups_count,
            })
        else:
            audit_log.append({
                "step": "Remove Duplicates",
                "status": "Verified",
                "details": "No duplicate rows found.",
                "rows_affected": 0,
            })

    # 5. Handle Target column missing values
    if target_col and target_col in df.columns:
        # Coerce target to numeric if needed
        df[target_col] = clean_numeric_series(df[target_col])
        missing_target = int(df[target_col].isnull().sum())
        if missing_target > 0:
            if drop_missing_target:
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                audit_log.append({
                    "step": "Handle Missing Target",
                    "status": "Applied",
                    "details": (
                        f"Dropped {missing_target} rows missing the target variable '{target_col}'. "
                        "Silently fabricating target values would induce label leakage and distort evaluation."
                    ),
                    "rows_affected": missing_target,
                })
            else:
                audit_log.append({
                    "step": "Handle Missing Target",
                    "status": "Retained",
                    "details": f"Kept {missing_target} missing target rows (requires filtering before ML training).",
                    "rows_affected": 0,
                })

    # 6. Outlier Analysis & Handling for numeric columns
    outlier_reports: Dict[str, Dict[str, Any]] = {}
    cols_to_check = selected_features or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for col in cols_to_check:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            cnt, low_b, high_b, mask = detect_outliers_iqr(df[col], multiplier=1.5)
            outlier_reports[col] = {
                "outlier_count": cnt,
                "lower_bound": round(low_b, 3) if np.isfinite(low_b) else None,
                "upper_bound": round(high_b, 3) if np.isfinite(high_b) else None,
            }

            if cnt > 0:
                if outlier_action == "cap_iqr":
                    df[col] = df[col].clip(lower=low_b, upper=high_b)
                    audit_log.append({
                        "step": f"Outlier Capping ({col})",
                        "status": "Applied",
                        "details": f"Capped {cnt} values outside IQR bounds [{low_b:.2f}, {high_b:.2f}].",
                        "rows_affected": cnt,
                    })
                elif outlier_action == "drop_extreme":
                    # Extreme multiplier 3.0
                    _, ex_low, ex_high, ex_mask = detect_outliers_iqr(df[col], multiplier=3.0)
                    extreme_count = int(ex_mask.sum())
                    if extreme_count > 0:
                        df = df[~ex_mask].reset_index(drop=True)
                        audit_log.append({
                            "step": f"Drop Extreme Outliers ({col})",
                            "status": "Applied",
                            "details": f"Removed {extreme_count} rows with values beyond 3x IQR [{ex_low:.2f}, {ex_high:.2f}].",
                            "rows_affected": extreme_count,
                        })

    final_rows = len(df)
    final_cols = len(df.columns)

    summary = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "rows_removed": initial_rows - final_rows,
        "initial_cols": initial_cols,
        "final_cols": final_cols,
        "target_col": target_col,
        "audit_log": audit_log,
        "outlier_reports": outlier_reports,
    }

    return df, summary
