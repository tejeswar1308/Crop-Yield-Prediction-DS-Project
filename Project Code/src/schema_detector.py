"""
Schema Detector Module
Dynamically analyzes dataset columns and suggests agricultural semantic roles
(Crop, Yield, Year, Region, Season, Cultivated Area, Production, Rainfall, Temperature, Soil, Fertilizer, Irrigation).
Allows users to verify, adjust, and confirm mappings.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


# Definition of standard agricultural roles and their keywords/patterns
ROLE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "yield": {
        "label": "Crop Yield (Target)",
        "patterns": [
            r"^yield$", r".*crop.*yield.*", r".*yield.*ha.*", r".*yield.*acre.*",
            r"^productivity$", r"^yield_ton.*", r"^yield_kg.*"
        ],
        "keywords": ["yield", "productivity", "output_per_ha", "yield_tonnes"],
        "prefer_numeric": True,
        "is_primary_target": True,
    },
    "crop": {
        "label": "Crop Name / Type",
        "patterns": [r"^crop$", r"^crop_name$", r"^commodity$", r"^crop_type$", r"^crop.*title$"],
        "keywords": ["crop", "crop_name", "commodity", "produce", "crop_type"],
        "prefer_numeric": False,
        "is_primary_target": False,
    },
    "area": {
        "label": "Cultivated Area",
        "patterns": [
            r"^area$", r".*cultivated.*area.*", r".*harvested.*area.*",
            r"^acreage$", r".*land.*area.*", r"^gross_area$"
        ],
        "keywords": ["area", "cultivated", "acreage", "hectare", "acres", "ha"],
        "prefer_numeric": True,
        "is_primary_target": False,
    },
    "production": {
        "label": "Crop Production (Total)",
        "patterns": [
            r"^production$", r".*total.*production.*", r"^produce$",
            r"^output$", r"^quantity_produced$", r"^production_tonnes$"
        ],
        "keywords": ["production", "produce", "harvest_volume", "total_output", "metric_tons"],
        "prefer_numeric": True,
        "is_primary_target": False,
    },
    "year": {
        "label": "Harvest Year / Time",
        "patterns": [r"^year$", r"^crop_year$", r"^yr$", r"^harvest_year$", r"^calendar_year$"],
        "keywords": ["year", "crop_year", "harvest_year"],
        "prefer_numeric": True,
        "is_primary_target": False,
    },
    "region": {
        "label": "State / District / Region",
        "patterns": [
            r"^state$", r"^state_name$", r"^district$", r"^district_name$",
            r"^region$", r"^county$", r"^province$", r"^location$", r"^zone$"
        ],
        "keywords": ["state", "district", "region", "county", "province", "location", "zone"],
        "prefer_numeric": False,
        "is_primary_target": False,
    },
    "season": {
        "label": "Growing Season",
        "patterns": [r"^season$", r"^crop_season$", r"^agricultural_season$"],
        "keywords": ["season", "crop_season", "kharif", "rabi", "zaid"],
        "prefer_numeric": False,
        "is_primary_target": False,
    },
    "rainfall": {
        "label": "Rainfall / Precipitation",
        "patterns": [r"^rainfall$", r"^precipitation$", r".*annual.*rainfall.*", r"^rain.*mm$", r"^precip$"],
        "keywords": ["rainfall", "rain", "precipitation", "precip", "annual_rainfall"],
        "prefer_numeric": True,
        "is_primary_target": False,
    },
    "temperature": {
        "label": "Temperature",
        "patterns": [
            r"^temperature$", r"^temp$", r".*avg.*temp.*", r".*mean.*temp.*",
            r"^temp_celsius$", r"^temp_f$"
        ],
        "keywords": ["temperature", "temp", "celsius", "fahrenheit"],
        "prefer_numeric": True,
        "is_primary_target": False,
    },
    "soil": {
        "label": "Soil Type / Properties (pH, N, P, K)",
        "patterns": [
            r"^soil$", r"^soil_type$", r"^soil_ph$", r"^ph$",
            r"^nitrogen$", r"^phosphorus$", r"^potassium$", r"^n$", r"^p$", r"^k$"
        ],
        "keywords": ["soil", "ph", "nitrogen", "phosphorus", "potassium", "soil_type", "npk"],
        "prefer_numeric": None,  # Can be numeric or categorical
        "is_primary_target": False,
    },
    "fertilizer": {
        "label": "Fertilizer Usage",
        "patterns": [r"^fertilizer$", r"^fertilizer_usage$", r"^urea$", r"^npk_fertilizer$"],
        "keywords": ["fertilizer", "urea", "manure", "npk_usage"],
        "prefer_numeric": True,
        "is_primary_target": False,
    },
    "irrigation": {
        "label": "Irrigation / Water Source",
        "patterns": [r"^irrigation$", r"^irrigation_type$", r"^irrigated_area$", r"^water_supply$"],
        "keywords": ["irrigation", "water_source", "tube_well", "canal"],
        "prefer_numeric": None,
        "is_primary_target": False,
    },
}


def score_column_for_role(col_name: str, series: pd.Series, role_key: str) -> float:
    """
    Computes a match score from 0.0 to 1.0 indicating how likely col_name represents role_key.
    """
    clean_name = re.sub(r"[_\-\s]+", "_", col_name.strip().lower())
    rule = ROLE_DEFINITIONS[role_key]
    score = 0.0

    # 1. Exact regex pattern match
    for pattern in rule["patterns"]:
        if re.match(pattern, clean_name):
            score += 0.65
            break

    # 2. Substring keyword match
    if score == 0.0:
        for kw in rule["keywords"]:
            if kw in clean_name:
                score += 0.40
                break

    # 3. Data type match / penalty
    is_numeric = pd.api.types.is_numeric_dtype(series)
    prefer_numeric = rule["prefer_numeric"]

    if prefer_numeric is True:
        if is_numeric:
            score += 0.20
        else:
            # Check if strings can be coerced to numeric
            try:
                coerced = pd.to_numeric(series.dropna().head(20), errors="coerce")
                if coerced.notnull().mean() > 0.8:
                    score += 0.15
                else:
                    score = max(0.0, score - 0.35)
            except Exception:
                score = max(0.0, score - 0.35)

    elif prefer_numeric is False:
        if not is_numeric:
            score += 0.20
        else:
            score = max(0.0, score - 0.30)

    # 4. Domain specific data heuristics
    if role_key == "year":
        # Check if values look like realistic calendar years (e.g. 1950 - 2035)
        try:
            sample = pd.to_numeric(series.dropna().head(50), errors="coerce").dropna()
            if not sample.empty and sample.between(1900, 2100).mean() > 0.85:
                score += 0.25
        except Exception:
            pass

    if role_key == "crop" and not is_numeric:
        # Check unique ratio (moderate unique count, typical of crops: 2 to 200)
        n_unique = series.nunique(dropna=True)
        if 2 <= n_unique <= 300:
            score += 0.15

    if role_key == "season" and not is_numeric:
        n_unique = series.nunique(dropna=True)
        if 2 <= n_unique <= 12:
            score += 0.20

    if role_key == "yield" and is_numeric:
        # Yield is typically non-negative
        sample = series.dropna().head(50)
        if (sample >= 0).mean() > 0.95:
            score += 0.10

    return min(1.0, round(score, 2))


def detect_schema_mappings(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyzes DataFrame columns and auto-detects roles.
    Returns:
        Dict mapping role_key -> {
            "column": matched_col_or_None,
            "confidence": float,
            "label": str,
            "reason": str
        }
    """
    detected: Dict[str, Dict[str, Any]] = {}
    assigned_cols = set()

    # Calculate matrix of scores: roles x columns
    score_matrix: List[Tuple[float, str, str]] = []
    for col in df.columns:
        series = df[col]
        for role_key in ROLE_DEFINITIONS.keys():
            sc = score_column_for_role(col, series, role_key)
            if sc >= 0.40:
                score_matrix.append((sc, role_key, col))

    # Sort candidates by score descending
    score_matrix.sort(key=lambda x: x[0], reverse=True)

    assigned_roles = set()

    for sc, role_key, col in score_matrix:
        # Each primary role assigned once, and each column assigned at most once
        # (unless soil, which could have multiple columns like N, P, K, pH)
        if role_key not in assigned_roles and col not in assigned_cols:
            assigned_roles.add(role_key)
            assigned_cols.add(col)
            detected[role_key] = {
                "column": col,
                "confidence": sc,
                "label": ROLE_DEFINITIONS[role_key]["label"],
                "reason": f"Matched pattern/keyword '{col}' with score {sc*100:.0f}%",
            }

    # Fill in unmapped roles
    for role_key, meta in ROLE_DEFINITIONS.items():
        if role_key not in detected:
            detected[role_key] = {
                "column": None,
                "confidence": 0.0,
                "label": meta["label"],
                "reason": "No confident match detected",
            }

    return detected


def can_derive_yield(mapping: Dict[str, Optional[str]], df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Checks if Crop Yield can be dynamically derived via Production / Cultivated Area.
    """
    area_col = mapping.get("area")
    prod_col = mapping.get("production")

    if not area_col or area_col not in df.columns:
        return False, "Cultivated Area column is not specified or not found in dataset."
    if not prod_col or prod_col not in df.columns:
        return False, "Crop Production column is not specified or not found in dataset."

    is_area_num = pd.api.types.is_numeric_dtype(df[area_col])
    is_prod_num = pd.api.types.is_numeric_dtype(df[prod_col])

    if not is_area_num or not is_prod_num:
        return False, "Both Area and Production columns must be numeric to calculate Yield."

    return True, f"Yield can be calculated as: {prod_col} / {area_col}"
