"""
Sample Dataset Generator
Generates realistic agricultural datasets in CSV and Excel formats
for testing, validation, and demo purposes.
"""

import os
import numpy as np
import pandas as pd

def generate_datasets():
    np.random.seed(42)
    os.makedirs("data", exist_ok=True)

    crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Barley", "Soybean", "Groundnut"]
    states = ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat"]
    seasons = ["Kharif", "Rabi", "Whole Year", "Summer"]
    soil_types = ["Alluvial", "Black", "Red", "Laterite", "Loamy"]

    # Base yield parameters (tonnes/ha) per crop
    crop_yield_base = {
        "Rice": (3.5, 0.6),
        "Wheat": (4.2, 0.7),
        "Maize": (3.0, 0.5),
        "Cotton": (1.8, 0.4),
        "Sugarcane": (65.0, 8.0),
        "Barley": (2.8, 0.5),
        "Soybean": (2.1, 0.35),
        "Groundnut": (2.4, 0.4),
    }

    n_samples = 400
    rows = []

    for _ in range(n_samples):
        crop = np.random.choice(crops)
        state = np.random.choice(states)
        season = np.random.choice(seasons)
        soil = np.random.choice(soil_types)
        year = int(np.random.choice(range(2012, 2024)))

        # Environmental variables
        rainfall = round(float(np.random.normal(850, 250)), 1)
        rainfall = max(200.0, rainfall)

        temp = round(float(np.random.normal(26.5, 4.5)), 1)
        temp = max(12.0, min(42.0, temp))

        fertilizer = round(float(np.random.normal(120, 35)), 1)
        fertilizer = max(30.0, fertilizer)

        area = round(float(np.random.exponential(scale=1500) + 100), 1)

        # Realistic Yield physics
        base_mean, base_sd = crop_yield_base[crop]
        # Rainfall effect (diminishing return or optimal window)
        rf_effect = 1.0 + (rainfall - 800) / 3000
        # Temp effect
        temp_effect = 1.0 - abs(temp - 26) / 50
        # Fertilizer effect
        fert_effect = 1.0 + (fertilizer - 100) / 400

        noise = np.random.normal(0, base_sd * 0.25)
        calculated_yield = max(0.5, (base_mean * rf_effect * temp_effect * fert_effect) + noise)
        calculated_yield = round(float(calculated_yield), 2)

        production = round(float(calculated_yield * area), 1)

        rows.append({
            "Crop_Year": year,
            "State_Name": state,
            "Crop": crop,
            "Season": season,
            "Area_Hectares": area,
            "Production_Tonnes": production,
            "Yield_Ton_Ha": calculated_yield,
            "Annual_Rainfall_mm": rainfall,
            "Avg_Temperature_C": temp,
            "Soil_Type": soil,
            "Fertilizer_Usage_kg": fertilizer,
        })

    df = pd.DataFrame(rows)

    # Introduce a few realistic missing values and duplicates to test cleaning
    df_with_noise = df.copy()
    # 5 missing rainfall values
    df_with_noise.loc[np.random.choice(n_samples, 5, replace=False), "Annual_Rainfall_mm"] = np.nan
    # 3 missing soil types
    df_with_noise.loc[np.random.choice(n_samples, 3, replace=False), "Soil_Type"] = np.nan
    # 4 duplicate rows
    dups = df_with_noise.iloc[0:4].copy()
    df_with_noise = pd.concat([df_with_noise, dups], ignore_index=True)

    csv_path = "data/agricultural_yield_dataset.csv"
    df_with_noise.to_csv(csv_path, index=False)
    print(f"Generated CSV: {csv_path} with {len(df_with_noise)} rows.")

    # Excel version without explicit Yield column (to test yield derivation)
    df_no_yield = df.drop(columns=["Yield_Ton_Ha"]).copy()
    excel_path = "data/crop_production_derive_yield.xlsx"
    df_no_yield.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"Generated Excel: {excel_path} with {len(df_no_yield)} rows (ready for Yield derivation testing).")

if __name__ == "__main__":
    generate_datasets()
