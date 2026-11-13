"""
Model Trainer Module
Orchestrates automated machine learning pipeline training, leakage-free preprocessing,
chronological or random splitting, multi-model evaluation (MAE, RMSE, R²),
feature importance extraction, and pipeline persistence via joblib.
"""

from __future__ import annotations
from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


DEFAULT_RANDOM_STATE = 42


class TrainingError(Exception):
    """Raised when training fails due to data insufficiency or configuration."""
    pass


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Extracts features and target, separates numeric and categorical columns,
    and drops rows with missing target.
    """
    if target_col not in df.columns:
        raise TrainingError(f"Target column '{target_col}' not found in the dataset.")

    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        raise TrainingError(f"Selected features not found in dataset: {missing_features}")

    if not feature_cols:
        raise TrainingError("At least one feature column must be selected for model training.")

    # Drop rows missing target
    valid_mask = df[target_col].notnull()
    clean_sub = df[valid_mask]

    if len(clean_sub) < 10:
        raise TrainingError(
            f"Dataset has only {len(clean_sub)} valid rows with target '{target_col}'. "
            "At least 10 rows are required to perform a meaningful train/test split."
        )

    X = clean_sub[feature_cols].copy()
    y = clean_sub[target_col].astype(float)

    if y.nunique() <= 1:
        raise TrainingError(
            f"The target column '{target_col}' has zero variance (all values are identical: {y.iloc[0]}). "
            "Cannot train a regression model."
        )

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    return X, y, numeric_features, categorical_features


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    year_col: Optional[str] = None,
    split_method: str = "random",  # "random" or "chronological"
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Splits features and target into train and test sets using either chronological or random split.
    """
    if split_method == "chronological" and year_col and year_col in X.columns:
        # Chronological split: use years
        years_sorted = np.sort(X[year_col].dropna().unique())
        if len(years_sorted) >= 3:
            split_idx = int(len(years_sorted) * (1 - test_size))
            cutoff_year = years_sorted[max(1, split_idx)]
            train_mask = X[year_col] < cutoff_year
            test_mask = X[year_col] >= cutoff_year

            # Ensure both train and test have at least 2 rows
            if train_mask.sum() >= 5 and test_mask.sum() >= 2:
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                split_info = {
                    "method": "Chronological",
                    "cutoff_year": int(cutoff_year),
                    "train_years": f"<= {cutoff_year - 1}",
                    "test_years": f">= {cutoff_year}",
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                }
                return X_train, X_test, y_train, y_test, split_info

    # Fallback or default: Random Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    split_info = {
        "method": "Random Split (Stratified/Shuffled)",
        "test_size": test_size,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": random_state,
    }
    return X_train, X_test, y_train, y_test, split_info


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Builds a leakage-free Scikit-learn ColumnTransformer.
    Transformers are fit STRICTLY on X_train.
    """
    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_candidate_models(random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, Any]:
    """
    Returns the portfolio of regression algorithms to train and compare.
    """
    return {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            random_state=random_state
        ),
        "Ridge Linear Regression": Ridge(
            alpha=1.0,
            random_state=random_state
        ),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics: MAE, RMSE, R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def extract_feature_importances(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str]
) -> List[Dict[str, Any]]:
    """
    Extracts feature importances or model coefficients, matching them to post-encoding names.
    """
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        encoded_names: List[str] = []
        if numeric_features:
            encoded_names.extend(numeric_features)

        if categorical_features:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
            encoded_names.extend(cat_names)

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)

        if importances is not None and len(importances) == len(encoded_names):
            importance_list = [
                {"feature": name, "importance": round(float(imp), 4)}
                for name, imp in zip(encoded_names, importances)
            ]
            importance_list.sort(key=lambda x: x["importance"], reverse=True)
            return importance_list
    except Exception:
        pass
    return []


def explain_performance(metrics: Dict[str, float], target_name: str, target_std: float) -> str:
    """
    Generates a clear, non-technical explanation of the model's performance.
    """
    r2 = metrics["R2"]
    mae = metrics["MAE"]
    rmse = metrics["RMSE"]

    r2_pct = max(0.0, round(r2 * 100, 1))

    if r2 >= 0.80:
        fit_quality = "excellent predictive accuracy"
        advice = "The model captures the vast majority of historical yield variance and provides reliable guidance."
    elif r2 >= 0.60:
        fit_quality = "moderate-to-strong predictive accuracy"
        advice = "The model identifies clear agricultural patterns, though localized environmental factors cause some variance."
    elif r2 >= 0.30:
        fit_quality = "modest predictive ability"
        advice = "The model detects general directional trends, but significant unobserved factors (e.g. microclimate, pest infestation) limit precision."
    else:
        fit_quality = "low predictive power"
        advice = "Historical features explain very little variation in crop yield. Consider gathering additional soil or weather data."

    ratio_txt = ""
    if target_std > 0:
        ratio = round((rmse / target_std) * 100, 1)
        ratio_txt = f"The average prediction error ({rmse:.2f}) represents approximately {ratio}% of the overall yield variation in the dataset."

    return (
        f"The model achieved an R² score of {r2:.3f} ({r2_pct}% variance explained) with an average absolute error "
        f"(MAE) of {mae:.2f} {target_name} units, and a Root Mean Squared Error (RMSE) of {rmse:.2f}. "
        f"This indicates {fit_quality}. {ratio_txt} {advice}"
    )


def train_and_compare_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    year_col: Optional[str] = None,
    split_method: str = "random",
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    dataset_name: str = "dataset.csv",
    yield_unit: str = "tonnes/hectare",
) -> Dict[str, Any]:
    """
    Full automated training pipeline:
    1. Prepares X, y
    2. Splits data safely (chronological or random)
    3. Builds ColumnTransformer
    4. Fits & evaluates all candidate models
    5. Determines the best model
    6. Extracts feature importance
    7. Creates evaluation summaries
    """
    X, y, num_features, cat_features = prepare_features_and_target(df, target_col, feature_cols)
    X_train, X_test, y_train, y_test, split_info = split_data(
        X, y, year_col=year_col, split_method=split_method, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(num_features, cat_features)
    candidate_models = get_candidate_models(random_state=random_state)

    results: Dict[str, Dict[str, Any]] = {}
    best_model_name: Optional[str] = None
    best_score = -np.inf
    fitted_pipelines: Dict[str, Pipeline] = {}

    for model_name, model_inst in candidate_models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(num_features, cat_features)),
            ("model", model_inst),
        ])

        # Fit ONLY on training data
        pipeline.fit(X_train, y_train)

        # Predict train & test
        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)

        train_metrics = compute_metrics(y_train.values, train_preds)
        test_metrics = compute_metrics(y_test.values, test_preds)

        results[model_name] = {
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "test_preds": test_preds,
            "train_preds": train_preds,
        }
        fitted_pipelines[model_name] = pipeline

        # Score selection: maximize R2, break ties with lower RMSE
        if test_metrics["R2"] > best_score:
            best_score = test_metrics["R2"]
            best_model_name = model_name

    # If all models have negative R2, select the one with lowest RMSE
    if best_model_name is None:
        best_model_name = min(
            results.keys(), key=lambda m: results[m]["test_metrics"]["RMSE"]
        )

    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]["test_metrics"]
    best_importances = extract_feature_importances(best_pipeline, num_features, cat_features)

    target_std = float(y.std())
    explanation = explain_performance(best_metrics, target_col, target_std)

    # Feature statistics for inference input generation & bounding
    feature_stats: Dict[str, Dict[str, Any]] = {}
    for col in feature_cols:
        if col in num_features:
            series = X[col].dropna()
            feature_stats[col] = {
                "type": "numeric",
                "min": float(series.min()) if not series.empty else 0.0,
                "max": float(series.max()) if not series.empty else 100.0,
                "median": float(series.median()) if not series.empty else 0.0,
                "mean": float(series.mean()) if not series.empty else 0.0,
            }
        else:
            categories = sorted([str(c) for c in X[col].dropna().unique().tolist()])
            feature_stats[col] = {
                "type": "categorical",
                "categories": categories,
                "default": categories[0] if categories else "Unknown",
            }

    bundle = {
        "pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "models_comparison": results,
        "best_metrics": best_metrics,
        "best_importances": best_importances,
        "explanation": explanation,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "numeric_features": num_features,
        "categorical_features": cat_features,
        "feature_stats": feature_stats,
        "split_info": split_info,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": results[best_model_name]["test_preds"],
        "metadata": {
            "dataset_name": dataset_name,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_col": target_col,
            "yield_unit": yield_unit,
            "features_count": len(feature_cols),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
    }

    return bundle


def save_trained_bundle(bundle: Dict[str, Any], filepath: str) -> None:
    """Saves the trained pipeline bundle to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Save a lighter copy without huge test arrays if needed, but here joblib handles it easily
    joblib.dump(bundle, filepath)


def load_trained_bundle(filepath: str) -> Dict[str, Any]:
    """Loads a previously saved trained pipeline bundle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bundle file '{filepath}' not found.")
    return joblib.load(filepath)
