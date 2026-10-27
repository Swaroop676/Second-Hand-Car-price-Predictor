"""Prepare data, train price models, evaluate them, and save the artifacts."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = DATA_DIR / "used_cars.csv"
ARTIFACT_PATH = MODEL_DIR / "car_price_models.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
TARGET = "selling_price_inr"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "engine_cc",
    "max_power_bhp",
    "seats",
]
CATEGORICAL_FEATURES = [
    "company",
    "model",
    "fuel",
    "transmission",
    "owner",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# These fictional observations are for demonstrating the workflow only.
VEHICLES = [
    ("Maruti", "Swift", 760_000, 1197, 88, 5, ["Petrol", "CNG"]),
    ("Maruti", "Baleno", 920_000, 1197, 89, 5, ["Petrol", "CNG"]),
    ("Maruti", "Dzire", 850_000, 1197, 89, 5, ["Petrol", "CNG"]),
    ("Maruti", "Wagon R", 690_000, 998, 66, 5, ["Petrol", "CNG"]),
    ("Hyundai", "i20", 1_050_000, 1197, 87, 5, ["Petrol", "Diesel"]),
    ("Hyundai", "Creta", 1_650_000, 1497, 113, 5, ["Petrol", "Diesel"]),
    ("Honda", "City", 1_450_000, 1498, 119, 5, ["Petrol", "Diesel"]),
    ("Tata", "Nexon", 1_300_000, 1199, 118, 5, ["Petrol", "Diesel"]),
    ("Tata", "Punch", 950_000, 1199, 87, 5, ["Petrol", "CNG"]),
    ("Toyota", "Innova", 2_250_000, 2393, 148, 7, ["Petrol", "Diesel"]),
    ("Mahindra", "XUV300", 1_450_000, 1497, 115, 5, ["Petrol", "Diesel"]),
    ("Kia", "Seltos", 1_700_000, 1497, 113, 5, ["Petrol", "Diesel"]),
]


def generate_synthetic_dataset(path: Path = DATA_PATH, rows: int = 360) -> pd.DataFrame:
    """Create a repeatable, clearly synthetic dataset when no CSV is present."""
    generator = random.Random(RANDOM_STATE)
    records: list[dict[str, Any]] = []

    for row_number in range(rows):
        company, model, base_price, engine_cc, power, seats, fuels = VEHICLES[
            row_number % len(VEHICLES)
        ]
        year = generator.randint(2012, 2025)
        age = 2026 - year
        km_driven = generator.randint(4_000, 190_000)
        fuel = generator.choice(fuels)
        transmission = generator.choice(["Manual", "Automatic"])
        owner = generator.choices(
            ["First Owner", "Second Owner", "Third Owner", "Fourth & Above"],
            weights=[58, 27, 11, 4],
            k=1,
        )[0]

        age_factor = max(0.25, 1 - 0.075 * age + 0.0018 * age**2)
        distance_factor = max(0.58, 1 - km_driven / 420_000)
        owner_factor = {
            "First Owner": 1.0,
            "Second Owner": 0.94,
            "Third Owner": 0.87,
            "Fourth & Above": 0.80,
        }[owner]
        fuel_factor = 1.04 if fuel == "Diesel" else 0.98 if fuel == "CNG" else 1.0
        transmission_factor = 1.06 if transmission == "Automatic" else 1.0
        variation = generator.uniform(0.92, 1.08)
        selling_price = (
            base_price
            * age_factor
            * distance_factor
            * owner_factor
            * fuel_factor
            * transmission_factor
            * variation
        )

        records.append(
            {
                "company": company,
                "model": model,
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "transmission": transmission,
                "owner": owner,
                "engine_cc": engine_cc,
                "max_power_bhp": power,
                "seats": seats,
                "selling_price_inr": round(selling_price / 100) * 100,
                "data_source": "synthetic_demo",
            }
        )

    dataset = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    return dataset


def load_and_clean_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, bool]:
    """Load the local dataset, creating synthetic demo data if it is absent."""
    is_synthetic = not path.exists()
    dataset = generate_synthetic_dataset(path) if is_synthetic else pd.read_csv(path)
    if "data_source" in dataset.columns:
        is_synthetic = bool(dataset["data_source"].eq("synthetic_demo").all())

    missing_columns = sorted(set(FEATURES + [TARGET]) - set(dataset.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    dataset = dataset[FEATURES + [TARGET]].drop_duplicates().copy()
    dataset[TARGET] = pd.to_numeric(dataset[TARGET], errors="coerce")
    dataset = dataset.dropna(subset=[TARGET])
    if len(dataset) < 10:
        raise ValueError("At least 10 usable, non-duplicate rows are required.")
    return dataset, is_synthetic


def build_preprocessor(polynomial: bool = False) -> ColumnTransformer:
    """Build train-fitted imputing, scaling, and encoding steps."""
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if polynomial:
        numeric_steps.append(
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False))
        )

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, actual: pd.Series
) -> dict[str, float]:
    predictions = model.predict(features)
    mse = mean_squared_error(actual, predictions)
    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(actual, predictions)),
    }


def train_and_save() -> dict[str, Any]:
    dataset, is_synthetic = load_and_clean_data()
    features = dataset[FEATURES]
    target = dataset[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("regressor", LinearRegression()),
            ]
        ),
        "Polynomial Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(polynomial=True)),
                ("regressor", LinearRegression()),
            ]
        ),
    }
    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, list[float]] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics[name] = evaluate_model(model, x_test, y_test)
        predictions[name] = model.predict(x_test).astype(float).tolist()

    # RMSE is the main comparison measure because it penalizes large misses.
    best_model_name = min(metrics, key=lambda name: metrics[name]["rmse"])
    artifact = {
        "models": models,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "test_actual": y_test.astype(float).tolist(),
        "test_predictions": predictions,
        "feature_names": FEATURES,
        "is_synthetic": is_synthetic,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    artifact = train_and_save()
    data_label = "synthetic demo data" if artifact["is_synthetic"] else "local dataset"
    print(f"Training complete using {data_label}.")
    print(f"Best test-set model by RMSE: {artifact['best_model_name']}")
    for name, scores in artifact["metrics"].items():
        print(
            f"{name}: MAE={scores['mae']:,.0f}, "
            f"RMSE={scores['rmse']:,.0f}, R2={scores['r2']:.3f}"
        )
    print(f"Saved model artifacts to: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
