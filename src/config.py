"""Centralized configuration constants and typed schemas for training and serving."""

from __future__ import annotations

from pathlib import Path
from typing import Final

# Deterministic seed used across data splits, cross-validation, and model estimators.
SEED: Final[int] = 42

# Project directory layout resolved relative to repository root.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "heart_disease_uci.csv"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
MODEL_ARTIFACT_PATH: Final[Path] = MODELS_DIR / "model_pipeline.bin"
METRICS_REPORT_PATH: Final[Path] = MODELS_DIR / "metrics_report.json"

TARGET_COLUMN: Final[str] = "heart_disease"

NUMERICAL_FEATURES: Final[list[str]] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
    "clinical_center",
]

ALL_FEATURE_COLUMNS: Final[list[str]] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Indicators explicitly preserved so tree and linear models capture informative missingness.
MISSING_INDICATOR_COLUMNS: Final[list[str]] = [
    "chol",
    "ca",
    "thal",
    "slope",
]

# Columns where zero is physiologically impossible in living adult subjects.
ZERO_AS_MISSING_COLUMNS: Final[tuple[str, ...]] = ("chol", "trestbps")

# Split proportions: 20% test, and 25% of remaining 80% (= 20% total) validation -> 60/20/20.
TEST_SPLIT_RATIO: Final[float] = 0.20
VAL_SPLIT_RATIO_OF_FULL_TRAIN: Final[float] = 0.25

# Stratified CV folds and bootstrap resamples for 95% confidence intervals.
CV_FOLDS: Final[int] = 5
BOOTSTRAP_RESAMPLES: Final[int] = 1000

# Hyperparameter search spaces evaluated during model selection.
LOGISTIC_C_GRID: Final[list[float]] = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
DECISION_TREE_MAX_DEPTH_GRID: Final[list[int | None]] = [2, 3, 4, 5, 6, 8, 10, None]
DECISION_TREE_MIN_LEAF_GRID: Final[list[int]] = [1, 5, 10, 15, 20, 30]
RANDOM_FOREST_ESTIMATORS_GRID: Final[list[int]] = [50, 100, 150, 200]
RANDOM_FOREST_MAX_DEPTH_GRID: Final[list[int]] = [4, 6, 8, 10]
RANDOM_FOREST_MIN_LEAF_GRID: Final[list[int]] = [1, 3, 5]
XGB_LEARNING_RATE_GRID: Final[list[float]] = [0.01, 0.03, 0.05, 0.1, 0.2]
XGB_MAX_DEPTH_GRID: Final[list[int]] = [2, 3, 4]
XGB_MIN_CHILD_WEIGHT_GRID: Final[list[int]] = [1, 3, 5]
