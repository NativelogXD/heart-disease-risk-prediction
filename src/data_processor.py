"""Data splitting and leakage-free clinical preprocessing transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    MISSING_INDICATOR_COLUMNS,
    NUMERICAL_FEATURES,
    SEED,
    TARGET_COLUMN,
    TEST_SPLIT_RATIO,
    VAL_SPLIT_RATIO_OF_FULL_TRAIN,
    ZERO_AS_MISSING_COLUMNS,
)


@dataclass(frozen=True)
class DatasetSplits:
    """Immutable container holding stratified train, validation, full-train, and test splits."""

    df_full_train: pd.DataFrame
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    y_full_train: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def split_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    seed: int = SEED,
) -> DatasetSplits:
    """Create deterministic stratified 60/20/20 splits before any preprocessing fit."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Hold out 20% stratified test partition first so it never informs tuning or imputation.
    df_full_train, df_test = train_test_split(
        df,
        test_size=TEST_SPLIT_RATIO,
        random_state=seed,
        stratify=df[target_column],
    )
    # Split remaining 80% into 60% train and 20% validation (0.25 * 0.80 = 0.20).
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=VAL_SPLIT_RATIO_OF_FULL_TRAIN,
        random_state=seed,
        stratify=df_full_train[target_column],
    )

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train[target_column].to_numpy(dtype=np.int64)
    y_train = df_train[target_column].to_numpy(dtype=np.int64)
    y_val = df_val[target_column].to_numpy(dtype=np.int64)
    y_test = df_test[target_column].to_numpy(dtype=np.int64)

    return DatasetSplits(
        df_full_train=df_full_train.drop(columns=[target_column]),
        df_train=df_train.drop(columns=[target_column]),
        df_val=df_val.drop(columns=[target_column]),
        df_test=df_test.drop(columns=[target_column]),
        y_full_train=y_full_train,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


class ClinicalFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-free preprocessor converting clinical tables or dicts into clean record dicts.

    Fits numerical medians exclusively on the training partition, replaces physiologically
    impossible zeros in cholesterol and resting blood pressure with NaN prior to imputation,
    preserves missingness indicator flags, and encodes missing categorical values as 'missing'.
    """

    def __init__(
        self,
        numerical_features: Sequence[str] = tuple(NUMERICAL_FEATURES),
        categorical_features: Sequence[str] = tuple(CATEGORICAL_FEATURES),
        missing_indicator_features: Sequence[str] = tuple(MISSING_INDICATOR_COLUMNS),
        zero_as_missing_features: Sequence[str] = ZERO_AS_MISSING_COLUMNS,
    ) -> None:
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.missing_indicator_features = missing_indicator_features
        self.zero_as_missing_features = zero_as_missing_features

    def _coerce_to_dataframe(
        self,
        X: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> pd.DataFrame:
        """Normalize supported input structures into a column-aligned DataFrame."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, Mapping):
            df = pd.DataFrame([dict(X)])
        elif isinstance(X, Sequence):
            df = pd.DataFrame([dict(row) for row in X])
        else:
            raise TypeError(
                f"Unsupported input type '{type(X).__name__}' for ClinicalFeaturePreprocessor."
            )

        expected_cols = list(self.numerical_features) + list(self.categorical_features)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df[expected_cols].copy()

        # Vectorized numeric coercion and replacement of impossible <= 0 physiological readings.
        for col in self.numerical_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in self.zero_as_missing_features:
                df.loc[df[col] <= 0, col] = np.nan

        return df

    def fit(
        self,
        X: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        y: Any = None,
    ) -> ClinicalFeaturePreprocessor:
        """Compute training medians and categorical fallbacks without touching validation/test."""
        df = self._coerce_to_dataframe(X)

        self.medians_: dict[str, float] = {}
        for col in self.numerical_features:
            median_val = float(df[col].median())
            # Guard against an all-NaN column during degenerate synthetic unit tests.
            self.medians_[col] = 0.0 if np.isnan(median_val) else median_val

        self.categorical_fallback_ = "missing"
        self.feature_names_in_ = np.array(ALL_FEATURE_COLUMNS, dtype=object)
        return self

    def transform(
        self,
        X: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply learned imputations, add missingness flags, and emit DictVectorizer records."""
        if not hasattr(self, "medians_"):
            raise RuntimeError("ClinicalFeaturePreprocessor must be fitted before transform().")

        df = self._coerce_to_dataframe(X)

        # Create explicit binary missingness indicators before imputing missing values.
        for col in self.missing_indicator_features:
            df[f"{col}_missing"] = df[col].isna().astype(np.int64)

        # Impute numerical variables using medians learned solely during fit().
        for col in self.numerical_features:
            df[col] = df[col].fillna(self.medians_[col]).astype(np.float64)

        # Preserve missing categorical states as a dedicated 'missing' level rather than dropping rows.
        for col in self.categorical_features:
            col_series = df[col].astype(object)
            df[col] = (
                col_series.where(col_series.notna(), self.categorical_fallback_)
                .astype(str)
                .str.strip()
                .str.lower()
            )

        return df.to_dict(orient="records")
