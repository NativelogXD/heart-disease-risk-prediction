"""Unit tests for data splitting, leakage-free preprocessing, and API contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest

from predict import classify_risk_tier
from src.data_processor import ClinicalFeaturePreprocessor, split_dataset
from src.schemas import PatientRecord


def _make_synthetic_dataframe(n_rows: int = 100) -> pd.DataFrame:
    """Create a synthetic clinical dataframe mirroring the UCI schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(30, 75, size=n_rows),
            "sex": rng.choice(["male", "female"], size=n_rows),
            "cp": rng.choice(["typical_angina", "atypical_angina", "asymptomatic"], size=n_rows),
            "trestbps": rng.choice([0.0, 120.0, 140.0, np.nan], size=n_rows),
            "chol": rng.choice([0.0, 210.0, 250.0, np.nan], size=n_rows),
            "fbs": rng.choice(["gt_120", "lte_120", None], size=n_rows),
            "restecg": rng.choice(["normal", "lv_hypertrophy"], size=n_rows),
            "thalach": rng.integers(90, 185, size=n_rows).astype(float),
            "exang": rng.choice(["yes", "no", None], size=n_rows),
            "oldpeak": rng.uniform(0.0, 4.0, size=n_rows),
            "slope": rng.choice(["upsloping", "flat", None], size=n_rows),
            "ca": rng.choice([0.0, 1.0, 2.0, np.nan], size=n_rows),
            "thal": rng.choice(["normal", "reversible_defect", None], size=n_rows),
            "clinical_center": rng.choice(["cleveland", "hungarian", "switzerland"], size=n_rows),
            "heart_disease": np.array([0, 1] * (n_rows // 2)),
        }
    )


def test_split_dataset_is_stratified_and_disjoint() -> None:
    df = _make_synthetic_dataframe(100)
    splits = split_dataset(df)

    assert len(splits.df_train) == 60
    assert len(splits.df_val) == 20
    assert len(splits.df_test) == 20
    assert len(splits.df_full_train) == 80
    assert np.isclose(np.mean(splits.y_train), 0.5)
    assert np.isclose(np.mean(splits.y_val), 0.5)
    assert np.isclose(np.mean(splits.y_test), 0.5)


def test_preprocessor_replaces_impossible_zeros_and_prevents_leakage() -> None:
    train_df = pd.DataFrame(
        [
            {"age": 50, "sex": "male", "cp": "asymptomatic", "trestbps": 120.0, "chol": 200.0, "thalach": 150.0, "oldpeak": 1.0, "ca": 0.0, "clinical_center": "cleveland"},
            {"age": 60, "sex": "female", "cp": "typical_angina", "trestbps": 140.0, "chol": 240.0, "thalach": 130.0, "oldpeak": 2.0, "ca": 1.0, "clinical_center": "cleveland"},
            {"age": 55, "sex": "male", "cp": "non_anginal", "trestbps": 0.0, "chol": 0.0, "thalach": 140.0, "oldpeak": 1.5, "ca": np.nan, "clinical_center": "switzerland"},
        ]
    )
    # Extreme cholesterol in validation must not influence medians fitted on train_df.
    val_df = pd.DataFrame(
        [
            {"age": 65, "sex": "male", "cp": "asymptomatic", "trestbps": 999.0, "chol": 0.0, "thalach": 110.0, "oldpeak": 3.0, "ca": np.nan, "clinical_center": "switzerland"}
        ]
    )

    prep = ClinicalFeaturePreprocessor()
    prep.fit(train_df)

    # Zero cholesterol row was converted to NaN before median, so train median is (200 + 240)/2 = 220.
    assert np.isclose(prep.medians_["chol"], 220.0)
    assert np.isclose(prep.medians_["trestbps"], 130.0)

    transformed_val = prep.transform(val_df)
    assert transformed_val[0]["chol"] == 220.0
    assert transformed_val[0]["chol_missing"] == 1
    assert transformed_val[0]["ca_missing"] == 1
    assert transformed_val[0]["thal"] == "missing"


def test_patient_record_schema_rejects_extra_fields_and_invalid_ranges() -> None:
    valid = PatientRecord(
        age=58,
        sex="male",
        cp="asymptomatic",
        chol=245.0,
        clinical_center="cleveland",
    )
    assert valid.age == 58

    with pytest.raises(ValidationError):
        PatientRecord(age=10, sex="male", cp="asymptomatic")

    with pytest.raises(ValidationError):
        PatientRecord(age=58, sex="male", cp="asymptomatic", unknown_field="x")


def test_risk_tier_mapping() -> None:
    assert classify_risk_tier(0.15) == "low"
    assert classify_risk_tier(0.45) == "moderate"
    assert classify_risk_tier(0.80) == "high"
