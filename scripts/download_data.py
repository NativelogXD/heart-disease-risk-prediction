"""Download and normalize the UCI Multi-Center Heart Disease dataset."""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Final
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

# Configure module logger for reproducible CLI execution.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

UCI_BASE_URL: Final[str] = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
)

# Map each raw UCI center file to an explicit clinical_center identifier.
CENTER_FILES: Final[dict[str, str]] = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}

RAW_COLUMNS: Final[list[str]] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Decode numeric UCI category codes into self-explanatory domain labels.
CODE_MAPPINGS: Final[dict[str, dict[float, str]]] = {
    "sex": {1.0: "male", 0.0: "female"},
    "cp": {
        1.0: "typical_angina",
        2.0: "atypical_angina",
        3.0: "non_anginal",
        4.0: "asymptomatic",
    },
    "fbs": {1.0: "gt_120", 0.0: "lte_120"},
    "restecg": {
        0.0: "normal",
        1.0: "st_t_abnormality",
        2.0: "lv_hypertrophy",
    },
    "exang": {1.0: "yes", 0.0: "no"},
    "slope": {1.0: "upsloping", 2.0: "flat", 3.0: "downsloping"},
    "thal": {3.0: "normal", 6.0: "fixed_defect", 7.0: "reversible_defect"},
}

OUTPUT_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "data" / "heart_disease_uci.csv"


def fetch_center_dataframe(center_name: str, filename: str, timeout: int = 30) -> pd.DataFrame:
    """Download one UCI center file and parse raw columns with missing-value markers."""
    url = f"{UCI_BASE_URL}{filename}"
    logger.info("Fetching %s from %s", center_name, url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to download UCI dataset file '{url}': {exc}") from exc

    # Replace '?' markers with NaN at read time using vectorized C parser.
    df_center = pd.read_csv(
        io.StringIO(payload),
        header=None,
        names=RAW_COLUMNS,
        na_values="?",
    )
    df_center["clinical_center"] = center_name
    return df_center


def normalize_clinical_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw numeric codes into canonical clinical labels and binary target."""
    df = df_raw.copy()

    # Coerce all raw columns except center identifier to numeric before mapping.
    for col in RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Map categorical integer codes uniformly to human-readable string labels.
    for col, mapping in CODE_MAPPINGS.items():
        df[col] = df[col].map(mapping)

    # Binarize angiographic stenosis severity: 0 -> no disease, 1..4 -> heart disease.
    df["heart_disease"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    # Standardize column order with predictors followed by binary target.
    ordered_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "clinical_center",
        "heart_disease",
    ]
    return df[ordered_columns]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum for data integrity verification."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_save(output_path: Path = OUTPUT_PATH) -> Path:
    """Download, concatenate, normalize, and persist the multi-center dataset."""
    frames = [
        fetch_center_dataframe(center_name, filename)
        for center_name, filename in CENTER_FILES.items()
    ]
    df_combined = pd.concat(frames, ignore_index=True)
    df_clean = normalize_clinical_dataset(df_combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    checksum = compute_sha256(output_path)
    positive_rate = float(np.mean(df_clean["heart_disease"]))
    logger.info(
        "Saved %d rows x %d columns to %s (positive_rate=%.4f, sha256=%s)",
        df_clean.shape[0],
        df_clean.shape[1],
        output_path,
        positive_rate,
        checksum,
    )
    return output_path


if __name__ == "__main__":
    download_and_save()
