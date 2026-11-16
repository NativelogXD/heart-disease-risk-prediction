"""Pydantic v2 contracts enforcing strict input validation and typed API responses."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    """Input clinical profile for coronary heart disease risk estimation."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=18, le=110, description="Patient age in years.")
    sex: Literal["male", "female"] = Field(..., description="Biological sex.")
    cp: Literal[
        "typical_angina",
        "atypical_angina",
        "non_anginal",
        "asymptomatic",
    ] = Field(..., description="Chest pain presentation category.")
    trestbps: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=260.0,
        description="Resting blood pressure on hospital admission (mm Hg).",
    )
    chol: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=700.0,
        description="Serum total cholesterol (mg/dl).",
    )
    fbs: Optional[Literal["gt_120", "lte_120"]] = Field(
        default=None,
        description="Fasting blood sugar indicator (> 120 mg/dl vs <= 120 mg/dl).",
    )
    restecg: Optional[Literal["normal", "st_t_abnormality", "lv_hypertrophy"]] = Field(
        default=None,
        description="Resting electrocardiographic result.",
    )
    thalach: Optional[float] = Field(
        default=None,
        ge=40.0,
        le=230.0,
        description="Maximum heart rate achieved during exercise testing (bpm).",
    )
    exang: Optional[Literal["yes", "no"]] = Field(
        default=None,
        description="Exercise-induced angina indicator.",
    )
    oldpeak: Optional[float] = Field(
        default=None,
        ge=-3.0,
        le=10.0,
        description="ST depression induced by exercise relative to rest (mm).",
    )
    slope: Optional[Literal["upsloping", "flat", "downsloping"]] = Field(
        default=None,
        description="Slope of the peak exercise ST segment.",
    )
    ca: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=3.0,
        description="Number of major vessels colored by fluoroscopy (0-3).",
    )
    thal: Optional[Literal["normal", "fixed_defect", "reversible_defect"]] = Field(
        default=None,
        description="Thallium stress test perfusion defect classification.",
    )
    clinical_center: Literal[
        "cleveland",
        "hungarian",
        "switzerland",
        "va_long_beach",
    ] = Field(
        default="cleveland",
        description="Clinical cohort origin identifier.",
    )


class PredictionResponse(BaseModel):
    """Typed prediction payload returned by POST /predict."""

    heart_disease_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability of angiographic coronary disease (>50% narrowing).",
    )
    heart_disease_prediction: bool = Field(
        ...,
        description="Binary classification decision relative to the tuned threshold.",
    )
    risk_tier: Literal["low", "moderate", "high"] = Field(
        ...,
        description="Clinical triage tier derived from predicted risk probability.",
    )
    decision_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Decision cutoff applied to convert probability into a binary label.",
    )
    model_name: str = Field(..., description="Identifier of the serialized winning model.")


class HealthResponse(BaseModel):
    """Typed service health status returned by GET /health."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: Optional[str] = None
