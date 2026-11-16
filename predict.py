"""FastAPI inference microservice serving the trained coronary heart disease pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import pickle
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
import uvicorn

from src.config import MODEL_ARTIFACT_PATH
from src.schemas import HealthResponse, PatientRecord, PredictionResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level state populated during FastAPI lifespan startup and cleared on shutdown.
model_bundle: dict[str, Any] | None = None


def resolve_artifact_path() -> Path:
    """Resolve model artifact path from MODEL_PATH environment variable or default location."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
        rel_candidate = Path(__file__).resolve().parent / env_path
        if rel_candidate.exists():
            return rel_candidate
        raise FileNotFoundError(f"Configured MODEL_PATH does not exist: {env_path}")

    if MODEL_ARTIFACT_PATH.exists():
        return MODEL_ARTIFACT_PATH

    raise FileNotFoundError(f"Default model artifact not found at: {MODEL_ARTIFACT_PATH}")


def classify_risk_tier(probability: float) -> Literal["low", "moderate", "high"]:
    """Map continuous disease probability into clinical triage risk categories."""
    if probability < 0.30:
        return "low"
    if probability < 0.65:
        return "moderate"
    return "high"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load serialized pipeline bundle once at startup and release on shutdown."""
    global model_bundle
    artifact_path = resolve_artifact_path()
    logger.info("Loading model artifact from %s", artifact_path)

    with artifact_path.open("rb") as handle:
        loaded = pickle.load(handle)

    if not isinstance(loaded, dict) or "pipeline" not in loaded:
        raise RuntimeError("Invalid model artifact format: expected dict containing 'pipeline'.")

    model_bundle = loaded
    logger.info(
        "Model '%s' loaded successfully (decision_threshold=%.2f)",
        model_bundle.get("model_name", "unknown"),
        float(model_bundle.get("decision_threshold", 0.5)),
    )
    yield
    model_bundle = None
    logger.info("Released model bundle on service shutdown.")


app = FastAPI(
    title="Coronary Heart Disease Risk Prediction API",
    description="Clinical inference microservice estimating angiographic coronary stenosis probability.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness and readiness probe verifying whether the model pipeline is loaded."""
    if model_bundle is None:
        return HealthResponse(status="degraded", model_loaded=False, model_name=None)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=str(model_bundle.get("model_name", "unknown")),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientRecord) -> PredictionResponse:
    """Score an incoming validated patient profile and return calibrated risk predictions."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    pipeline = model_bundle["pipeline"]
    threshold = float(model_bundle.get("decision_threshold", 0.5))
    model_name = str(model_bundle.get("model_name", "unknown"))

    try:
        record = patient.model_dump()
        probability = float(pipeline.predict_proba(record)[0, 1])
    except (ValueError, TypeError, KeyError) as exc:
        logger.exception("Input feature transformation failed during prediction.")
        raise HTTPException(status_code=400, detail=f"Invalid feature payload: {exc}") from exc
    except RuntimeError as exc:
        logger.exception("Pipeline inference failed.")
        raise HTTPException(status_code=500, detail=f"Inference failure: {exc}") from exc

    return PredictionResponse(
        heart_disease_probability=round(probability, 4),
        heart_disease_prediction=probability >= threshold,
        risk_tier=classify_risk_tier(probability),
        decision_threshold=round(threshold, 4),
        model_name=model_name,
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9696"))
    uvicorn.run("predict:app", host=host, port=port, reload=False)
