# Multi-layer slim production image pinned to Python 3.13 matching .python-version
FROM python:3.13-slim-bookworm

# Install OpenMP runtime required by XGBoost and scikit-learn parallel extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy deterministic uv package manager binaries from official Astral release
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# Activate virtual environment binaries and disable Python stdout buffering
ENV PATH="/code/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="/code/models/model_pipeline.bin"

# Copy lockfiles before application code to leverage Docker build layer caching
COPY pyproject.toml uv.lock .python-version ./

# Synchronize production dependencies deterministically without dev tooling
RUN uv sync --frozen --no-dev --no-install-project

# Copy inference source package, serialized model artifact, and FastAPI entrypoint
COPY src ./src
COPY models/model_pipeline.bin ./models/model_pipeline.bin
COPY predict.py ./predict.py

# Run service as unprivileged system user for container hardening
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /code
USER appuser

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
