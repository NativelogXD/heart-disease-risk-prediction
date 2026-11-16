# Coronary Heart Disease Risk Prediction

End-to-end Machine Learning Engineering project built for the **DataTalks.Club Machine Learning Zoomcamp (2026 Cohort) - Midterm Project**.

---

## 1. Problem Description and Clinical Context

Coronary artery disease (CAD) is one of the primary causes of cardiovascular morbidity and mortality globally. In clinical triage, cardiologists and emergency physicians evaluate non-invasive demographic, symptomatic, resting electrocardiographic, and exercise-stress parameters before referring patients for invasive coronary angiography.

- **Prediction Objective:** Estimate the calibrated probability $P(Y = 1 \mid \mathbf{x})$ that a patient has significant angiographic coronary artery narrowing (`> 50%` diameter stenosis in at least one major coronary vessel).
- **Target Variable (`heart_disease`):** Binary indicator (`1` = coronary disease present, `0` = absence of significant stenosis).
- **Operational Use Case:** Deployed as a containerized FastAPI microservice (`POST /predict`), the system accepts validated patient clinical records, handles institutional missing lab measurements deterministically, and returns:
  - Continuous probability score (`[0.0, 1.0]`).
  - Binary triage classification evaluated at a tuned sensitivity-focused decision threshold (`0.43`).
  - Clinical risk tier (`low`, `moderate`, `high`).
- **Evaluation Metric Alignment:** Because missed coronary stenosis cases (false negatives) carry severe clinical risk, model selection optimizes ranking discrimination (`ROC-AUC`, `PR-AUC`) and probability calibration (`Brier Score`, `Log-Loss`), followed by validation threshold tuning to maximize `F1-Score` and `Recall`.

---

## 2. Dataset Description and Reproducibility

The dataset aggregates four independent multi-center clinical cohorts from the official **UCI Machine Learning Repository (Heart Disease Database)**:

| Clinical Center | Raw Archive File | Records | Disease Prevalence (Train) |
| :--- | :--- | :---: | :---: |
| Cleveland Clinic Foundation | `processed.cleveland.data` | 303 | 44.85% |
| Hungarian Institute of Cardiology | `processed.hungarian.data` | 294 | 37.21% |
| University Hospital, Zurich & Basel (Switzerland) | `processed.switzerland.data` | 123 | 91.04% |
| V.A. Medical Center, Long Beach | `processed.va.data` | 200 | 78.15% |
| **Combined Cohort** | `data/heart_disease_uci.csv` | **920** | **55.33% overall (509 / 920)** |

This dataset is distinct from all course homework datasets (`car_fuel_efficiency`, `course_lead_scoring`, `telco_churn`, `credit_scoring`, and `clothing_classification`).

### Data Dictionary (14 Predictors + 1 Binary Target)

| Column | Type | Description |
| :--- | :--- | :--- |
| `age` | Continuous | Patient age in years (`28.0` to `77.0`) |
| `sex` | Categorical | Biological sex (`male`, `female`) |
| `cp` | Categorical | Chest pain presentation (`typical_angina`, `atypical_angina`, `non_anginal`, `asymptomatic`) |
| `trestbps` | Continuous | Resting systolic blood pressure on admission in mm Hg |
| `chol` | Continuous | Serum total cholesterol in mg/dl |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl (`gt_120`, `lte_120`) |
| `restecg` | Categorical | Resting ECG result (`normal`, `st_t_abnormality`, `lv_hypertrophy`) |
| `thalach` | Continuous | Maximum heart rate achieved during exercise stress testing (bpm) |
| `exang` | Categorical | Exercise-induced angina (`yes`, `no`) |
| `oldpeak` | Continuous | Exercise-induced ST depression relative to rest (`-2.6` to `6.2` mm) |
| `slope` | Categorical | Slope of peak exercise ST segment (`upsloping`, `flat`, `downsloping`) |
| `ca` | Continuous/Ordinal | Major coronary vessels colored by fluoroscopy (`0.0` to `3.0`) |
| `thal` | Categorical | Thallium perfusion scan status (`normal`, `fixed_defect`, `reversible_defect`) |
| `clinical_center` | Categorical | Contributing hospital cohort (`cleveland`, `hungarian`, `switzerland`, `va_long_beach`) |
| `heart_disease` | Binary Target | Angiographic stenosis > 50% (`1` = positive, `0` = negative) |

### Data Integrity Verification

Running `scripts/download_data.py` fetches all four raw UCI archives directly, normalizes categorical codes, binarizes `num > 0` into `heart_disease` (dropping `num` to prevent target leakage), and persists `data/heart_disease_uci.csv`:

- **Dimensions:** `920` rows x `15` columns
- **SHA-256 Checksum:** `aaa995d8692c4290d9b0e1c5bc5adcba7690ceddbd07d6b6f97ecebae86f0871`

---

## 3. Exploratory Data Analysis and Inferential Statistics

To guarantee zero data leakage, all exploratory diagnostics, hypothesis tests, and preprocessing parameters were computed exclusively on the stratified training partition (`N = 552`, `60%`) after holding out stratified `validation` (`N = 184`, `20%`) and `test` (`N = 184`, `20%`) splits with `SEED = 42`.

### Data Quality and Structural Missingness

1. **Physiologically Impossible Zeros:** Audit of numerical ranges uncovered `172` zero entries in `chol` (`0 mg/dl`, predominantly from the Switzerland cohort where unrecorded cholesterol was encoded as `0`) and `1` zero entry in `trestbps`, alongside explicit `NaN` counts (`ca`: `611` / `66.41%`, `thal`: `486` / `52.83%`, `slope`: `309` / `33.59%`, `fbs`: `90` / `9.78%`, `oldpeak`: `62` / `6.74%`, `trestbps`: `59` / `6.41%`, `thalach`: `55` / `5.98%`, `exang`: `55` / `5.98%`, `chol`: `30` explicit `NaN` / `3.26%`, `restecg`: `2` / `0.22%`).
2. **Leakage-Free Preprocessing (`ClinicalFeaturePreprocessor`):** Values `<= 0` in `chol` and `trestbps` are coerced to `NaN`; binary missingness indicators (`chol_missing`, `ca_missing`, `thal_missing`, `slope_missing`) are created before imputation; numerical medians are fitted strictly on `df_train` (`age: 54.0`, `trestbps: 130.0`, `chol: 238.0`, `thalach: 140.0`, `oldpeak: 0.5`, `ca: 0.0`); and missing categorical levels are preserved as `'missing'`.

### Inferential Hypothesis Testing Summary (`df_train`, `N = 552`, `alpha = 0.05`)

- **Continuous Predictors (Welch's t-test, Mann-Whitney U test, Cohen's d):**
  - `ca`: `d = 1.1574`, `t = 7.6631`, `p < 1e-6`
  - `thalach`: `d = -0.9001`, `t = -10.2271`, `p < 1e-6`
  - `oldpeak`: `d = 0.8230`, `t = 9.6894`, `p < 1e-6`
  - `age`: `d = 0.7222`, `t = 8.3920`, `p < 1e-6`
  - `trestbps`: `d = 0.2583`, `t = 2.9465`, `p_welch = 0.003361`
  - `chol` (non-zero valid readings): `d = 0.2505`, `t = 2.6241`, `p_welch = 0.009002`
- **Categorical Predictors (Chi-square Independence, Cramer's V, Mutual Information, Risk Ratio):**
  - `cp`: `chi2 = 140.55`, `V = 0.5046`, `MI = 0.1346` nats (`asymptomatic` Risk Ratio = `1.4049`)
  - `exang`: `chi2 = 109.95`, `V = 0.4463`, `MI = 0.1060` nats (`yes` Risk Ratio = `1.5037`)
  - `clinical_center`: `chi2 = 91.10`, `V = 0.4062`, `MI = 0.0904` nats (`switzerland` Risk Ratio = `1.6478`)
  - `slope`: `chi2 = 79.93`, `V = 0.3805`, `MI = 0.0751` nats (`flat` Risk Ratio = `1.3989`)
  - `thal`: `chi2 = 72.32`, `V = 0.3620`, `MI = 0.0688` nats (`reversible_defect` Risk Ratio = `1.4606`)
  - `sex`: `chi2 = 60.79`, `V = 0.3319`, `MI = 0.0578` nats (`male` Risk Ratio = `1.1629`)
  - `fbs`: `chi2 = 33.99`, `V = 0.2482`, `MI = 0.0332` nats (`missing` Risk Ratio = `1.5513`)
  - `restecg`: `chi2 = 11.06`, `p = 0.003961`, `V = 0.1416`, `MI = 0.0102` nats (`st_t_abnormality` Risk Ratio = `1.2172`)

Full visualizations, correlation matrices, ROC/PR curves, and detailed analyses are available in [`notebook.ipynb`](./notebook.ipynb).

---

## 4. Architecture and Software Engineering Principles

The codebase adheres to **SOLID** separation of concerns, strict Python type annotations, deterministic random seeding (`SEED = 42`), structured logging, and Pydantic v2 contract enforcement:

```mermaid
flowchart LR
    A[UCI Raw Archives] -->|scripts/download_data.py| B[data/heart_disease_uci.csv]
    B -->|src/data_processor.py: Stratified 60/20/20 Split| C[Train / Val / Test]
    C -->|5-Fold CV + Grid Parallelization| D[train.py]
    D -->|ClinicalFeaturePreprocessor -> DictVectorizer -> StandardScaler -> Estimator| E[models/model_pipeline.bin]
    E -->|Lifespan Load| F[predict.py: FastAPI + Pydantic v2]
    G[Client HTTP JSON] -->|POST /predict| F
    F -->|Probability, Prediction, Risk Tier| G
```

- `src/config.py`: Centralized typed configuration, feature sets, paths, and hyperparameter search spaces.
- `src/data_processor.py`: Leakage-free stratified splitter (`split_dataset`) and custom `sklearn` transformer (`ClinicalFeaturePreprocessor`).
- `src/evaluator.py`: Discrimination/calibration metrics, vectorized F1 threshold optimizer, percentile bootstrap 95% confidence intervals (`B = 1000`), and subgroup error slicer.
- `src/schemas.py`: Strict Pydantic v2 contracts (`ConfigDict(extra="forbid")`, physiological numeric bounds, `Literal` enumerations).
- `train.py`: Automated multi-family tuning, model selection, full-train refit, held-out test reporting, and binary artifact export.
- `predict.py`: FastAPI ASGI service exposing `/health` and `/predict`.

---

## 5. Modeling Results and Final Test Evaluation

Five distinct model families were tuned using 5-fold stratified cross-validation on `df_train` (`N = 552`) and compared on the untouched validation split (`N = 184`):

| Model Family | Best Tuned Hyperparameters | Train 5-Fold CV ROC-AUC | Val ROC-AUC (95% CI) | Val PR-AUC | Val Brier | Val Log-Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `L2`, `C = 0.01`, `solver = liblinear` | **0.9000 +/- 0.0269** | **0.8905** `[0.8430, 0.9360]` | 0.8971 | 0.1334 | 0.4271 |
| Random Forest | `n_estimators = 100`, `max_depth = 6`, `min_samples_leaf = 1` | 0.8922 +/- 0.0298 | 0.8879 `[0.8400, 0.9338]` | 0.9051 | 0.1364 | 0.4295 |
| XGBoost | `n_estimators = 150`, `lr = 0.05`, `max_depth = 2`, `subsample = 0.8` | 0.8922 +/- 0.0295 | 0.8857 `[0.8331, 0.9338]` | 0.9043 | **0.1329** | **0.4235** |
| Decision Tree | `max_depth = 4`, `min_samples_leaf = 15` | 0.8222 +/- 0.0601 | 0.8112 `[0.7451, 0.8751]` | 0.7969 | 0.1730 | 0.5595 |
| Baseline (Dummy Prior) | `strategy = prior` | 0.5000 +/- 0.0000 | 0.5000 `[0.5000, 0.5000]` | 0.5543 | 0.2470 | 0.6872 |

### Final Model Selection and Held-Out Test Performance

Regularized `Logistic Regression (L2, C = 0.01)` achieved the highest validation `ROC-AUC` (`0.8905`) and 5-fold train CV `ROC-AUC` (`0.9000`), while offering direct log-odds interpretability and faster training/inference latency than tree ensembles on `N = 920`. Optimizing the decision threshold on validation improved `F1-Score` from `0.8309` (`t = 0.50`) to `0.8545` at `t* = 0.43`.

Refitted on `df_full_train` (`N = 736`, 5-fold CV `ROC-AUC = 0.9010 +/- 0.0190`) and evaluated once on the held-out `test` set (`N = 184`, `t* = 0.43`):

- **Test ROC-AUC:** `0.9249` (95% Bootstrap CI: `[0.8819, 0.9603]`)
- **Test PR-AUC:** `0.9254` (95% Bootstrap CI: `[0.8614, 0.9714]`)
- **Test F1-Score:** `0.8704`
- **Test Recall (Sensitivity):** `0.9216` (`94` true positives detected; `8` false negatives)
- **Test Precision:** `0.8246`
- **Test Accuracy:** `0.8478`
- **Test Brier Score / Log-Loss:** `0.1116` / `0.3705`
- **Confusion Matrix (`test`):** `TN = 62`, `FP = 20`, `FN = 8`, `TP = 94`

### Subgroup Error Analysis (`test`, `N = 184`)

| Slice | Stratum | N | Positive Rate | ROC-AUC | F1-Score | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sex` | `female` | 35 | 22.86% | 0.9352 | 0.8235 | 0.7778 | 0.8750 |
| `sex` | `male` | 149 | 63.09% | 0.9066 | 0.8744 | 0.8286 | 0.9255 |
| `clinical_center` | `cleveland` | 55 | 47.27% | 0.9430 | 0.8276 | 0.7500 | 0.9231 |
| `clinical_center` | `hungarian` | 64 | 34.38% | 0.9329 | 0.7805 | 0.8421 | 0.7273 |
| `clinical_center` | `va_long_beach` | 37 | 72.97% | 0.7593 | 0.8710 | 0.7714 | 1.0000 |
| `clinical_center` | `switzerland` | 28 | 96.43% | 0.2222 | 0.9818 | 0.9643 | 1.0000 |

Note on `switzerland` test slice: With 27 positives and only 1 negative instance in the test split (`96.43%` prevalence), recall is `1.0000` (`F1 = 0.9818`) while rank-based `ROC-AUC` (`0.2222`) has extreme single-negative variance.

---

## 6. Repository Structure

```text
heart-disease-risk-prediction/
├── .dockerignore
├── .env.example
├── .gitignore
├── .python-version
├── Dockerfile
├── README.md
├── fly.toml
├── render.yaml
├── notebook.ipynb
├── pyproject.toml
├── uv.lock
├── data/
│   └── heart_disease_uci.csv
├── models/
│   ├── metrics_report.json
│   └── model_pipeline.bin
├── scripts/
│   └── download_data.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processor.py
│   ├── evaluator.py
│   └── schemas.py
├── tests/
│   └── test_pipeline.py
├── train.py
├── predict.py
└── test_service.py
```

---

## 7. Local Setup and Execution with `uv`

All dependencies are pinned cryptographically in `uv.lock` for Python `3.13`.

### 1. Clone and Synchronize Virtual Environment

```bash
git clone <your-repository-url>
cd heart-disease-risk-prediction

# Install uv if not already available
pip install uv

# Create .venv and install locked dependencies
uv sync
```

Optional virtual environment activation:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Download Data (Optional, Dataset Already Included)

```bash
uv run python scripts/download_data.py
```

### 3. Train and Evaluate All Models

```bash
uv run python train.py
```

This generates `models/model_pipeline.bin` and `models/metrics_report.json`.

### 4. Run Unit Tests

```bash
uv run pytest -v
```

### 5. Start the FastAPI Service and Verify Endpoints

Terminal 1:

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

Terminal 2:

```bash
uv run python test_service.py
```

Interactive OpenAPI documentation is available at `http://localhost:9696/docs`.

---

## 8. API Usage Examples

### Health Check (`GET /health`)

```bash
curl -s http://localhost:9696/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "logistic_regression"
}
```

### Clinical Risk Prediction (`POST /predict`)

High-risk patient example:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 64,
    "sex": "male",
    "cp": "asymptomatic",
    "trestbps": 152.0,
    "chol": 286.0,
    "fbs": "gt_120",
    "restecg": "lv_hypertrophy",
    "thalach": 108.0,
    "exang": "yes",
    "oldpeak": 2.8,
    "slope": "flat",
    "ca": 2.0,
    "thal": "reversible_defect",
    "clinical_center": "cleveland"
  }'
```

Observed JSON response:

```json
{
  "heart_disease_probability": 0.9699,
  "heart_disease_prediction": true,
  "risk_tier": "high",
  "decision_threshold": 0.43,
  "model_name": "logistic_regression"
}
```

Low-risk patient example (`test_service.py` verified):

```json
{
  "heart_disease_probability": 0.032,
  "heart_disease_prediction": false,
  "risk_tier": "low",
  "decision_threshold": 0.43,
  "model_name": "logistic_regression"
}
```

Sending extra unknown fields or out-of-range physiological values triggers an HTTP `422 Unprocessable Entity` validation error via Pydantic v2.

---

## 9. Containerization with Docker

The `Dockerfile` builds a slim Debian Bookworm Python `3.13` image with `libgomp1`, installs only production runtime dependencies using `uv sync --frozen --no-dev --no-install-project`, and runs Uvicorn under a non-root user (`appuser`).

### Build and Run Locally

```bash
# Build container image
docker build -t heart-disease-service:latest .

# Run container exposing port 9696
docker run --rm -p 9696:9696 heart-disease-service:latest
```

### Verify Running Container

```bash
curl -s http://localhost:9696/health
uv run python test_service.py
```

---

## 10. Cloud Deployment Guide (No Hardcoded Credentials)

Configuration templates are provided for zero-secret deployment; runtime environment variables follow `.env.example` (`MODEL_PATH`, `HOST`, `PORT`, `LOG_LEVEL`) and authentication tokens are managed exclusively through each platform CLI.

### Option A: Fly.io (`fly.toml`)

```bash
# Authenticate interactively via CLI (no credentials stored in repository)
fly auth login

# Initialize app name or deploy using fly.toml
fly launch --no-deploy
fly deploy

# Verify health and prediction endpoint
curl https://<your-fly-app>.fly.dev/health

# Teardown when finished to avoid charges
fly apps destroy <your-fly-app>
```

### Option B: Render (`render.yaml`)

1. Push this repository to GitHub.
2. In the Render dashboard, select **New -> Blueprint** and connect the repository; Render automatically detects `render.yaml` (`runtime: docker`, `healthCheckPath: /health`).
3. Verify `https://<your-render-service>.onrender.com/health` and delete the web service from the Render dashboard when testing concludes.

### Option C: Google Cloud Run

```bash
gcloud run deploy heart-disease-risk-api \
  --source . \
  --port 9696 \
  --region us-central1 \
  --allow-unauthenticated
```

Tear down after verification:

```bash
gcloud run services delete heart-disease-risk-api --region us-central1
```

---

## 11. Known Limitations and Next Steps

1. **Historical Cohort Shift:** The UCI multi-center cohorts were collected across four institutions with differing diagnostic protocols and baseline prevalences (`37%` in Hungary vs. `91%` in Switzerland). Prospective validation on contemporary hospital EHR streams is required before clinical adoption.
2. **Center Identifier Dependency:** Because `clinical_center` encodes institutional referral bias, production deployment in a new clinic requires recalibrating the intercept or training a center-agnostic variant (`clinical_center` excluded) and monitoring calibration drift (`Brier Score` / `ECE`).
3. **Next Steps:**
   - Add probability calibration diagnostics (Platt scaling vs. isotonic calibration) across external cohorts.
   - Integrate SHAP linear/additive explanations into `/predict` responses for physician auditability.
   - Add GitHub Actions CI running `pytest` and `docker build` checks on pull requests.
