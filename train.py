"""Reproducible training pipeline for coronary heart disease risk prediction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
import sys
from typing import Any, Callable

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from scripts.download_data import download_and_save
from src.config import (
    CV_FOLDS,
    DATA_PATH,
    DECISION_TREE_MAX_DEPTH_GRID,
    DECISION_TREE_MIN_LEAF_GRID,
    LOGISTIC_C_GRID,
    METRICS_REPORT_PATH,
    MODEL_ARTIFACT_PATH,
    RANDOM_FOREST_ESTIMATORS_GRID,
    RANDOM_FOREST_MAX_DEPTH_GRID,
    RANDOM_FOREST_MIN_LEAF_GRID,
    SEED,
    TARGET_COLUMN,
    XGB_LEARNING_RATE_GRID,
    XGB_MAX_DEPTH_GRID,
    XGB_MIN_CHILD_WEIGHT_GRID,
)
from src.data_processor import ClinicalFeaturePreprocessor, DatasetSplits, split_dataset
from src.evaluator import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    compute_confusion_counts,
    evaluate_by_subgroup,
    find_optimal_threshold,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateResult:
    """Best hyperparameter configuration and validation metrics for one model family."""

    name: str
    params: dict[str, Any]
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    val_metrics: dict[str, float]
    estimator_factory: Callable[[], BaseEstimator]


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Assemble preprocessing, vectorization, scaling, and estimator stages."""
    return Pipeline(
        steps=[
            ("preprocessor", ClinicalFeaturePreprocessor()),
            ("vectorizer", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def load_training_dataframe(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the normalized clinical dataset or download it if absent."""
    if not data_path.exists():
        logger.info("Dataset not found at %s; downloading from UCI repository.", data_path)
        download_and_save(data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded dataset: %d rows x %d columns.", df.shape[0], df.shape[1])
    return df


def precompute_cv_folds(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = CV_FOLDS,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Precompute leakage-free transformed feature matrices once per stratified fold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for train_idx, holdout_idx in cv.split(X, y):
        prep = Pipeline(
            steps=[
                ("preprocessor", ClinicalFeaturePreprocessor()),
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
            ]
        )
        X_tr = prep.fit_transform(X.iloc[train_idx])
        X_ho = prep.transform(X.iloc[holdout_idx])
        folds.append((X_tr, y[train_idx], X_ho, y[holdout_idx]))

    return folds


def _score_single_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ho: np.ndarray,
    y_ho: np.ndarray,
    estimator: BaseEstimator,
) -> float:
    """Fit one cloned estimator on a training fold and compute holdout ROC-AUC."""
    estimator.fit(X_tr, y_tr)
    probs = estimator.predict_proba(X_ho)[:, 1]
    return float(roc_auc_score(y_ho, probs))


def evaluate_grid_parallel(
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    param_configs: list[dict[str, Any]],
    estimator_builder: Callable[[dict[str, Any]], BaseEstimator],
) -> list[tuple[dict[str, Any], float, float]]:
    """Evaluate all (config, fold) combinations in parallel and rank by mean CV ROC-AUC."""
    tasks = [
        (cfg_idx, X_tr, y_tr, X_ho, y_ho, estimator_builder(cfg))
        for cfg_idx, cfg in enumerate(param_configs)
        for (X_tr, y_tr, X_ho, y_ho) in folds
    ]
    raw_scores = Parallel(n_jobs=-1)(
        delayed(_score_single_fold)(X_tr, y_tr, X_ho, y_ho, est)
        for _, X_tr, y_tr, X_ho, y_ho, est in tasks
    )

    n_folds = len(folds)
    grouped_results: list[tuple[dict[str, Any], float, float]] = []
    for cfg_idx, cfg in enumerate(param_configs):
        fold_aucs = raw_scores[cfg_idx * n_folds : (cfg_idx + 1) * n_folds]
        grouped_results.append((cfg, float(np.mean(fold_aucs)), float(np.std(fold_aucs))))

    grouped_results.sort(key=lambda item: item[1], reverse=True)
    return grouped_results


def cross_validate_roc_auc(
    X: pd.DataFrame,
    y: np.ndarray,
    estimator_factory: Callable[[], BaseEstimator],
    n_splits: int = CV_FOLDS,
    seed: int = SEED,
) -> tuple[float, float]:
    """Evaluate stratified K-fold ROC-AUC while fitting preprocessors strictly per fold."""
    folds = precompute_cv_folds(X, y, n_splits=n_splits, seed=seed)
    fold_scores = [_score_single_fold(X_tr, y_tr, X_ho, y_ho, estimator_factory()) for X_tr, y_tr, X_ho, y_ho in folds]
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def tune_candidate_families(splits: DatasetSplits) -> tuple[list[CandidateResult], dict[str, np.ndarray]]:
    """Tune all candidate model families via CV on train and score on held-out validation."""
    candidates: list[CandidateResult] = []
    val_probabilities: dict[str, np.ndarray] = {}
    train_folds = precompute_cv_folds(splits.df_train, splits.y_train)

    def register_candidate(
        name: str,
        params: dict[str, Any],
        cv_mean: float,
        cv_std: float,
        estimator_factory: Callable[[], BaseEstimator],
    ) -> None:
        pipe = build_pipeline(estimator_factory())
        pipe.fit(splits.df_train, splits.y_train)
        p_val = pipe.predict_proba(splits.df_val)[:, 1]
        val_metrics = compute_classification_metrics(splits.y_val, p_val, threshold=0.5)
        val_probabilities[name] = p_val
        candidates.append(
            CandidateResult(
                name=name,
                params=params,
                cv_roc_auc_mean=cv_mean,
                cv_roc_auc_std=cv_std,
                val_metrics=val_metrics,
                estimator_factory=estimator_factory,
            )
        )
        logger.info(
            "Candidate %-24s | CV AUC=%.4f +/- %.4f | Val AUC=%.4f | Val Brier=%.4f",
            name,
            cv_mean,
            cv_std,
            val_metrics["roc_auc"],
            val_metrics["brier_score"],
        )

    # 0. Naive Majority-Prior Baseline
    register_candidate(
        name="dummy_prior_baseline",
        params={"strategy": "prior"},
        cv_mean=0.5,
        cv_std=0.0,
        estimator_factory=lambda: DummyClassifier(strategy="prior"),
    )

    # 1. Regularized Logistic Regression
    lr_configs = [
        {"penalty": reg_name, "l1_ratio": l1_ratio, "C": c_val}
        for l1_ratio, reg_name in [(0.0, "l2"), (1.0, "l1")]
        for c_val in LOGISTIC_C_GRID
    ]
    lr_builder = lambda p: LogisticRegression(
        solver="liblinear", l1_ratio=float(p["l1_ratio"]), C=float(p["C"]), max_iter=2000, random_state=SEED
    )
    lr_params, lr_cv_mean, lr_cv_std = evaluate_grid_parallel(train_folds, lr_configs, lr_builder)[0]
    register_candidate(
        name="logistic_regression",
        params=lr_params,
        cv_mean=lr_cv_mean,
        cv_std=lr_cv_std,
        estimator_factory=lambda p=lr_params: lr_builder(p),
    )

    # 2. Decision Tree Classifier
    dt_configs = [
        {"max_depth": max_depth, "min_samples_leaf": min_leaf}
        for max_depth in DECISION_TREE_MAX_DEPTH_GRID
        for min_leaf in DECISION_TREE_MIN_LEAF_GRID
    ]
    dt_builder = lambda p: DecisionTreeClassifier(
        max_depth=p["max_depth"], min_samples_leaf=int(p["min_samples_leaf"]), random_state=SEED
    )
    dt_params, dt_cv_mean, dt_cv_std = evaluate_grid_parallel(train_folds, dt_configs, dt_builder)[0]
    register_candidate(
        name="decision_tree",
        params=dt_params,
        cv_mean=dt_cv_mean,
        cv_std=dt_cv_std,
        estimator_factory=lambda p=dt_params: dt_builder(p),
    )

    # 3. Random Forest Classifier (parallelized across config-fold grid)
    rf_configs = [
        {"n_estimators": n_est, "max_depth": max_depth, "min_samples_leaf": min_leaf}
        for n_est in RANDOM_FOREST_ESTIMATORS_GRID
        for max_depth in RANDOM_FOREST_MAX_DEPTH_GRID
        for min_leaf in RANDOM_FOREST_MIN_LEAF_GRID
    ]
    rf_builder = lambda p: RandomForestClassifier(
        n_estimators=int(p["n_estimators"]),
        max_depth=int(p["max_depth"]),
        min_samples_leaf=int(p["min_samples_leaf"]),
        n_jobs=1,
        random_state=SEED,
    )
    rf_params, rf_cv_mean, rf_cv_std = evaluate_grid_parallel(train_folds, rf_configs, rf_builder)[0]
    register_candidate(
        name="random_forest",
        params=rf_params,
        cv_mean=rf_cv_mean,
        cv_std=rf_cv_std,
        estimator_factory=lambda p=rf_params: rf_builder(p),
    )

    # 4. XGBoost Gradient Boosted Trees (parallelized across config-fold grid)
    xgb_configs = [
        {
            "n_estimators": 150,
            "learning_rate": eta,
            "max_depth": max_depth,
            "min_child_weight": min_child,
            "subsample": subsample,
            "colsample_bytree": 0.8,
        }
        for eta in XGB_LEARNING_RATE_GRID
        for max_depth in XGB_MAX_DEPTH_GRID
        for min_child in XGB_MIN_CHILD_WEIGHT_GRID
        for subsample in [0.8, 1.0]
    ]
    xgb_builder = lambda p: XGBClassifier(
        n_estimators=int(p["n_estimators"]),
        learning_rate=float(p["learning_rate"]),
        max_depth=int(p["max_depth"]),
        min_child_weight=int(p["min_child_weight"]),
        subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]),
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
    )
    xgb_params, xgb_cv_mean, xgb_cv_std = evaluate_grid_parallel(train_folds, xgb_configs, xgb_builder)[0]
    register_candidate(
        name="xgboost",
        params=xgb_params,
        cv_mean=xgb_cv_mean,
        cv_std=xgb_cv_std,
        estimator_factory=lambda p=xgb_params: xgb_builder(p),
    )

    return candidates, val_probabilities


def train_and_export(
    data_path: Path = DATA_PATH,
    artifact_path: Path = MODEL_ARTIFACT_PATH,
    report_path: Path = METRICS_REPORT_PATH,
) -> dict[str, Any]:
    """Execute full model selection, refit winning pipeline on full_train, and save artifacts."""
    df = load_training_dataframe(data_path)
    splits = split_dataset(df, target_column=TARGET_COLUMN, seed=SEED)

    candidates, val_probs = tune_candidate_families(splits)

    # Select winning model family by validation ROC-AUC (with Brier score as tiebreaker).
    winner = max(
        candidates,
        key=lambda c: (c.val_metrics["roc_auc"], -c.val_metrics["brier_score"]),
    )
    optimal_threshold, best_val_f1 = find_optimal_threshold(
        splits.y_val,
        val_probs[winner.name],
    )
    logger.info(
        "Selected winner: %s (Val ROC-AUC=%.4f, optimal_threshold=%.2f, Val F1=%.4f)",
        winner.name,
        winner.val_metrics["roc_auc"],
        optimal_threshold,
        best_val_f1,
    )

    # Verify full-train 5-fold cross-validation stability of the selected estimator.
    full_cv_mean, full_cv_std = cross_validate_roc_auc(
        splits.df_full_train,
        splits.y_full_train,
        winner.estimator_factory,
    )
    logger.info(
        "Full-train 5-fold CV ROC-AUC for %s: %.4f +/- %.4f",
        winner.name,
        full_cv_mean,
        full_cv_std,
    )

    # Refit final production pipeline on full_train (80%) and evaluate once on test (20%).
    final_pipeline = build_pipeline(winner.estimator_factory())
    final_pipeline.fit(splits.df_full_train, splits.y_full_train)

    p_test = final_pipeline.predict_proba(splits.df_test)[:, 1]
    test_metrics = compute_classification_metrics(
        splits.y_test,
        p_test,
        threshold=optimal_threshold,
    )
    test_confusion = compute_confusion_counts(
        splits.y_test,
        p_test,
        threshold=optimal_threshold,
    )
    roc_ci = bootstrap_metric_ci(splits.y_test, p_test, roc_auc_score)
    pr_ci = bootstrap_metric_ci(splits.y_test, p_test, average_precision_score)

    subgroup_analysis = {
        "sex": evaluate_by_subgroup(
            splits.df_test, splits.y_test, p_test, "sex", threshold=optimal_threshold
        ),
        "clinical_center": evaluate_by_subgroup(
            splits.df_test,
            splits.y_test,
            p_test,
            "clinical_center",
            threshold=optimal_threshold,
        ),
    }

    vectorizer: DictVectorizer = final_pipeline.named_steps["vectorizer"]
    feature_names = [str(f) for f in vectorizer.get_feature_names_out()]

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": final_pipeline,
        "decision_threshold": optimal_threshold,
        "model_name": winner.name,
        "params": winner.params,
        "feature_names": feature_names,
    }
    with artifact_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    report = {
        "seed": SEED,
        "dataset_rows": int(len(df)),
        "split_sizes": {
            "train": int(len(splits.df_train)),
            "validation": int(len(splits.df_val)),
            "full_train": int(len(splits.df_full_train)),
            "test": int(len(splits.df_test)),
        },
        "validation_candidates": [
            {
                "name": c.name,
                "params": c.params,
                "train_cv_roc_auc_mean": round(c.cv_roc_auc_mean, 4),
                "train_cv_roc_auc_std": round(c.cv_roc_auc_std, 4),
                "val_metrics": {k: round(v, 4) for k, v in c.val_metrics.items()},
            }
            for c in sorted(candidates, key=lambda item: item.val_metrics["roc_auc"], reverse=True)
        ],
        "selected_model": {
            "name": winner.name,
            "params": winner.params,
            "decision_threshold": round(optimal_threshold, 4),
            "full_train_cv_roc_auc_mean": round(full_cv_mean, 4),
            "full_train_cv_roc_auc_std": round(full_cv_std, 4),
        },
        "test_evaluation": {
            "metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "confusion_matrix": test_confusion,
            "roc_auc_bootstrap_95ci": {k: round(v, 4) for k, v in roc_ci.items()},
            "pr_auc_bootstrap_95ci": {k: round(v, 4) for k, v in pr_ci.items()},
            "subgroup_analysis": subgroup_analysis,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info("Saved serialized model bundle to %s", artifact_path)
    logger.info("Saved evaluation report to %s", report_path)
    logger.info(
        "Held-out Test ROC-AUC=%.4f (95%% CI [%.4f, %.4f]), PR-AUC=%.4f, F1=%.4f, Recall=%.4f",
        test_metrics["roc_auc"],
        roc_ci["ci_lower_95"],
        roc_ci["ci_upper_95"],
        test_metrics["pr_auc"],
        test_metrics["f1_score"],
        test_metrics["recall"],
    )
    return report


if __name__ == "__main__":
    try:
        train_and_export()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("Training pipeline aborted: %s", exc)
        sys.exit(1)
