"""Statistical evaluation utilities, threshold optimization, and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import BOOTSTRAP_RESAMPLES, SEED


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute discrimination, calibration, and thresholded classification metrics."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


def compute_confusion_counts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Return TN, FP, FN, TP integer counts for a given probability cutoff."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low: float = 0.10,
    high: float = 0.90,
    step: float = 0.01,
) -> tuple[float, float]:
    """Find the probability cutoff maximizing F1-Score via vectorized grid comparison."""
    thresholds = np.round(np.arange(low, high + step / 2.0, step), 4)
    # Broadcast predicted probabilities against all candidate thresholds in one matrix op.
    pred_matrix = y_prob[:, None] >= thresholds[None, :]
    y_col = y_true[:, None].astype(bool)

    tp = np.sum(pred_matrix & y_col, axis=0)
    fp = np.sum(pred_matrix & ~y_col, axis=0)
    fn = np.sum(~pred_matrix & y_col, axis=0)

    denom = 2 * tp + fp + fn
    f1_scores = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> dict[str, float]:
    """Estimate non-parametric 95% confidence interval via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    scores: list[float] = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, size=n_samples)
        y_boot = y_true[indices]
        # Skip single-class resamples where ROC-AUC or PR-AUC is mathematically undefined.
        if len(np.unique(y_boot)) < 2:
            continue
        scores.append(float(metric_fn(y_boot, y_prob[indices])))

    if not scores:
        raise RuntimeError("Bootstrap failed: all resamples contained a single class.")

    arr = np.asarray(scores, dtype=np.float64)
    return {
        "point_estimate": float(metric_fn(y_true, y_prob)),
        "bootstrap_mean": float(np.mean(arr)),
        "ci_lower_95": float(np.percentile(arr, 2.5)),
        "ci_upper_95": float(np.percentile(arr, 97.5)),
        "std_error": float(np.std(arr, ddof=1)),
    }


def evaluate_by_subgroup(
    df_features: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroup_col: str,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Perform slice-based error analysis across demographic or clinical center strata."""
    if subgroup_col not in df_features.columns:
        raise ValueError(f"Subgroup column '{subgroup_col}' not found in feature DataFrame.")

    series = df_features[subgroup_col].fillna("missing").astype(str)
    results: list[dict[str, Any]] = []

    for group_value in sorted(series.unique()):
        mask = (series == group_value).to_numpy()
        y_sub = y_true[mask]
        p_sub = y_prob[mask]
        y_pred_sub = (p_sub >= threshold).astype(np.int64)

        has_both_classes = len(np.unique(y_sub)) == 2
        roc_val = float(roc_auc_score(y_sub, p_sub)) if has_both_classes else None

        results.append(
            {
                "subgroup": subgroup_col,
                "value": group_value,
                "n_samples": int(np.sum(mask)),
                "prevalence": float(np.mean(y_sub)),
                "roc_auc": roc_val,
                "f1_score": float(f1_score(y_sub, y_pred_sub, zero_division=0)),
                "precision": float(precision_score(y_sub, y_pred_sub, zero_division=0)),
                "recall": float(recall_score(y_sub, y_pred_sub, zero_division=0)),
            }
        )

    return results
