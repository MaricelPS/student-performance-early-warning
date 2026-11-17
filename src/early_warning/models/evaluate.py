"""Metrics, operating-point selection and grouped bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def classification_metrics(y, p) -> dict:
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def regression_metrics(y, yhat, pass_threshold: float = 5.0) -> dict:
    failed = (np.asarray(y) < pass_threshold).astype(int)
    return {
        "mae": mean_absolute_error(y, yhat),
        "rmse": root_mean_squared_error(y, yhat),
        "r2": r2_score(y, yhat),
        # Ranking quality of the predicted grade as a "risk score" (lower = riskier),
        # comparable with the classifiers' PR-AUC.
        "pr_auc_from_grade": average_precision_score(failed, -np.asarray(yhat)),
    }


def threshold_for_recall(y, p, target_recall: float) -> float:
    """Highest threshold whose recall is still >= target (maximizes precision)."""
    precision, recall, thr = precision_recall_curve(y, p)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def operating_point(y, p, threshold: float) -> dict:
    pred = (np.asarray(p) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "alert_rate": float(pred.mean()),
    }


def grouped_bootstrap_ci(
    metric_fn, y, p, groups, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling *students* (not rows)."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in sample])
        try:
            stats.append(metric_fn(y[idx], p[idx]))
        except ValueError:  # resample with a single class
            continue
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric_fn(y, p)), float(lo), float(hi)
