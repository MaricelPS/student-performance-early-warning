import numpy as np
import pandas as pd

from early_warning.models.evaluate import (
    grouped_bootstrap_ci,
    operating_point,
    threshold_for_recall,
)
from early_warning.models.train import grouped_holdout, select_model


def test_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 500), 0, 1)
    thr = threshold_for_recall(y, p, 0.8)
    assert operating_point(y, p, thr)["recall"] >= 0.8


def test_grouped_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    yhat = y + rng.normal(scale=0.5, size=300)
    groups = np.repeat(np.arange(100), 3)
    est, lo, hi = grouped_bootstrap_ci(lambda a, b: np.mean(np.abs(a - b)), y, yhat, groups, 200)
    assert lo <= est <= hi


def test_one_standard_error_rule_prefers_simpler_model():
    cv = pd.DataFrame(
        {
            "model": ["lightgbm_tuned"] * 5 + ["logistic_regression"] * 5,
            "pr_auc": [0.31, 0.29, 0.30, 0.32, 0.28, 0.30, 0.28, 0.29, 0.31, 0.29],
        }
    )
    chosen, summary = select_model(cv, "classification")
    assert chosen == "logistic_regression"
    assert summary.loc["lightgbm_tuned", "best_mean"]


def test_one_standard_error_rule_keeps_clear_winner():
    cv = pd.DataFrame(
        {
            "model": ["random_forest"] * 4 + ["ridge"] * 4,
            "mae": [0.30, 0.31, 0.30, 0.31, 0.50, 0.52, 0.51, 0.50],
        }
    )
    chosen, _ = select_model(cv, "regression")
    assert chosen == "random_forest"


def test_grouped_holdout_has_no_student_overlap(tables):
    from early_warning.features.build import build_training_frame

    frame = build_training_frame(tables, 60)
    train, test = grouped_holdout(frame, 0.2, 42)
    assert set(train["user_uuid"]).isdisjoint(test["user_uuid"])
    assert len(train) + len(test) == len(frame)
