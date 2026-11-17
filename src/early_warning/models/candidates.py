"""Candidate model families and their tuning spaces.

Candidates are listed from simplest to most complex. That order is used by the
one-standard-error rule in model selection: when two models are statistically
tied, the simpler one wins.
"""

from __future__ import annotations

import warnings

from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from early_warning.models.transformers import DropAllMissing

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _imputed(est, scale: bool = False) -> Pipeline:
    steps = [SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)]
    if scale:
        steps.append(StandardScaler())
    steps.append(est)
    return make_pipeline(*steps)


def classification_candidates(seed: int) -> dict:
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": _imputed(
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000), scale=True
        ),
        "decision_tree": _imputed(
            DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=seed
            )
        ),
        "random_forest": _imputed(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
        ),
        "hist_gradient_boosting": make_pipeline(
            DropAllMissing(),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "lightgbm_tuned": LGBMClassifier(random_state=seed, verbose=-1, n_jobs=1),
    }


def regression_candidates(seed: int) -> dict:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": _imputed(Ridge(alpha=1.0), scale=True),
        "decision_tree": _imputed(
            DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=seed)
        ),
        "random_forest": _imputed(
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed
            )
        ),
        "hist_gradient_boosting": make_pipeline(
            DropAllMissing(),
            HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed
            ),
        ),
        "lightgbm_tuned": LGBMRegressor(random_state=seed, verbose=-1, n_jobs=1),
    }


LGBM_SPACE = {
    "n_estimators": randint(100, 600),
    "learning_rate": loguniform(0.01, 0.2),
    "num_leaves": randint(4, 32),
    "min_child_samples": randint(10, 80),
    "subsample": uniform(0.6, 0.4),
    "subsample_freq": [1],
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": loguniform(1e-3, 10),
}
LGBM_CLASSIFIER_EXTRA = {"class_weight": [None, "balanced"]}

COMPLEXITY_ORDER = [
    "dummy_prior",
    "dummy_mean",
    "logistic_regression",
    "ridge",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
    "lightgbm_tuned",
]
