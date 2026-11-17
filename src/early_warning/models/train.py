"""Model comparison, selection, calibration and final evaluation.

For every cutoff (course day) and task:

1. Grouped hold-out split by student (``user_uuid``) — a student never appears
   in both train and test, because students take ~5 courses and their
   behaviour is correlated across courses.
2. LightGBM is tuned with randomized search on grouped CV folds.
3. All candidates are compared with *repeated* stratified-grouped K-fold CV.
4. Selection uses the one-standard-error rule (simplest model within 1 SE of
   the best mean score).
5. Classification: the chosen model is probability-calibrated (sigmoid) and an
   alert threshold is chosen on out-of-fold predictions to reach the target
   recall.
6. The untouched test set is scored once; CIs come from a grouped bootstrap.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold

from early_warning.config import Config
from early_warning.data.clean import CleanTables
from early_warning.features.build import FEATURES, build_training_frame
from early_warning.models import candidates as C
from early_warning.models.evaluate import (
    classification_metrics,
    grouped_bootstrap_ci,
    operating_point,
    regression_metrics,
    threshold_for_recall,
)
from early_warning.models.registry import save_model

log = logging.getLogger(__name__)

TASKS = {
    "classification": {"target": "failed", "primary": "pr_auc", "greater_is_better": True},
    "regression": {"target": "final_grade", "primary": "mae", "greater_is_better": False},
}


def grouped_holdout(frame: pd.DataFrame, test_size: float, seed: int):
    n_splits = round(1 / test_size)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(frame, frame["failed"], frame["user_uuid"]))
    return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)


def cv_splits(frame: pd.DataFrame, folds: int, repeats: int, seed: int) -> list:
    """Repeated stratified-grouped folds (stratified on failure for both tasks)."""
    splits = []
    for r in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        splits += list(sgkf.split(frame, frame["failed"], frame["user_uuid"]))
    return splits


def _predict(model, X, task: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if task == "classification" else model.predict(X)


def _score(task: str, y, pred, pass_threshold: float) -> dict:
    if task == "classification":
        return classification_metrics(y, pred)
    return regression_metrics(y, pred, pass_threshold)


def tune_lightgbm(task: str, X, y, splits, n_iter: int, seed: int):
    est = (
        C.classification_candidates(seed)["lightgbm_tuned"]
        if task == "classification"
        else C.regression_candidates(seed)["lightgbm_tuned"]
    )
    space = dict(C.LGBM_SPACE)
    if task == "classification":
        space |= C.LGBM_CLASSIFIER_EXTRA
    search = RandomizedSearchCV(
        est,
        space,
        n_iter=n_iter,
        cv=splits,
        random_state=seed,
        n_jobs=-1,
        scoring="average_precision" if task == "classification" else "neg_mean_absolute_error",
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def compare_models(task: str, train: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    t = cfg.training
    seed = t["random_state"]
    spec = TASKS[task]
    X, y = train[FEATURES], train[spec["target"]]
    splits = cv_splits(train, t["cv_folds"], t["cv_repeats"], seed)

    models = (
        C.classification_candidates(seed)
        if task == "classification"
        else C.regression_candidates(seed)
    )
    tuning_splits = splits[: t["cv_folds"]]
    models["lightgbm_tuned"], best_params = tune_lightgbm(
        task, X, y, tuning_splits, t["tuning_iterations"], seed
    )

    rows = []
    for name, model in models.items():
        for k, (tr, va) in enumerate(splits):
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            scores = _score(
                task, y.iloc[va], _predict(m, X.iloc[va], task), cfg.data["pass_threshold"]
            )
            rows.append({"model": name, "fold": k, **scores})
    cv = pd.DataFrame(rows)
    return cv, {"models": models, "lightgbm_params": best_params}


def select_model(cv: pd.DataFrame, task: str) -> tuple[str, pd.DataFrame]:
    """One-standard-error rule over the primary metric."""
    spec = TASKS[task]
    metric, greater = spec["primary"], spec["greater_is_better"]
    summ = cv.groupby("model")[metric].agg(["mean", "std", "count"])
    summ["se"] = summ["std"] / np.sqrt(summ["count"])
    best = summ["mean"].idxmax() if greater else summ["mean"].idxmin()
    bound = (
        summ.loc[best, "mean"] - summ.loc[best, "se"]
        if greater
        else summ.loc[best, "mean"] + summ.loc[best, "se"]
    )
    within = summ.index[summ["mean"] >= bound] if greater else summ.index[summ["mean"] <= bound]
    chosen = min(within, key=C.COMPLEXITY_ORDER.index)
    summ["selected"] = summ.index == chosen
    summ["best_mean"] = summ.index == best
    return chosen, summ.sort_values("mean", ascending=not greater)


def oof_predictions(model, X, y, splits) -> np.ndarray:
    """Out-of-fold predictions on the training set (first repeat only)."""
    oof = np.full(len(X), np.nan)
    for tr, va in splits:
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def train_one(tables: CleanTables, cutoff: int, task: str, cfg: Config) -> dict:
    t, f = cfg.training, cfg.features
    seed = t["random_state"]
    spec = TASKS[task]
    frame = build_training_frame(
        tables,
        cutoff,
        recent_days=f["recent_window_days"],
        low_score_threshold=f["low_score_threshold"],
        pass_threshold=cfg.data["pass_threshold"],
    )
    train, test = grouped_holdout(frame, t["test_size"], seed)
    log.info("cutoff=%s task=%s train=%s test=%s", cutoff, task, len(train), len(test))

    cv, info = compare_models(task, train, cfg)
    chosen, summary = select_model(cv, task)
    base = info["models"][chosen]
    X_tr, y_tr = train[FEATURES], train[spec["target"]]
    X_te, y_te = test[FEATURES], test[spec["target"]]
    splits = cv_splits(train, t["cv_folds"], 1, seed)

    result: dict = {
        "cutoff": cutoff,
        "task": task,
        "selected_model": chosen,
        "n_train": len(train),
        "n_test": len(test),
        "n_failed_train": int(train["failed"].sum()),
        "n_failed_test": int(test["failed"].sum()),
    }
    if chosen == "lightgbm_tuned":
        result["lightgbm_params"] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in info["lightgbm_params"].items()
        }

    if task == "classification":
        # Inner calibration folds are stratified (not grouped): calibration only fits
        # two sigmoid parameters, so student overlap there has negligible impact.
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        model = CalibratedClassifierCV(clone(base), method="sigmoid", cv=inner)
        model.fit(X_tr, y_tr)
        oof = oof_predictions(model, X_tr, y_tr, splits)
        threshold = threshold_for_recall(y_tr, oof, t["target_recall"])
        p_te = model.predict_proba(X_te)[:, 1]
        test_metrics = {
            k: grouped_bootstrap_ci(fn, y_te, p_te, test["user_uuid"], t["n_bootstrap"], seed)
            for k, fn in {
                "pr_auc": lambda a, b: classification_metrics(a, b)["pr_auc"],
                "roc_auc": lambda a, b: classification_metrics(a, b)["roc_auc"],
                "brier": lambda a, b: classification_metrics(a, b)["brier"],
            }.items()
        }
        result["operating_point_test"] = operating_point(y_te, p_te, threshold)
        result["prevalence_test"] = float(y_te.mean())
        scoring = "average_precision"
        predictions = p_te
    else:
        threshold = None
        model = clone(base).fit(X_tr, y_tr)
        yhat = model.predict(X_te)
        pt = cfg.data["pass_threshold"]
        test_metrics = {
            k: grouped_bootstrap_ci(
                lambda a, b, k=k: regression_metrics(a, b, pt)[k],
                y_te,
                yhat,
                test["user_uuid"],
                t["n_bootstrap"],
                seed,
            )
            for k in ["mae", "rmse", "r2", "pr_auc_from_grade"]
        }
        scoring = "neg_mean_absolute_error"
        predictions = yhat

    result["test_metrics"] = {
        k: {"estimate": v[0], "ci95": [v[1], v[2]]} for k, v in test_metrics.items()
    }

    imp = permutation_importance(
        model, X_te, y_te, scoring=scoring, n_repeats=10, random_state=seed, n_jobs=-1
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": imp.importances_mean,
            "importance_std": imp.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    metadata = {
        **result,
        "features": FEATURES,
        "threshold": threshold,
        "pass_threshold": cfg.data["pass_threshold"],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save_model(model, metadata, cfg.path("models_dir"), task, cutoff)

    # No student/course ids: row-level data from the dataset is not redistributed.
    test_pred = test[["final_grade", "failed"]].assign(
        prediction=predictions, cutoff=cutoff, task=task
    )
    return {
        "result": result,
        "cv": cv.assign(cutoff=cutoff, task=task),
        "summary": summary.reset_index().assign(cutoff=cutoff, task=task),
        "importance": importance.assign(cutoff=cutoff, task=task),
        "test_predictions": test_pred,
    }


def train_all(tables: CleanTables, cfg: Config) -> dict:
    out_dir = cfg.path("metrics_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [train_one(tables, c, task, cfg) for c in cfg.features["cutoffs"] for task in TASKS]

    pd.concat([r["cv"] for r in runs]).to_csv(out_dir / "cv_folds.csv", index=False)
    summary = pd.concat([r["summary"] for r in runs])
    summary.to_csv(out_dir / "model_comparison.csv", index=False)
    pd.concat([r["importance"] for r in runs]).to_csv(
        out_dir / "feature_importance.csv", index=False
    )
    pd.concat([r["test_predictions"] for r in runs]).to_csv(
        out_dir / "test_predictions.csv", index=False
    )
    results = [r["result"] for r in runs]
    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2, default=float))
    return {"results": results, "summary": summary}
