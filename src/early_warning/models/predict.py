"""Inference helpers shared by the batch CLI and the API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from early_warning.models.registry import available_models, load_model


@dataclass
class LoadedModel:
    model: object
    metadata: dict

    @property
    def features(self) -> list[str]:
        return self.metadata["features"]

    def predict(self, X: pd.DataFrame):
        X = X[self.features]
        if self.metadata["task"] == "classification":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


def _single_threaded(model) -> None:
    """Inference on small payloads is faster without joblib's thread pool; scale
    out with more API workers instead of more threads per request."""
    for name in model.get_params():
        if name == "n_jobs" or name.endswith("__n_jobs"):
            model.set_params(**{name: 1})
    for cal in getattr(model, "calibrated_classifiers_", []):
        _single_threaded(cal.estimator)


class ModelStore:
    """Loads every model once and picks the right horizon for a request."""

    def __init__(self, models_dir: Path, single_threaded: bool = True):
        self.models: dict[tuple[str, int], LoadedModel] = {
            key: LoadedModel(*load_model(path))
            for key, path in available_models(models_dir).items()
        }
        if single_threaded:
            for m in self.models.values():
                _single_threaded(m.model)

    def cutoffs(self, task: str) -> list[int]:
        return sorted(c for t, c in self.models if t == task)

    def for_day(self, task: str, day: int) -> LoadedModel:
        """Latest model whose cutoff is <= day (it never sees future information)."""
        eligible = [c for c in self.cutoffs(task) if c <= day]
        if not eligible:
            raise LookupError(
                f"No {task} model available for day {day}; "
                f"earliest horizon is {min(self.cutoffs(task), default='n/a')}"
            )
        return self.models[(task, max(eligible))]
