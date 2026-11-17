"""Custom, picklable preprocessing steps."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropAllMissing(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely missing in the training data.

    Early horizons contain features that cannot exist yet (e.g. second-partial
    grade on day 15). Some estimators (sklearn's HistGradientBoosting) fail on
    all-NaN columns, so they are removed at fit time and ignored at predict time.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_ = ~np.isnan(X).all(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]
