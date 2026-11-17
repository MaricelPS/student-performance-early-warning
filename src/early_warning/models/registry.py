"""Minimal file-based model registry.

Each model is stored as ``models/<task>_day<cutoff>.joblib`` together with a
JSON sidecar holding its metadata (features, threshold, metrics, versions).
Swapping this for MLflow only requires reimplementing these two functions.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import sklearn

from early_warning import __version__


def model_path(models_dir: Path, task: str, cutoff: int) -> Path:
    return Path(models_dir) / f"{task}_day{cutoff:02d}.joblib"


def save_model(model, metadata: dict, models_dir: Path, task: str, cutoff: int) -> Path:
    path = model_path(models_dir, task, cutoff)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **metadata,
        "package_version": __version__,
        "sklearn_version": sklearn.__version__,
        "python": platform.python_version(),
    }
    joblib.dump({"model": model, "metadata": metadata}, path, compress=3)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=float))
    return path


def load_model(path: Path) -> tuple[object, dict]:
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]


def available_models(models_dir: Path) -> dict[tuple[str, int], Path]:
    out = {}
    for p in sorted(Path(models_dir).glob("*_day*.joblib")):
        task, day = p.stem.rsplit("_day", 1)
        out[(task, int(day))] = p
    return out
