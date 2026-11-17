"""End-to-end smoke test of the CLI on synthetic data (tiny training budget)."""

import json

import pandas as pd
import yaml

from early_warning.cli import main
from tests.conftest import make_raw


def test_cli_all_end_to_end(tmp_path):
    raw_path = tmp_path / "raw.csv"
    make_raw(n_students=40, n_courses=3, seed=1).to_csv(raw_path, sep=";", index=False)
    cfg = {
        "paths": {
            "raw_csv": str(raw_path),
            "interim_dir": str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
            "models_dir": str(tmp_path / "models"),
            "metrics_dir": str(tmp_path / "metrics"),
            "bi_dir": str(tmp_path / "bi"),
        },
        "data": {"csv_separator": ";", "pass_threshold": 5.0, "exclude_zero_final_grade": True},
        "features": {"cutoffs": [30, 60], "low_score_threshold": 60, "recent_window_days": 7},
        "training": {
            "random_state": 0,
            "test_size": 0.25,
            "cv_folds": 2,
            "cv_repeats": 1,
            "tuning_iterations": 2,
            "target_recall": 0.8,
            "n_bootstrap": 20,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    main(["--config", str(cfg_path), "all"])

    assert (tmp_path / "metrics" / "data_quality_report.md").exists()
    assert {p.name for p in (tmp_path / "models").glob("*.joblib")} == {
        "classification_day30.joblib",
        "classification_day60.joblib",
        "regression_day30.joblib",
        "regression_day60.joblib",
    }
    results = json.loads((tmp_path / "metrics" / "test_results.json").read_text())
    assert len(results) == 4
    comparison = pd.read_csv(tmp_path / "metrics" / "model_comparison.csv")
    assert comparison.groupby(["task", "cutoff"])["selected"].sum().eq(1).all()
    scores = pd.read_csv(tmp_path / "processed" / "risk_scores_day60.csv")
    assert scores["fail_probability"].between(0, 1).all()
    assert (tmp_path / "bi" / "fact_risk_score.csv").exists()
