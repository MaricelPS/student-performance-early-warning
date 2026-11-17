"""API tests against small models trained on the synthetic fixture."""

import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

from early_warning.api.app import create_app
from early_warning.api.schemas import EnrollmentActivity
from early_warning.features.build import FEATURES, build_training_frame
from early_warning.models.registry import save_model

EXAMPLE = EnrollmentActivity.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client(tables, tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    for cutoff in (30, 45):
        frame = build_training_frame(tables, cutoff)
        X = frame[FEATURES]
        imp = lambda: SimpleImputer(strategy="median", keep_empty_features=True)  # noqa: E731
        clf = make_pipeline(imp(), LogisticRegression(max_iter=500)).fit(X, frame["failed"])
        reg = make_pipeline(imp(), LinearRegression()).fit(X, frame["final_grade"])
        meta = {
            "cutoff": cutoff,
            "features": FEATURES,
            "selected_model": "test",
            "trained_at": "now",
            "test_metrics": {"m": {"estimate": 0.5}},
        }
        save_model(
            clf,
            {**meta, "task": "classification", "threshold": 0.3},
            models_dir,
            "classification",
            cutoff,
        )
        save_model(
            reg, {**meta, "task": "regression", "threshold": None}, models_dir, "regression", cutoff
        )
    with TestClient(create_app(models_dir)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_models_listing(client):
    listed = client.get("/models").json()
    assert {(m["task"], m["day"]) for m in listed} == {
        ("classification", 30),
        ("classification", 45),
        ("regression", 30),
        ("regression", 45),
    }


def test_predict_uses_latest_model_not_after_day(client):
    for day, expected in [(30, 30), (44, 30), (45, 45), (60, 45)]:
        r = client.post("/predict", json={**EXAMPLE, "day": day})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_day"] == expected
        assert 0 <= body["fail_probability"] <= 1
        assert 0 <= body["predicted_grade"] <= 10
        assert body["at_risk"] == (body["fail_probability"] >= body["alert_threshold"])
        assert "X-Process-Time-ms" in r.headers


def test_predict_before_first_horizon_is_422(client):
    r = client.post("/predict", json={**EXAMPLE, "day": 10})
    assert r.status_code == 422
    assert "earliest horizon" in r.json()["detail"]


@pytest.mark.parametrize(
    "patch",
    [
        {"day": 0},
        {"exams": [{"exam_code": "P1", "particion": 10, "grade": 11}]},
        {"exams": [{"exam_code": "XX", "particion": 10, "grade": 5}]},
        {"submissions": [{"particion": 70, "score": 50}]},
    ],
)
def test_invalid_payloads_rejected(client, patch):
    assert client.post("/predict", json={**EXAMPLE, **patch}).status_code == 422


def test_batch_with_duplicate_enrollments(client):
    """Same enrollment scored with different histories must not collide."""
    weak = {**EXAMPLE}
    strong = {**EXAMPLE, "exams": [{"exam_code": "P1", "particion": 27, "grade": 10}]}
    r = client.post("/predict/batch", json={"enrollments": [weak, strong, weak]})
    preds = r.json()["predictions"]
    assert len(preds) == 3
    assert preds[0]["fail_probability"] == preds[2]["fail_probability"]
    assert preds[0]["predicted_grade"] != preds[1]["predicted_grade"]


def test_empty_activity_is_scored(client):
    r = client.post("/predict", json={"user_uuid": "u", "course_uuid": "c", "day": 30})
    assert r.status_code == 200
