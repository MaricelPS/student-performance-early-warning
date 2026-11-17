"""FastAPI service for online and micro-batch scoring.

The service is stateless: models are loaded once at start-up from
``EW_MODELS_DIR`` (default ``models/``) and every request carries the activity
of the enrollment(s) to score. Features are computed with the *same*
``build_features`` function used in training, so there is no training/serving
skew. Run with::

    uvicorn early_warning.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from early_warning import __version__
from early_warning.api.schemas import (
    BatchRequest,
    BatchResponse,
    EnrollmentActivity,
    Prediction,
)
from early_warning.config import PROJECT_ROOT
from early_warning.data.clean import CleanTables
from early_warning.data.schema import KEYS
from early_warning.features.build import build_features
from early_warning.models.predict import ModelStore

log = logging.getLogger("early_warning.api")


def to_tables(items: list[EnrollmentActivity]) -> tuple[CleanTables, pd.DataFrame]:
    """Turn API payloads into the cleaned-table format expected by the features.

    Each payload gets a positional surrogate key, so a batch may contain the same
    enrollment several times (e.g. scored at different days) without collisions.
    Context features come from the payload, not from the cohort.
    """
    keys = [{"user_uuid": str(i), "course_uuid": "_"} for i in range(len(items))]
    exams = pd.DataFrame(
        [{**k, **x.model_dump()} for k, e in zip(keys, items, strict=True) for x in e.exams],
        columns=KEYS + ["exam_code", "particion", "grade"],
    )
    subs = pd.DataFrame(
        [{**k, **s.model_dump()} for k, e in zip(keys, items, strict=True) for s in e.submissions],
        columns=KEYS + ["particion", "score", "submission_type", "hours_before_due"],
    )
    subs["score"] = subs["score"].astype("float64")
    subs["hours_before_due"] = subs["hours_before_due"].astype("float64")
    aev = pd.DataFrame(
        [{**k, **a.model_dump()} for k, e in zip(keys, items, strict=True) for a in e.assignments],
        columns=KEYS + ["particion", "assignment_id"],
    )
    context = pd.DataFrame(
        [
            {**k, "n_courses_in_period": e.n_courses_in_period, "course_size": e.course_size}
            for k, e in zip(keys, items, strict=True)
        ]
    )
    enr = pd.DataFrame(keys)
    tables = CleanTables(
        enrollments=enr,
        exams=exams,
        assignments=pd.DataFrame(),
        assignment_events=aev,
        submissions=subs,
    )
    return tables, context


def score(store: ModelStore, items: list[EnrollmentActivity]) -> list[Prediction]:
    out: list[Prediction] = []
    by_day: dict[int, list[EnrollmentActivity]] = {}
    for e in items:
        by_day.setdefault(e.day, []).append(e)
    for day, group in by_day.items():
        clf = store.for_day("classification", day)
        reg = store.for_day("regression", day)
        tables, context = to_tables(group)
        feats = build_features(tables, day, context=context)
        p = clf.predict(feats)
        g = reg.predict(feats).clip(0, 10)
        thr = float(clf.metadata["threshold"])
        for e, pi, gi in zip(group, p, g, strict=True):
            out.append(
                Prediction(
                    user_uuid=e.user_uuid,
                    course_uuid=e.course_uuid,
                    day=day,
                    model_day=clf.metadata["cutoff"],
                    fail_probability=round(float(pi), 4),
                    at_risk=bool(pi >= thr),
                    alert_threshold=round(thr, 4),
                    predicted_grade=round(float(gi), 2),
                    models={
                        "classifier": clf.metadata["selected_model"],
                        "regressor": reg.metadata["selected_model"],
                    },
                )
            )
    return out


def create_app(models_dir: str | Path | None = None) -> FastAPI:
    models_dir = Path(models_dir or os.environ.get("EW_MODELS_DIR", PROJECT_ROOT / "models"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = ModelStore(models_dir)
        log.info("loaded %s models from %s", len(app.state.store.models), models_dir)
        yield

    app = FastAPI(
        title="Student early-warning API",
        version=__version__,
        lifespan=lifespan,
        description="Risk of failing a course, computed from activity up to a given day.",
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    @app.get("/health")
    def health(request: Request):
        store: ModelStore = request.app.state.store
        ok = bool(store.cutoffs("classification")) and bool(store.cutoffs("regression"))
        return {"status": "ok" if ok else "no-models", "version": __version__}

    @app.get("/models")
    def models(request: Request):
        store: ModelStore = request.app.state.store
        return [
            {
                "task": t,
                "day": d,
                "model": m.metadata["selected_model"],
                "trained_at": m.metadata["trained_at"],
                "threshold": m.metadata["threshold"],
                "test_metrics": {
                    k: round(v["estimate"], 4) for k, v in m.metadata["test_metrics"].items()
                },
            }
            for (t, d), m in sorted(store.models.items())
        ]

    @app.post("/predict", response_model=Prediction)
    def predict(item: EnrollmentActivity, request: Request):
        try:
            return score(request.app.state.store, [item])[0]
        except LookupError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest, request: Request):
        try:
            return BatchResponse(predictions=score(request.app.state.store, req.enrollments))
        except LookupError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
