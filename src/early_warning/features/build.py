"""Point-in-time feature engineering.

Features for a cutoff ``c`` (course day, 1..60) use only events whose
``particion < c``: exactly what an early-warning system would know on day ``c``.
The *same* function is used for training, batch scoring and the online API,
which removes training/serving skew by construction.

All aggregations are vectorized ``groupby`` operations, so cost is linear in
the number of events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from early_warning.data.clean import CleanTables
from early_warning.data.schema import KEYS

EXAM_CODES_ORDER = ["P1", "P2", "R1", "R2", "INT"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "context": ["n_courses_in_period", "course_size"],
    "engagement": [
        "n_submissions",
        "n_assignments_seen",
        "submission_rate",
        "active_days",
        "days_since_last_activity",
        "recent_submissions",
        "n_ungraded",
        "share_quiz",
        "share_upload",
        "share_external",
    ],
    "performance": [
        "mean_score",
        "min_score",
        "std_score",
        "share_low_score",
        "last_score",
    ],
    "timing": ["median_hours_before_due", "share_last_24h"],
    "exams": [
        "n_exams",
        "grade_P1",
        "grade_P2",
        "grade_R1",
        "grade_R2",
        "grade_INT",
        "took_R1",
        "took_R2",
        "best_first_partial",
        "best_second_partial",
        "mean_exam_grade",
        "min_exam_grade",
        "any_exam_below_pass",
    ],
}
FEATURES: list[str] = [f for group in FEATURE_GROUPS.values() for f in group]


def _context(enr: pd.DataFrame) -> pd.DataFrame:
    """Features known on day 1 (no activity needed)."""
    out = enr[KEYS + ["periodo"]].copy()
    out["n_courses_in_period"] = out.groupby(["user_uuid", "periodo"])["course_uuid"].transform(
        "count"
    )
    out["course_size"] = out.groupby("course_uuid")["user_uuid"].transform("count")
    return out.drop(columns="periodo")


def _submission_features(
    subs: pd.DataFrame, cutoff: int, recent_days: int, low_thr: float
) -> pd.DataFrame:
    if subs.empty:
        return pd.DataFrame(columns=KEYS)
    s = subs.copy()
    s["is_low"] = (s["score"] < low_thr).where(s["score"].notna())
    s["is_recent"] = s["particion"] >= cutoff - recent_days
    t = s["submission_type"].fillna("")
    s["is_quiz"] = t.eq("online_quiz")
    s["is_upload"] = t.eq("online_upload")
    s["is_external"] = t.isin(["external_tool", "basic_lti_launch"])
    s["last_24h"] = (s["hours_before_due"] < 24).where(s["hours_before_due"].notna())
    s = s.sort_values("particion")
    g = s.groupby(KEYS)
    out = pd.DataFrame(
        {
            "n_submissions": g.size(),
            "n_ungraded": g.size() - g["score"].count(),
            "recent_submissions": g["is_recent"].sum(),
            "share_quiz": g["is_quiz"].mean(),
            "share_upload": g["is_upload"].mean(),
            "share_external": g["is_external"].mean(),
            "mean_score": g["score"].mean(),
            "min_score": g["score"].min(),
            "std_score": g["score"].std(),
            "share_low_score": g["is_low"].mean(),
            "last_score": g["score"].last(),
            "median_hours_before_due": g["hours_before_due"].median(),
            "share_last_24h": g["last_24h"].mean(),
        }
    )
    return out.reset_index()


def _exam_features(exams: pd.DataFrame, pass_thr: float) -> pd.DataFrame:
    if exams.empty:
        return pd.DataFrame(columns=KEYS)
    wide = exams.pivot_table(index=KEYS, columns="exam_code", values="grade", aggfunc="max")
    wide = wide.reindex(columns=EXAM_CODES_ORDER)
    wide.columns = [f"grade_{c}" for c in wide.columns]
    g = exams.groupby(KEYS)["grade"]
    out = wide.assign(
        n_exams=g.size(),
        took_R1=wide["grade_R1"].notna().astype(int),
        took_R2=wide["grade_R2"].notna().astype(int),
        best_first_partial=wide[["grade_P1", "grade_R1"]].max(axis=1),
        best_second_partial=wide[["grade_P2", "grade_R2"]].max(axis=1),
        mean_exam_grade=g.mean(),
        min_exam_grade=g.min(),
        any_exam_below_pass=(g.min() < pass_thr).astype(int),
    )
    return out.reset_index()


def build_features(
    tables: CleanTables,
    cutoff: int,
    recent_days: int = 7,
    low_score_threshold: float = 60,
    pass_threshold: float = 5.0,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per enrollment with :data:`FEATURES` computed at ``cutoff``.

    ``context`` can override the day-1 context features (used by the API, where
    a single enrollment arrives without its cohort).
    """
    if not 1 <= cutoff <= 60:
        raise ValueError("cutoff must be a course day in 1..60")
    enr = tables.enrollments
    before = lambda t: t[t["particion"] < cutoff]  # noqa: E731 - point-in-time filter

    subs = before(tables.submissions)
    exams = before(tables.exams)
    aevents = before(tables.assignment_events)

    feats = context.copy() if context is not None else _context(enr)
    feats = feats.merge(
        _submission_features(subs, cutoff, recent_days, low_score_threshold), on=KEYS, how="left"
    )
    feats = feats.merge(_exam_features(exams, pass_threshold), on=KEYS, how="left")

    n_assign = aevents.groupby(KEYS).size().rename("n_assignments_seen").reset_index()
    feats = feats.merge(n_assign, on=KEYS, how="left")

    activity = pd.concat([subs[KEYS + ["particion"]], exams[KEYS + ["particion"]]])
    act = activity.groupby(KEYS)["particion"].agg(active_days="nunique", last_day="max")
    feats = feats.merge(act.reset_index(), on=KEYS, how="left")

    # Guarantee every column exists even when no event happened before the cutoff.
    feats = feats.reindex(columns=list(dict.fromkeys(list(feats.columns) + FEATURES)))
    # Counts are 0 (not missing) when nothing happened yet.
    for c in [
        "n_submissions",
        "n_ungraded",
        "recent_submissions",
        "n_exams",
        "n_assignments_seen",
        "active_days",
        "took_R1",
        "took_R2",
    ]:
        feats[c] = feats[c].fillna(0)
    feats["submission_rate"] = feats["n_submissions"] / feats["n_assignments_seen"].replace(
        0, np.nan
    )
    feats["days_since_last_activity"] = (cutoff - 1 - feats["last_day"].astype("float64")).fillna(
        cutoff
    )
    feats = feats.drop(columns="last_day")

    feats = feats.reindex(columns=KEYS + FEATURES)
    feats[FEATURES] = feats[FEATURES].astype("float64")
    return feats


def build_training_frame(tables: CleanTables, cutoff: int, **kw) -> pd.DataFrame:
    """Features + targets for modeling, excluding enrollments flagged in cleaning."""
    feats = build_features(tables, cutoff, **kw)
    target = tables.enrollments.loc[
        ~tables.enrollments["excluded"], KEYS + ["final_grade", "passed"]
    ]
    frame = feats.merge(target, on=KEYS, how="inner")
    frame["failed"] = 1 - frame["passed"]
    return frame
