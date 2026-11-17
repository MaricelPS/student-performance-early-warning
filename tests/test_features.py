import numpy as np
import pandas as pd
import pytest

from early_warning.data.clean import CleanTables
from early_warning.features.build import FEATURES, build_features, build_training_frame


def _with_future_events(tables: CleanTables, cutoff: int) -> CleanTables:
    """Copy of tables with extra, extreme events on/after the cutoff."""
    subs = tables.submissions
    future_subs = subs.assign(particion=cutoff, score=0.0, sub_uuid=subs["sub_uuid"] + "-future")
    exams = tables.exams
    future_exams = exams.assign(particion=59, exam_code="INT", grade=0.0)
    return CleanTables(
        enrollments=tables.enrollments,
        exams=pd.concat([exams, future_exams], ignore_index=True),
        assignments=tables.assignments,
        assignment_events=tables.assignment_events,
        submissions=pd.concat([subs, future_subs], ignore_index=True),
    )


@pytest.mark.parametrize("cutoff", [15, 30, 45])
def test_no_leakage_from_future_events(tables, cutoff):
    """Events on or after the cutoff must not change any feature."""
    before = build_features(tables, cutoff)
    after = build_features(_with_future_events(tables, cutoff), cutoff)
    pd.testing.assert_frame_equal(before, after)


def test_columns_and_rows(tables):
    f = build_features(tables, 30)
    assert list(f.columns) == ["user_uuid", "course_uuid"] + FEATURES
    assert len(f) == len(tables.enrollments)
    assert f[FEATURES].dtypes.eq("float64").all()


def test_counts_are_zero_before_any_activity(tables):
    f = build_features(tables, 1)
    for c in ["n_submissions", "n_exams", "active_days", "n_assignments_seen"]:
        assert (f[c] == 0).all(), c
    assert (f["days_since_last_activity"] == 1).all()
    assert f["grade_P1"].isna().all()


def test_exam_features_appear_after_exam_day(tables):
    assert build_features(tables, 27)["grade_P1"].isna().all()  # P1 is on particion 27
    f = build_features(tables, 28)
    assert f["grade_P1"].notna().all()
    assert np.allclose(f["best_first_partial"], f["grade_P1"])


def test_context_override(tables):
    ctx = tables.enrollments[["user_uuid", "course_uuid"]].assign(
        n_courses_in_period=9, course_size=99
    )
    f = build_features(tables, 30, context=ctx)
    assert (f["n_courses_in_period"] == 9).all() and (f["course_size"] == 99).all()


@pytest.mark.parametrize("cutoff", [0, 61])
def test_invalid_cutoff(tables, cutoff):
    with pytest.raises(ValueError):
        build_features(tables, cutoff)


def test_training_frame_excludes_flagged(tables):
    frame = build_training_frame(tables, 60)
    assert len(frame) == (~tables.enrollments["excluded"]).sum()
    assert set(frame["failed"]) <= {0, 1}
    assert (frame["failed"] == (frame["final_grade"] < 5)).all()
