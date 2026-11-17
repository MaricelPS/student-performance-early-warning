import pandas as pd
import pytest

from early_warning.data.clean import clean, normalize_periodo, parse_exam_name
from early_warning.data.load import SchemaError, load_raw


def test_load_raw_rejects_missing_columns(tmp_path, raw):
    path = tmp_path / "bad.csv"
    raw.drop(columns=["score"]).to_csv(path, sep=";", index=False)
    with pytest.raises(SchemaError, match="score"):
        load_raw(path)


def test_load_raw_roundtrip(tmp_path, raw):
    path = tmp_path / "ok.csv"
    raw.to_csv(path, sep=";", index=False)
    df = load_raw(path)
    assert len(df) == len(raw)
    assert str(df["particion"].dtype) == "Int16"


@pytest.mark.parametrize(
    "raw_value, expected", [("01-2022", "1-2022"), ("1-2022", "1-2022"), ("2-2022", "2-2022")]
)
def test_normalize_periodo(raw_value, expected):
    assert normalize_periodo(pd.Series([raw_value])).iloc[0] == expected


@pytest.mark.parametrize(
    "name, code, weight",
    [
        ("PRIMER PARCIAL(20)", "P1", 20.0),
        ("RECUPERATORIO SEGUNDO PARCIAL(20)", "R2", 20.0),
        ("INTEGRADOR(30)", "INT", 30.0),
        ("UNKNOWN", None, None),
    ],
)
def test_parse_exam_name(name, code, weight):
    assert parse_exam_name(name) == (code, weight)


def test_one_row_per_enrollment(tables, raw):
    enr = tables.enrollments
    assert not enr.duplicated(["user_uuid", "course_uuid"]).any()
    assert len(enr) == raw[["user_uuid", "course_uuid"]].drop_duplicates().shape[0]
    assert (enr["n_partitions"] == 60).all()


def test_periodo_normalized_and_raw_kept(tables):
    enr = tables.enrollments
    assert set(enr["periodo"]) == {"1-2022"}
    assert "01-2022" in set(enr["periodo_raw"])


def test_zero_final_grade_is_excluded(tables):
    enr = tables.enrollments.set_index(["user_uuid", "course_uuid"])
    assert enr.loc[("user-1", "course-0"), "excluded"]
    assert enr["excluded"].sum() == 1


def test_zero_final_grade_kept_when_configured(raw):
    tables, _ = clean(raw, exclude_zero_final_grade=False)
    assert not tables.enrollments["excluded"].any()


def test_exams_deduplicated(tables):
    assert not tables.exams.duplicated(["user_uuid", "course_uuid", "exam_code"]).any()
    assert set(tables.exams["exam_code"]) == {"P1", "P2"}


def test_submissions_linked_and_cleaned(tables):
    s = tables.submissions
    assert s["sub_uuid"].is_unique
    assert "s_created_at" not in s.columns
    assert s["assignment_id"].notna().all()
    assert (s["hours_before_due"] > 0).all()


def test_graded_before_submitted_is_nulled(raw):
    bad = raw.copy()
    i = bad.index[bad["sub_uuid"].notna()][0]
    bad.loc[i, "s_graded_at"] = bad.loc[i, "s_submitted_at"] - 10
    tables, report = clean(bad)
    s = tables.submissions.set_index("sub_uuid")
    assert pd.isna(s.loc[bad.loc[i, "sub_uuid"], "graded_at"])
    item = next(x for x in report.items if x["check"].startswith("Submissions graded before"))
    assert item["affected"] == 1


def test_report_renders_markdown(raw):
    _, report = clean(raw)
    md = report.to_markdown()
    assert md.startswith("| Check |")
    assert "Exact duplicate rows" in md
