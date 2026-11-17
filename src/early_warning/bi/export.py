"""Star-schema export for the Power BI dashboard.

Produces CSVs ready to be loaded in Power BI Desktop (see
``docs/powerbi_data_model.md`` for relationships and suggested DAX measures).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from early_warning.data.clean import CleanTables


def export_star_schema(
    tables: CleanTables, out_dir: Path, risk_scores: pd.DataFrame | None = None
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    e = tables.enrollments.copy()
    e["enrollment_id"] = e["user_uuid"] + "|" + e["course_uuid"]
    key = e[["user_uuid", "course_uuid", "enrollment_id"]]

    frames = {
        "dim_student": e[["user_uuid", "legajo"]].drop_duplicates("user_uuid"),
        "dim_course": (
            e.groupby(["course_uuid", "course_name"], as_index=False).agg(
                n_students=("user_uuid", "nunique"), periodo=("periodo", "first")
            )
        ),
        "dim_day": pd.DataFrame(
            {"particion": range(60), "day": range(1, 61), "week": [d // 7 + 1 for d in range(60)]}
        ),
        "fact_enrollment": e[
            [
                "enrollment_id",
                "user_uuid",
                "course_uuid",
                "periodo",
                "final_grade",
                "passed",
                "excluded",
            ]
        ],
        "fact_exam": tables.exams.merge(key, on=["user_uuid", "course_uuid"])[
            ["enrollment_id", "particion", "exam_code", "exam_name", "exam_weight", "grade"]
        ],
        "fact_submission": tables.submissions.merge(key, on=["user_uuid", "course_uuid"])[
            [
                "enrollment_id",
                "sub_uuid",
                "particion",
                "assignment_name",
                "submission_type",
                "score",
                "hours_before_due",
            ]
        ],
        "dim_assignment": tables.assignments[
            ["assignment_id", "course_uuid", "name", "points_possible", "due_at"]
        ],
    }
    if risk_scores is not None:
        frames["fact_risk_score"] = risk_scores.merge(key, on=["user_uuid", "course_uuid"])
    paths = []
    for name, df in frames.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths
