"""Cleaning and normalization of the raw activity log.

The raw file is a wide, denormalized log: one row per (student, course, day)
*and* per activity. Exam, assignment and submission columns live side by side
in the same row but are **not** related to each other (e.g. a row can hold the
assignment "TP1" next to a submission for "API2"). Cleaning therefore splits
the log into independent, de-duplicated tables:

* ``enrollments`` – one row per student-course (with the target)
* ``exams``       – one row per student-course-exam
* ``assignments`` – one row per assignment (course catalog)
* ``assignment_events`` – when an assignment appears in a student's timeline
* ``submissions`` – one row per submission (``sub_uuid``)

Every decision is recorded in a :class:`CleaningReport` so it can be audited
and rendered in the docs (see ``docs/data_quality_decisions.md``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from early_warning.data.schema import EXAM_CODES, KEYS, N_PARTITIONS


@dataclass
class CleaningReport:
    items: list[dict] = field(default_factory=list)

    def add(self, check: str, affected: int, decision: str) -> None:
        self.items.append({"check": check, "affected": int(affected), "decision": decision})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.items)

    def to_markdown(self) -> str:
        lines = ["| Check | Affected | Decision |", "|---|---:|---|"]
        lines += [f"| {i['check']} | {i['affected']:,} | {i['decision']} |" for i in self.items]
        return "\n".join(lines)


@dataclass
class CleanTables:
    enrollments: pd.DataFrame
    exams: pd.DataFrame
    assignments: pd.DataFrame
    assignment_events: pd.DataFrame
    submissions: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "enrollments": self.enrollments,
            "exams": self.exams,
            "assignments": self.assignments,
            "assignment_events": self.assignment_events,
            "submissions": self.submissions,
        }


def _to_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, unit="s", utc=True)


def normalize_periodo(s: pd.Series) -> pd.Series:
    """'01-2022' -> '1-2022'. Any other shape is left untouched (and reported)."""
    return s.str.replace(r"^0+(\d)-", r"\1-", regex=True)


def parse_exam_name(name: str) -> tuple[str | None, float | None]:
    """'RECUPERATORIO PRIMER PARCIAL(20)' -> ('R1', 20.0)."""
    if name is None or pd.isna(name):
        return None, None
    m = re.match(r"^\s*(.*?)\s*\((\d+)\)\s*$", str(name))
    base, weight = (m.group(1), float(m.group(2))) if m else (str(name).strip(), None)
    return EXAM_CODES.get(base.upper()), weight


def clean(
    raw: pd.DataFrame, pass_threshold: float = 5.0, exclude_zero_final_grade: bool = True
) -> tuple[CleanTables, CleaningReport]:
    rep = CleaningReport()
    df = raw.copy()

    # 1. Exact duplicate rows -------------------------------------------------
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    rep.add(
        "Exact duplicate rows",
        n_dup,
        "Dropped. They carry no extra information (mostly empty day-placeholder rows).",
    )

    # 2. Period label format --------------------------------------------------
    df["periodo_raw"] = df["periodo"]
    df["periodo"] = normalize_periodo(df["periodo"])
    n_fmt = int((df["periodo_raw"] != df["periodo"]).sum())
    rep.add(
        "`periodo` written as '01-2022' instead of '1-2022'",
        n_fmt,
        "Normalized to the dictionary format 'S-YYYY'. Raw label kept in `periodo_raw`. "
        "Note: these enrollments ran Jan-Mar 2022, earlier than the bulk of 1-2022.",
    )
    bad = ~df["periodo"].str.fullmatch(r"[12]-\d{4}")
    if bad.any():
        rep.add("Unparseable `periodo`", int(bad.sum()), "Rows dropped.")
        df = df[~bad]

    # 3. Partition range --------------------------------------------------------
    pmin, pmax = int(df["particion"].min()), int(df["particion"].max())
    rep.add(
        f"`particion` ranges {pmin}..{pmax}, dictionary says 1..60",
        len(df),
        "Kept as a 0-based day index (every enrollment has exactly 60 partitions). "
        "`day = particion + 1` is used in reports.",
    )
    out_of_range = (df["particion"] < 0) | (df["particion"] >= N_PARTITIONS)
    if out_of_range.any():
        rep.add("`particion` out of 0..59", int(out_of_range.sum()), "Rows dropped.")
        df = df[~out_of_range]

    # 4. Identity: user_uuid vs legajo -----------------------------------------
    legajos_per_user = df.groupby("user_uuid")["legajo"].nunique()
    rep.add(
        "Students (`user_uuid`) with more than one `legajo`",
        int((legajos_per_user > 1).sum()),
        "`user_uuid` is the identity key; `legajo` is an administrative attribute "
        "and is not used for joins or modeling.",
    )
    names_per_course = df.groupby("course_name")["course_uuid"].nunique()
    rep.add(
        "`course_name` shared by several `course_uuid`",
        int((names_per_course > 1).sum()),
        "`course_uuid` is the course key; names are display-only.",
    )

    # 5. Target consistency -----------------------------------------------------
    grades_per_enr = df.groupby(KEYS)["nota_final_materia"].nunique(dropna=False)
    n_incons = int((grades_per_enr > 1).sum())
    rep.add(
        "Enrollments with more than one final grade",
        n_incons,
        "None expected; if present, the last partition's grade is kept (dictionary: "
        "'última nota del examen final').",
    )

    last = df.sort_values("particion").groupby(KEYS, as_index=False).last()
    enr = last[
        KEYS + ["legajo", "course_name", "periodo", "periodo_raw", "nota_final_materia"]
    ].rename(columns={"nota_final_materia": "final_grade"})
    enr["n_partitions"] = df.groupby(KEYS)["particion"].nunique().values

    zero = enr["final_grade"] == 0
    rep.add(
        "Final grade = 0",
        int(zero.sum()),
        "Treated as 'absent at the final exam' (these students have partial-exam "
        "averages of 5-9.5). "
        + (
            "Excluded from modeling (ambiguous label)."
            if exclude_zero_final_grade
            else "Kept as failures."
        ),
    )
    enr["excluded"] = zero & exclude_zero_final_grade
    enr["passed"] = (enr["final_grade"] >= pass_threshold).astype(int)

    # 6. Exams ------------------------------------------------------------------
    ex = df.loc[
        df["nombre_examen"].notna(),
        KEYS + ["particion", "nombre_examen", "nota_parcial", "fecha_mesa_epoch"],
    ]
    n_before = len(ex)
    ex = ex.drop_duplicates(KEYS + ["nombre_examen"])
    rep.add(
        "Exam rows repeated across activity rows of the same day",
        n_before - len(ex),
        "De-duplicated to one row per student-course-exam (grade and day were identical).",
    )
    parsed = ex["nombre_examen"].map(parse_exam_name)
    ex = ex.assign(
        exam_code=[p[0] for p in parsed],
        exam_weight=[p[1] for p in parsed],
        exam_date=_to_ts(ex["fecha_mesa_epoch"]),
    ).rename(columns={"nota_parcial": "grade", "nombre_examen": "exam_name"})
    unknown = ex["exam_code"].isna()
    if unknown.any():
        rep.add("Unknown exam names", int(unknown.sum()), "Kept with exam_code = 'OTHER'.")
        ex.loc[unknown, "exam_code"] = "OTHER"
    exams = ex[KEYS + ["particion", "exam_code", "exam_name", "exam_weight", "grade", "exam_date"]]

    # 7. Assignments (catalog + per-student events) -----------------------------
    a = df[df["assignment_id"].notna()]
    assignments = (
        a[
            [
                "course_uuid",
                "assignment_id",
                "ass_name",
                "ass_created_at",
                "ass_due_at",
                "ass_unlock_at",
                "ass_lock_at",
                "points_possible",
            ]
        ]
        .drop_duplicates("assignment_id")
        .rename(columns={"ass_name": "name"})
    )
    for c in ["ass_created_at", "ass_due_at", "ass_unlock_at", "ass_lock_at"]:
        assignments[c.replace("ass_", "")] = _to_ts(assignments.pop(c))
    rep.add(
        "Assignments with `points_possible` = 0 but graded submissions",
        int((assignments["points_possible"] == 0).sum()),
        "Configuration error in the LMS. Scores are on a 0-100 scale everywhere, so "
        "`score` is used directly as a percentage instead of `score / points_possible`.",
    )
    late = (assignments["lock_at"] != assignments["due_at"]).sum()
    rep.add(
        "Assignments whose lock date differs from due date",
        int(late),
        "Lock = due for almost all assignments, so 'late submission' is not a usable "
        "signal; submission margin (hours before due) is used instead.",
    )
    assignment_events = a[KEYS + ["particion", "assignment_id"]].drop_duplicates(
        KEYS + ["assignment_id"]
    )

    # 8. Submissions ------------------------------------------------------------
    s = df[df["sub_uuid"].notna()]
    n_before = len(s)
    s = s.drop_duplicates("sub_uuid")
    rep.add(
        "Submission rows repeated (same `sub_uuid`)",
        n_before - len(s),
        "De-duplicated by `sub_uuid` (all attributes were identical).",
    )
    subs = s[
        KEYS
        + [
            "particion",
            "sub_uuid",
            "ass_name_sub",
            "score",
            "submission_type",
            "s_submitted_at",
            "s_graded_at",
            "s_created_at",
        ]
    ].rename(columns={"ass_name_sub": "assignment_name"})
    rep.add(
        "Submission `s_created_at` later than `s_submitted_at`",
        int((subs["s_created_at"] > subs["s_submitted_at"]).sum()),
        "~all share 2022-08-01 (a bulk load/migration timestamp). Column dropped.",
    )
    subs = subs.drop(columns="s_created_at")
    subs["submitted_at"] = _to_ts(subs.pop("s_submitted_at"))
    subs["graded_at"] = _to_ts(subs.pop("s_graded_at"))
    neg = subs["graded_at"] < subs["submitted_at"]
    rep.add(
        "Submissions graded before being submitted",
        int(neg.sum()),
        "Grading timestamp set to missing (impossible order).",
    )
    subs.loc[neg, "graded_at"] = pd.NaT
    rep.add(
        "Submissions without score (not graded)",
        int(subs["score"].isna().sum()),
        "Kept: they count as submissions; excluded from score statistics.",
    )
    out = (subs["score"] < 0) | (subs["score"] > 100)
    rep.add("Scores outside 0..100", int(out.sum()), "Clipped to 0..100.")
    subs["score"] = subs["score"].clip(0, 100)

    # Link submission -> assignment by (course, name): the row-level pairing is
    # meaningless, but names are unique within a course in ~all cases.
    catalog = assignments.sort_values("due_at").drop_duplicates(
        ["course_uuid", "name"], keep="last"
    )
    rep.add(
        "Assignment names duplicated within a course",
        len(assignments) - len(catalog),
        "When linking submissions to assignments by name, the one with the latest "
        "due date is used.",
    )
    subs = subs.merge(
        catalog[["course_uuid", "name", "assignment_id", "due_at"]].rename(
            columns={"name": "assignment_name"}
        ),
        on=["course_uuid", "assignment_name"],
        how="left",
    )
    subs["hours_before_due"] = (subs["due_at"] - subs["submitted_at"]).dt.total_seconds() / 3600
    rep.add(
        "Submissions that could not be linked to an assignment",
        int(subs["assignment_id"].isna().sum()),
        "Kept; their timing features are missing (handled by the models).",
    )

    tables = CleanTables(
        enrollments=enr.reset_index(drop=True),
        exams=exams.reset_index(drop=True),
        assignments=assignments.reset_index(drop=True),
        assignment_events=assignment_events.reset_index(drop=True),
        submissions=subs.reset_index(drop=True),
    )
    return tables, rep


def save_tables(tables: CleanTables, out_dir) -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.as_dict().items():
        t.to_parquet(out / f"{name}.parquet", index=False)


def load_tables(in_dir) -> CleanTables:
    from pathlib import Path

    d = Path(in_dir)
    return CleanTables(
        **{
            n: pd.read_parquet(d / f"{n}.parquet")
            for n in ["enrollments", "exams", "assignments", "assignment_events", "submissions"]
        }
    )


__all__ = ["CleanTables", "CleaningReport", "clean", "load_tables", "save_tables"]
