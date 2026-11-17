"""Synthetic fixtures that mimic the raw file's shape and its known quirks.

Tests never need the real dataset, so CI runs without data access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from early_warning.data.clean import clean
from early_warning.data.schema import REQUIRED_COLUMNS

T0 = 1_646_000_000  # ~2022-02-28


def make_raw(n_students: int = 30, n_courses: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_students):
        for c in range(n_courses):
            ability = rng.normal(7, 1.8)
            final = float(np.clip(round(ability), 0, 10))
            base = {
                "user_uuid": f"user-{s}",
                "course_uuid": f"course-{c}",
                "course_name": f"Course {c}",
                "legajo": f"L{s:04d}",
                "periodo": "01-2022" if (s == 0 and c == 0) else "1-2022",
                "nota_final_materia": final,
            }
            for p in range(60):  # one placeholder row per day
                rows.append({**base, "particion": p})
            for code, day in [("PRIMER PARCIAL(20)", 27), ("SEGUNDO PARCIAL(20)", 45)]:
                g = float(np.clip(round(ability + rng.normal(0, 1)), 0, 10))
                exam = {
                    **base,
                    "particion": day,
                    "nombre_examen": code,
                    "nota_parcial": g,
                    "fecha_mesa_epoch": T0 + day * 86400,
                }
                rows += [exam, exam]  # exam repeated across activity rows of the same day
            for k in range(4):
                day = 5 + k * 12
                aid = 1000 * c + k
                rows.append(
                    {
                        **base,
                        "particion": day,
                        "assignment_id": aid,
                        "ass_name": f"TP{k}",
                        "ass_created_at": T0,
                        "ass_due_at": T0 + (day + 3) * 86400,
                        "ass_unlock_at": T0,
                        "ass_lock_at": T0 + (day + 3) * 86400,
                        "points_possible": 100.0,
                    }
                )
                if rng.random() < 0.85:
                    sub_t = T0 + (day + 1) * 86400
                    rows.append(
                        {
                            **base,
                            "particion": day + 1,
                            "ass_name_sub": f"TP{k}",
                            "sub_uuid": f"sub-{s}-{c}-{k}",
                            "score": float(np.clip(ability * 10 + rng.normal(0, 8), 0, 100)),
                            "submission_type": "online_quiz",
                            "s_submitted_at": sub_t,
                            "s_graded_at": sub_t + 3600,
                            "s_created_at": 1_659_392_827,
                        }
                    )
    df = pd.DataFrame(rows).reindex(columns=REQUIRED_COLUMNS)
    df.loc[(df.user_uuid == "user-1") & (df.course_uuid == "course-0"), "nota_final_materia"] = 0.0
    # Force at least a few failures for stratified splits.
    fail = df.user_uuid.isin([f"user-{i}" for i in range(2, 8)])
    df.loc[fail, "nota_final_materia"] = 3.0
    return df


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def tables(raw):
    tables, _ = clean(raw)
    return tables
