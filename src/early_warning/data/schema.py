"""Raw-data contract: expected columns and dtypes of the challenge CSV."""

from __future__ import annotations

KEYS = ["user_uuid", "course_uuid"]

RAW_DTYPES: dict[str, str] = {
    "particion": "Int16",
    "periodo": "string",
    "nota_final_materia": "float64",
    "fecha_mesa_epoch": "float64",
    "nombre_examen": "string",
    "nota_parcial": "float64",
    "assignment_id": "Int64",
    "ass_name": "string",
    "ass_created_at": "float64",
    "ass_due_at": "float64",
    "ass_unlock_at": "float64",
    "ass_lock_at": "float64",
    "points_possible": "float64",
    "ass_name_sub": "string",
    "sub_uuid": "string",
    "score": "float64",
    "submission_type": "string",
    "s_submitted_at": "float64",
    "s_graded_at": "float64",
    "s_created_at": "float64",
    "user_uuid": "string",
    "course_uuid": "string",
    "course_name": "string",
    "legajo": "string",
}

REQUIRED_COLUMNS = list(RAW_DTYPES)

# Canonical exam codes. The number in parentheses in the raw name is the exam's
# weight in the course; it is parsed separately.
EXAM_CODES = {
    "PRIMER PARCIAL": "P1",
    "SEGUNDO PARCIAL": "P2",
    "RECUPERATORIO PRIMER PARCIAL": "R1",
    "RECUPERATORIO SEGUNDO PARCIAL": "R2",
    "INTEGRADOR": "INT",
}

N_PARTITIONS = 60  # course days, stored 0-based in the raw file (0..59)
