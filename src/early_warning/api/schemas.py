"""Request/response contracts of the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Exam(BaseModel):
    exam_code: str = Field(
        ...,
        pattern="^(P1|P2|R1|R2|INT|OTHER)$",
        description="P1/P2 partials, R1/R2 make-ups, INT integrator",
    )
    particion: int = Field(..., ge=0, le=59, description="0-based course day")
    grade: float = Field(..., ge=0, le=10)


class Submission(BaseModel):
    particion: int = Field(..., ge=0, le=59)
    score: float | None = Field(None, ge=0, le=100, description="null if not graded yet")
    submission_type: str | None = None
    hours_before_due: float | None = Field(
        None, description="due date minus submission time, in hours (negative = late)"
    )


class AssignmentEvent(BaseModel):
    particion: int = Field(..., ge=0, le=59)
    assignment_id: int


class EnrollmentActivity(BaseModel):
    """Everything known about one student in one course up to `day`."""

    user_uuid: str
    course_uuid: str
    day: int = Field(..., ge=1, le=60, description="current course day (1..60)")
    n_courses_in_period: int = Field(1, ge=1, description="courses the student takes this term")
    course_size: int = Field(1, ge=1, description="students enrolled in the course")
    exams: list[Exam] = []
    submissions: list[Submission] = []
    assignments: list[AssignmentEvent] = []

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_uuid": "student-123",
                "course_uuid": "course-abc",
                "day": 45,
                "n_courses_in_period": 6,
                "course_size": 30,
                "exams": [{"exam_code": "P1", "particion": 27, "grade": 4}],
                "submissions": [
                    {
                        "particion": 10,
                        "score": 70,
                        "submission_type": "online_quiz",
                        "hours_before_due": 30,
                    },
                    {
                        "particion": 25,
                        "score": 55,
                        "submission_type": "online_upload",
                        "hours_before_due": 2,
                    },
                ],
                "assignments": [
                    {"particion": 5, "assignment_id": 1},
                    {"particion": 20, "assignment_id": 2},
                    {"particion": 35, "assignment_id": 3},
                ],
            }
        }
    }


class Prediction(BaseModel):
    user_uuid: str
    course_uuid: str
    day: int
    model_day: int = Field(..., description="horizon of the model used (<= day)")
    fail_probability: float
    at_risk: bool
    alert_threshold: float
    predicted_grade: float
    models: dict[str, str]


class BatchRequest(BaseModel):
    enrollments: list[EnrollmentActivity] = Field(..., min_length=1, max_length=5000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
