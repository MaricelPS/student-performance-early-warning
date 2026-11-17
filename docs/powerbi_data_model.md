# Power BI data model

`early-warning export-bi` writes a star schema to `data/bi/` (CSV, ready for *Get Data → Text/CSV*). The `.pbix` file is not versioned: it is a binary file and has to be built in Power BI Desktop from these tables.

## Tables

| Table | Grain | Key | Main columns |
|---|---|---|---|
| `fact_enrollment` | student × course | `enrollment_id` | `final_grade`, `passed`, `excluded`, `periodo` |
| `fact_exam` | student × course × exam | (`enrollment_id`, `exam_code`) | `particion`, `exam_weight`, `grade` |
| `fact_submission` | submission | `sub_uuid` | `particion`, `submission_type`, `score`, `hours_before_due` |
| `fact_risk_score` | enrollment × scoring day | (`enrollment_id`, `scoring_day`) | `fail_probability`, `at_risk`, `predicted_grade`, `model_day` |
| `dim_student` | student | `user_uuid` | `legajo` |
| `dim_course` | course | `course_uuid` | `course_name`, `n_students`, `periodo` |
| `dim_day` | course day | `particion` | `day` (1..60), `week` |
| `dim_assignment` | assignment | `assignment_id` | `name`, `points_possible`, `due_at` |

## Relationships (single direction, dimension → fact)

```
dim_student[user_uuid]   1 ── * fact_enrollment[user_uuid]
dim_course[course_uuid]  1 ── * fact_enrollment[course_uuid]
fact_enrollment[enrollment_id] 1 ── * fact_exam[enrollment_id]
fact_enrollment[enrollment_id] 1 ── * fact_submission[enrollment_id]
fact_enrollment[enrollment_id] 1 ── * fact_risk_score[enrollment_id]
dim_day[particion]       1 ── * fact_exam[particion]
dim_day[particion]       1 ── * fact_submission[particion]
```

`fact_enrollment` behaves as a bridge for the three event facts. Filter the report with `fact_enrollment[excluded] = FALSE` (page-level filter).

## Suggested DAX measures

```DAX
Enrollments      = COUNTROWS ( fact_enrollment )
Pass rate        = DIVIDE ( SUM ( fact_enrollment[passed] ), [Enrollments] )
Failure rate     = 1 - [Pass rate]
Avg final grade  = AVERAGE ( fact_enrollment[final_grade] )
Submissions per enrollment = DIVIDE ( COUNTROWS ( fact_submission ), [Enrollments] )
Avg assignment score = AVERAGE ( fact_submission[score] )
Avg exam grade   = AVERAGE ( fact_exam[grade] )
Students at risk =
    CALCULATE ( COUNTROWS ( fact_risk_score ), fact_risk_score[at_risk] = 1 )
Alert precision (historical) =
    DIVIDE (
        CALCULATE ( COUNTROWS ( fact_risk_score ),
                    fact_risk_score[at_risk] = 1,
                    TREATAS ( { 0 }, fact_enrollment[passed] ) ),
        [Students at risk] )
```

## Suggested report pages

1. **Overview:** KPI cards (enrollments, pass rate, average grade), grade distribution, pass rate by course (top/bottom N, with a minimum-size filter).
2. **Engagement:** cumulative submissions by day split by outcome (`dim_day[day]` on the axis), score distribution, submission-type mix.
3. **Exams:** exam timing (box plot by `exam_code`), grade by exam, matrix of partial average vs final grade.
4. **Risk monitor:** slicer on `scoring_day`; table of at-risk enrollments sorted by `fail_probability`, with drill-through to the student's activity timeline.

> **Note:** the `fact_risk_score` table exported here scores the same cohort the models were trained on, so it is an *in-sample demo* of the monitor page. Model quality must be read from the held-out results in `reports/metrics/`, not from this table.
