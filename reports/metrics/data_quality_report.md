| Check | Affected | Decision |
|---|---:|---|
| Exact duplicate rows | 1,730 | Dropped. They carry no extra information (mostly empty day-placeholder rows). |
| `periodo` written as '01-2022' instead of '1-2022' | 7,730 | Normalized to the dictionary format 'S-YYYY'. Raw label kept in `periodo_raw`. Note: these enrollments ran Jan-Mar 2022, earlier than the bulk of 1-2022. |
| `particion` ranges 0..59, dictionary says 1..60 | 194,394 | Kept as a 0-based day index (every enrollment has exactly 60 partitions). `day = particion + 1` is used in reports. |
| Students (`user_uuid`) with more than one `legajo` | 1 | `user_uuid` is the identity key; `legajo` is an administrative attribute and is not used for joins or modeling. |
| `course_name` shared by several `course_uuid` | 17 | `course_uuid` is the course key; names are display-only. |
| Enrollments with more than one final grade | 0 | None expected; if present, the last partition's grade is kept (dictionary: 'última nota del examen final'). |
| Final grade = 0 | 5 | Treated as 'absent at the final exam' (these students have partial-exam averages of 5-9.5). Excluded from modeling (ambiguous label). |
| Exam rows repeated across activity rows of the same day | 1,168 | De-duplicated to one row per student-course-exam (grade and day were identical). |
| Assignments with `points_possible` = 0 but graded submissions | 3 | Configuration error in the LMS. Scores are on a 0-100 scale everywhere, so `score` is used directly as a percentage instead of `score / points_possible`. |
| Assignments whose lock date differs from due date | 5 | Lock = due for almost all assignments, so 'late submission' is not a usable signal; submission margin (hours before due) is used instead. |
| Submission rows repeated (same `sub_uuid`) | 2,055 | De-duplicated by `sub_uuid` (all attributes were identical). |
| Submission `s_created_at` later than `s_submitted_at` | 21,929 | ~all share 2022-08-01 (a bulk load/migration timestamp). Column dropped. |
| Submissions graded before being submitted | 5 | Grading timestamp set to missing (impossible order). |
| Submissions without score (not graded) | 326 | Kept: they count as submissions; excluded from score statistics. |
| Scores outside 0..100 | 0 | Clipped to 0..100. |
| Assignment names duplicated within a course | 11 | When linking submissions to assignments by name, the one with the latest due date is used. |
| Submissions that could not be linked to an assignment | 2,430 | Kept; their timing features are missing (handled by the models). |
