# Data quality: findings and decisions

The challenge warns that the dataset contains inconsistencies and asks for the decisions taken and why. Every item below is **detected and handled in code** (`src/early_warning/data/clean.py`), and the counts come from the report that the cleaning step writes on every run (`reports/metrics/data_quality_report.md`).

## 1. How the raw file is structured (the root cause of most issues)

`challenge_edMachina.csv` (196,124 rows × 24 columns, `;`-separated) is a **wide, denormalized activity log**:

* Every enrollment (student × course) has exactly **60 rows minimum, one per day** (`particion`), even on days with no activity. 76% of rows carry no activity at all.
* Exam, assignment and submission columns are filled **independently** on the same row. A row can hold assignment "TP1" next to a submission for "API2": in 4,129 rows the assignment and the submission refer to different tasks. Those columns must not be read as one record.

**Decision:** split the log into five normalized tables, each de-duplicated on its own natural key:

| Table | Grain | Key |
|---|---|---|
| `enrollments` | student × course (+ target) | `user_uuid, course_uuid` |
| `exams` | student × course × exam | `user_uuid, course_uuid, exam_code` |
| `assignments` | assignment catalog | `assignment_id` |
| `assignment_events` | when an assignment appears in a student's timeline | `user_uuid, course_uuid, assignment_id` |
| `submissions` | submission | `sub_uuid` |

As the challenge suggests, the analysis is then aggregated per `user_uuid / course_uuid / particion`, and features are computed per enrollment up to a cutoff day.

## 2. Findings and decisions

| # | Finding | Affected | Decision | Why |
|---|---|---:|---|---|
| 1 | Exact duplicate rows | 1,730 | Dropped | Mostly empty day placeholders; they carry no extra information. |
| 2 | `periodo` written `01-2022` besides `1-2022` | 7,730 rows (119 enrollments) | Normalized to `1-2022`; raw label kept in `periodo_raw` | The dictionary defines `S-YYYY`. **However**, those submissions are dated Jan 28 – Mar 2, 2022, earlier than the rest of 1-2022 (Feb 19 – Jul 19). It may be a different cohort (an early or intensive course), so the original label is kept for audit. |
| 3 | `particion` is 0..59; the dictionary says 1..60 | all rows | Kept 0-based; `day = particion + 1` in reports | Every enrollment has exactly 60 partitions, so it is a pure offset; shifting it would add no information. |
| 4 | One student (`user_uuid`) with two `legajo` | 1 | `user_uuid` is the identity key; `legajo` is not used | The platform id is the consistent key; the administrative number may change (e.g. re-enrollment). |
| 5 | Same `course_name` for different `course_uuid` | 17 names | `course_uuid` is the course key; names are display-only | Names look like commission/section labels and are not unique. |
| 6 | Final grade constant within each enrollment? | 0 inconsistencies | Verified; the last partition's value would win otherwise | The dictionary defines it as the *last* final-exam grade. |
| 7 | Final grade = 0 | 5 enrollments | **Excluded from modeling** (configurable) | These students have partial-exam averages of 5–9.5. A 0 is most likely "absent at the final" (an administrative code), not a measured grade. Keeping them would teach the model that good students fail. |
| 8 | Exam rows repeated within the same day | 1,168 | De-duplicated per student × course × exam | Grade and date are identical; the repetition comes from the wide format. |
| 9 | `score > points_possible` (naive row-level check: 32 rows) | — | Not an error in itself | The check compares a submission with an *unrelated* assignment on the same row (see §1). |
| 10 | Assignments with `points_possible = 0` but graded submissions | 3 assignments | `score` used directly as a 0–100 percentage | Scores are on a 0–100 scale everywhere (max = 100); 0 points is an LMS configuration error. |
| 11 | Lock date = due date | all but 5 assignments | No "late submission" feature; the submission margin (hours before due) is used instead | Late submission is practically impossible (0.03%), so the useful signal is how close to the deadline students submit. |
| 12 | Submission rows repeated (same `sub_uuid`) | 2,055 | De-duplicated by `sub_uuid` | All attributes identical. |
| 13 | `s_created_at` later than `s_submitted_at` | 21,929 of 21,938 | **Column dropped** | Almost all share 2022-08-01, a bulk-load/migration timestamp, not a real event time. |
| 14 | Graded before submitted | 5 | Grading timestamp set to missing | Impossible order. |
| 15 | Submissions without score | 326 | Kept as submissions; excluded from score statistics | Not graded yet ≠ scored 0. |
| 16 | Duplicate assignment names within a course | 11 | When linking by name, the latest due date wins | Needed to compute submission timing. |
| 17 | Submissions that cannot be linked to an assignment | 2,430 | Kept; timing features missing | Gradient boosting handles missing values natively; linear models get median imputation plus missing indicators. |
| 18 | Only 9 enrollments in `2-2022` | 9 | Kept | Too few to analyse the second semester separately; noted as a limitation (no out-of-time validation). |

## 3. Target definition

* `failed = 1` if `nota_final_materia < 5`, else 0 (dictionary: ≥ 5 is a pass).
* Unit = enrollment. 2,964 enrollments after exclusion, of which **156 failed (5.3%)**.
* The final grade is also modeled directly (regression, 0–10).

## 4. Leakage controls

* **Point-in-time features:** for a horizon *c*, only events with `particion < c` are used. A test (`tests/test_features.py::test_no_leakage_from_future_events`) injects extreme future events and asserts that no feature changes.
* **Grouped splits:** students take ~5 courses (ICC ≈ 0.47), so train/test and CV folds are grouped by `user_uuid`.
* **No target-derived features:** no course-level pass rate or target encoding (41% of courses have a single student, so it would leak).
