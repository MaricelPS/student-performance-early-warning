# Data

The dataset was provided by the EdMachina technical challenge and **is not redistributed in this repository**.

To reproduce the pipeline, place the file as:

```
data/raw/challenge_edMachina.csv     # ';'-separated, 196,124 rows x 24 columns
```

Folders created by the pipeline (all git-ignored):

| Folder | Produced by | Content |
|---|---|---|
| `interim/` | `early-warning clean` | normalized tables (parquet) |
| `processed/` | `early-warning features` / `score` | feature matrices per horizon, risk scores |
| `bi/` | `early-warning export-bi` | star schema CSVs for Power BI |

The trained models in `models/` and the metrics in `reports/` *are* versioned, so the API and the results can be inspected without the raw data.

### Data dictionary (raw file)

| Column | Description |
|---|---|
| `user_uuid` | student id on the platform |
| `course_uuid` | course id |
| `legajo` | administrative student number |
| `particion` | course day, 0..59 in the file (the dictionary says 1..60) |
| `periodo` | semester and year, `S-YYYY` (e.g. `1-2022`) |
| `nota_final_materia` | last final-exam grade (target; ≥ 5 = pass) |
| `nota_parcial` | partial-exam grade |
| `nombre_examen` | exam type, e.g. `PRIMER PARCIAL(20)` (weight in parentheses) |
| `course_name` | course name |
| `fecha_mesa_epoch` | exam date (epoch seconds) |
| `assignment_id`, `ass_name` | assignment id and name |
| `ass_created_at`, `ass_due_at`, `ass_unlock_at`, `ass_lock_at` | assignment dates (epoch) |
| `points_possible` | maximum points of the assignment |
| `ass_name_sub`, `sub_uuid` | name of the submitted assignment, submission id |
| `score` | submission score |
| `submission_type` | submission format (quiz, upload, external tool, …) |
| `s_submitted_at`, `s_graded_at`, `s_created_at` | submission timestamps (epoch) |
