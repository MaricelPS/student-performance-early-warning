# Model card: student early-warning models

## Purpose

Estimate, on a given course day, the probability that a student **fails a course** (final grade < 5) and their expected final grade, so tutors can prioritize support. This is a decision-support tool for humans: it should trigger a conversation, never an automatic sanction.

## Models

Eight models: two tasks × four horizons (course days 15, 30, 45, 60). A request on day *d* is served by the latest horizon ≤ *d*.

| Horizon | Failure classifier (calibrated) | Grade regressor |
|---|---|---|
| Day 15 | LightGBM (tuned) | LightGBM (tuned) |
| Day 30 | Logistic regression | LightGBM (tuned) |
| Day 45 | Logistic regression | Random forest |
| Day 60 | LightGBM (tuned) | Random forest |

Selection used repeated stratified-grouped CV on the training set with the **one-standard-error rule** (pick the simplest model within 1 SE of the best). Classifiers get sigmoid calibration, and their alert threshold is set on out-of-fold training predictions for **recall ≥ 0.80**. Hyper-parameters are in each model's JSON sidecar (`models/*.json`).

## Data

* 2,964 enrollments (600 students, 582 courses), almost all from semester 1-2022; 5.3% failures.
* 37 point-in-time features in five groups: context, engagement, assignment performance, submission timing and exams (`src/early_warning/features/build.py`).
* Excluded: 5 enrollments with final grade 0 (likely "absent", see `docs/data_quality_decisions.md`).

## Evaluation (held-out 20%, grouped by student; 593 enrollments, 31 failures)

95% CIs from a grouped bootstrap (1,000 resamples of students).

| Horizon | PR-AUC (baseline 0.052) | ROC-AUC | Brier | Recall @ threshold | Precision @ threshold | Flagged |
|---|---|---|---|---|---|---|
| Day 15 | 0.07 [0.04–0.12] | 0.63 [0.53–0.72] | 0.049 | 0.87 | 0.07 | 70% |
| Day 30 | 0.15 [0.08–0.28] | 0.73 [0.66–0.81] | 0.048 | 0.77 | 0.09 | 47% |
| Day 45 | 0.28 [0.16–0.42] | 0.87 [0.81–0.91] | 0.044 | 0.77 | 0.17 | 23% |
| Day 60 | 0.50 [0.32–0.75] | 0.94 [0.90–0.98] | 0.032 | 0.74 | 0.32 | 12% |

| Horizon | MAE (grade points) | RMSE | R² |
|---|---|---|---|
| Day 15 | 1.30 [1.20–1.41] | 1.64 | 0.07 |
| Day 30 | 1.06 [0.96–1.15] | 1.42 | 0.30 |
| Day 45 | 0.74 [0.65–0.83] | 1.14 | 0.55 |
| Day 60 | 0.32 [0.24–0.39] | 0.74 | 0.81 |

## Intended use

* **Day 30:** a broad, low-cost nudge (automated message, tutor check-in) for flagged students.
* **Day 45:** a targeted intervention. This is the best trade-off: about 1 in 4 students flagged, with failures 3× more frequent than baseline among them.
* **Days 15 and 60:** monitoring and analytics only. Day 15 is barely above chance; on day 60 it is too late to act and the model mostly reproduces the grading rule.

## Limitations and risks

* **Small positive class** (156 failures). Estimates have wide CIs, and the ranking of close models is uncertain.
* **One semester.** No out-of-time validation is possible; performance must be re-checked on the next cohort before use.
* **Exams dominate.** Signals that do not depend on exams are weak before the first partial (~day 28).
* **Associations, not causes.** A low assignment score may be a symptom; do not use the features as levers without experiments.
* **Feedback loop.** If interventions work, flagged students will fail less and future labels will be biased. Log interventions and keep a holdout group when retraining.
* **Fairness.** No demographic attributes are available, so disparate impact cannot be audited. Do not join demographic data for prediction; use it only for auditing.

## Maintenance

Retrain every semester (`make pipeline`). Monitor feature drift (PSI), the alert rate and calibration (Brier) per cohort. Artifacts carry the package, sklearn and Python versions; `requirements.lock` pins the versions used for training.
