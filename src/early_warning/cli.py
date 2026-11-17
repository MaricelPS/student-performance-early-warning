"""Command-line entry point.

early-warning clean       # raw CSV -> normalized parquet tables + data-quality report
early-warning features    # parquet tables -> feature matrix per cutoff
early-warning train       # compare, select, calibrate, evaluate and save models
early-warning score --day 30   # batch risk scores for every enrollment
early-warning export-bi   # star schema CSVs for Power BI
early-warning all         # clean + features + train + score + export-bi
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from early_warning.config import load_config
from early_warning.data.clean import clean, load_tables, save_tables
from early_warning.data.load import load_raw
from early_warning.features.build import build_features, build_training_frame

log = logging.getLogger("early_warning")


def cmd_clean(cfg) -> None:
    raw = load_raw(cfg.path("raw_csv"), sep=cfg.data["csv_separator"])
    tables, report = clean(raw, cfg.data["pass_threshold"], cfg.data["exclude_zero_final_grade"])
    save_tables(tables, cfg.path("interim_dir"))
    cfg.path("metrics_dir").mkdir(parents=True, exist_ok=True)
    (cfg.path("metrics_dir") / "data_quality_report.md").write_text(report.to_markdown() + "\n")
    log.info("raw rows=%s -> %s", len(raw), {k: len(v) for k, v in tables.as_dict().items()})


def _feature_kwargs(cfg) -> dict:
    f = cfg.features
    return {
        "recent_days": f["recent_window_days"],
        "low_score_threshold": f["low_score_threshold"],
        "pass_threshold": cfg.data["pass_threshold"],
    }


def cmd_features(cfg) -> None:
    tables = load_tables(cfg.path("interim_dir"))
    out = cfg.path("processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    for c in cfg.features["cutoffs"]:
        build_training_frame(tables, c, **_feature_kwargs(cfg)).to_parquet(
            out / f"features_day{c:02d}.parquet", index=False
        )
    log.info("feature matrices written to %s", out)


def cmd_train(cfg) -> None:
    from early_warning.models.train import train_all

    tables = load_tables(cfg.path("interim_dir"))
    out = train_all(tables, cfg)
    for r in out["results"]:
        log.info(
            "day=%s task=%s model=%s test=%s",
            r["cutoff"],
            r["task"],
            r["selected_model"],
            {k: round(v["estimate"], 3) for k, v in r["test_metrics"].items()},
        )


def score_all(cfg, day: int) -> pd.DataFrame:
    from early_warning.models.predict import ModelStore

    tables = load_tables(cfg.path("interim_dir"))
    store = ModelStore(cfg.path("models_dir"))
    feats = build_features(tables, day, **_feature_kwargs(cfg))
    clf, reg = store.for_day("classification", day), store.for_day("regression", day)
    p = clf.predict(feats)
    return feats[["user_uuid", "course_uuid"]].assign(
        scoring_day=day,
        model_day=clf.metadata["cutoff"],
        fail_probability=p,
        at_risk=(p >= clf.metadata["threshold"]).astype(int),
        predicted_grade=reg.predict(feats).clip(0, 10),
    )


def cmd_score(cfg, day: int) -> None:
    scores = score_all(cfg, day)
    out = cfg.path("processed_dir") / f"risk_scores_day{day:02d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    log.info(
        "scored %s enrollments, %s flagged at risk -> %s",
        len(scores),
        int(scores["at_risk"].sum()),
        out,
    )


def cmd_export_bi(cfg) -> None:
    from early_warning.bi.export import export_star_schema

    tables = load_tables(cfg.path("interim_dir"))
    risk = None
    try:
        risk = pd.concat([score_all(cfg, d) for d in cfg.features["cutoffs"]])
    except (LookupError, FileNotFoundError, ValueError):
        log.warning("no trained models found: exporting without fact_risk_score")
    paths = export_star_schema(tables, cfg.path("bi_dir"), risk)
    log.info("exported %s tables to %s", len(paths), cfg.path("bi_dir"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="early-warning",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["clean", "features", "train", "export-bi", "all"]:
        sub.add_parser(name)
    sc = sub.add_parser("score")
    sc.add_argument("--day", type=int, required=True, help="course day (1..60)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    t0 = time.perf_counter()
    if args.cmd in ("clean", "all"):
        cmd_clean(cfg)
    if args.cmd in ("features", "all"):
        cmd_features(cfg)
    if args.cmd in ("train", "all"):
        cmd_train(cfg)
    if args.cmd == "score":
        cmd_score(cfg, args.day)
    if args.cmd == "all":
        cmd_score(cfg, max(cfg.features["cutoffs"]))
    if args.cmd in ("export-bi", "all"):
        cmd_export_bi(cfg)
    log.info("done in %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
