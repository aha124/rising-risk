"""Compare the inputs and scores of a current population against a reference.

    .venv/bin/python monitoring/run_drift.py \\
        --reference data/marts/member_year.parquet --reference-years 2015-2021 \\
        --current data/marts/member_year.parquet --current-years 2025 \\
        --output build/drift/2025.json

Both files need the model's feature columns. The year filters are optional and use the
`index_year` column, so one mart file can serve as both sides. Each feature gets a PSI
and a KS statistic from riskmodel.drift, and both populations are scored with the saved
model so the score distribution is compared too.

Exit status: 0 nothing at or above the threshold, 1 at least one alert (or watch, with
--fail-on watch), 2 bad arguments or unreadable input. A scheduler can act on that
without parsing the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from riskmodel import config, dataset, drift
from riskmodel.scoring import Scorer, sigmoid


def parse_years(text: str | None) -> list[int] | None:
    """ "2015-2021", "2024,2025" or "2025"."""
    if not text:
        return None
    years: list[int] = []
    for part in text.split(","):
        if "-" in part:
            first, last = (int(p) for p in part.split("-", 1))
            years.extend(range(first, last + 1))
        else:
            years.append(int(part))
    return years


def load(path: Path, years: list[int] | None, features: list[str], side: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if years is not None:
        if "index_year" not in df.columns:
            raise ValueError(f"{side}: a year filter needs an index_year column")
        df = df[df["index_year"].isin(years)]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{side}: missing feature columns {missing[:5]}")
    if len(df) == 0:
        raise ValueError(f"{side}: no rows after filtering")
    return df


def build_report(args: argparse.Namespace) -> dict:
    scorer = Scorer.load(args.model_dir)
    features = scorer.features
    reference = load(args.reference, parse_years(args.reference_years), features, "reference")
    current = load(args.current, parse_years(args.current_years), features, "current")

    score_rows, score_note = reference, "all reference rows"
    if args.score_reference == "holdout":
        if "member_id" not in reference.columns:
            raise ValueError("--score-reference holdout needs a member_id column")
        held = reference[reference["member_id"].astype(str).map(dataset.is_holdout)]
        if len(held) >= 200:
            score_rows, score_note = held, "reference rows of members held out of training"
    score_reference = sigmoid(scorer.raw(dataset.feature_matrix(score_rows, features)))
    score_current = sigmoid(scorer.raw(dataset.feature_matrix(current, features)))

    report = drift.drift_report(reference, current, features, score_reference, score_current)
    watch_or_worse = report["alerts"] + report["watch"]
    if report.get("score", {}).get("status") == "watch":
        watch_or_worse.append("model score")
    failing = report["alerts"] if args.fail_on == "alert" else watch_or_worse
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": scorer.metadata["model_version"],
        "reference": {"path": str(args.reference), "years": args.reference_years},
        "current": {"path": str(args.current), "years": args.current_years},
        "score_reference": score_note,
        "fail_on": args.fail_on,
        "failed": bool(failing) and args.fail_on != "never",
        **report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, required=True, help="reference parquet")
    parser.add_argument("--current", type=Path, required=True, help="current parquet")
    parser.add_argument("--reference-years", help='for example "2015-2021"')
    parser.add_argument("--current-years", help='for example "2025"')
    parser.add_argument("--output", type=Path, help="where to write the JSON report")
    parser.add_argument("--model-dir", type=Path, default=None, help="default artifacts/model")
    parser.add_argument(
        "--score-reference",
        choices=["holdout", "all"],
        default="holdout",
        help="score drift is measured against held-out members when the reference has "
        "enough of them, because scores on training rows are optimistic",
    )
    parser.add_argument("--fail-on", choices=["alert", "watch", "never"], default="alert")
    args = parser.parse_args(argv)

    try:
        report = build_report(args)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    score = report["score"]
    print(
        f"{report['n_current']:,} current rows against {report['n_reference']:,} reference rows, "
        f"{len(report['features'])} features. Thresholds: watch {config.PSI_WATCH}, "
        f"alert {config.PSI_ALERT}."
    )
    print(f"model score: PSI {score['psi']:.4f}, KS {score['ks']:.4f} ({score['status']})")
    for row in report["features"]:
        if row["status"] != "ok":
            print(f"  {row['status']:<5} {row['feature']:<28} PSI {row['psi']:.3f}")
    if not report["alerts"] and not report["watch"]:
        print("  no feature at watch or alert level")
    if args.output:
        print(f"report: {args.output}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
