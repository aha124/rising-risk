"""Input and score drift: population stability index and Kolmogorov-Smirnov.

`psi`, `ks_statistic` and `compare` have no dependency on the rest of the package,
so the monitoring CLI can call them on any two tables. `run` writes the drift report
for this project: every index year against the training years, then one simulated
upstream change.

Labels lag inputs by a year. The features for index year Y exist on January 1 of Y,
but whether a member ended up high cost is only known 12 months later. A monitor
therefore sees inputs move a year before it can see accuracy move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, dataset
from . import metrics as M
from .scoring import Scored, load_scored, sigmoid

EPS = 1e-4

LABEL_LAG_NOTE = (
    "Labels for an index year are complete only 12 months later, so a live monitor sees "
    "inputs drift about a year before it can see accuracy drop. In this table the latest "
    "year is marked as not mature: its outcomes are in the data because the data is "
    "retrospective, but on the scoring date they would not exist yet."
)

# One-off events in the generator's calendar, from warehouse/README.md.
CALENDAR_SHOCKS = {
    2014: "the generator's chronic pain and opioid modules switch on in calendar 2014, with a "
    "one-off wave of diagnoses and ambulatory visits",
    2021: "about 15,000 vaccination encounters in calendar 2021",
}


def _clean(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def bin_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    """Inner cut points from the reference, missing values ignored.

    With more distinct values than bins these are the reference quantiles (duplicates
    dropped). With `bins` or fewer distinct values (flags, small counts) each value
    gets its own bin, because quantile cuts would collapse into one.
    """
    ref = reference[~np.isnan(reference)]
    if len(ref) == 0:
        return np.array([])
    distinct = np.unique(ref)
    if len(distinct) <= bins:
        return (distinct[:-1] + distinct[1:]) / 2
    cuts = np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(cuts)


def bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of rows in each bin. The last entry is the share that is missing."""
    values = _clean(values)
    missing = np.isnan(values)
    idx = np.searchsorted(edges, values[~missing], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, missing.sum())
    return counts / max(1, len(values))


def psi(reference, current, bins: int = 10, eps: float = EPS) -> float:
    """Population stability index of `current` against `reference`.

    Bins come from the reference (see bin_edges). Missing values form one extra bin,
    so a field that stops arriving shows up as drift. Shares below `eps` are raised
    to `eps` so an empty bin does not give an infinite value.

    PSI = sum over bins of (current share - reference share) * ln(current / reference).
    """
    reference, current = _clean(reference), _clean(current)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = bin_edges(reference, bins)
    a = np.clip(bin_shares(reference, edges), eps, None)
    b = np.clip(bin_shares(current, edges), eps, None)
    return float(np.sum((b - a) * np.log(b / a)))


def ks_statistic(reference, current) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two
    empirical distribution functions. Missing values are dropped."""
    a, b = _clean(reference), _clean(current)
    a, b = np.sort(a[~np.isnan(a)]), np.sort(b[~np.isnan(b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def status(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "ok"
    if value >= config.PSI_ALERT:
        return "alert"
    if value >= config.PSI_WATCH:
        return "watch"
    return "ok"


def compare(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> list[dict]:
    """PSI, KS and status for each column, largest PSI first."""
    rows = []
    for name in columns:
        ref = reference[name].to_numpy(dtype=float, na_value=np.nan)
        cur = current[name].to_numpy(dtype=float, na_value=np.nan)
        value = psi(ref, cur)
        rows.append(
            {
                "feature": name,
                "psi": _num(value),
                "ks": _num(ks_statistic(ref, cur)),
                "status": status(value),
                "missing_rate_reference": float(np.isnan(ref).mean()),
                "missing_rate_current": float(np.isnan(cur).mean()),
            }
        )
    return sorted(rows, key=lambda r: -(r["psi"] or 0))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    score_reference=None,
    score_current=None,
) -> dict:
    """Everything a scheduled check needs: per-feature drift, score drift, alert list."""
    features = compare(reference, current, columns)
    report = {
        "thresholds": {"watch": config.PSI_WATCH, "alert": config.PSI_ALERT},
        "n_reference": len(reference),
        "n_current": len(current),
        "features": features,
        "alerts": [r["feature"] for r in features if r["status"] == "alert"],
        "watch": [r["feature"] for r in features if r["status"] == "watch"],
        "label_lag_note": LABEL_LAG_NOTE,
    }
    if score_reference is not None and score_current is not None:
        value = psi(score_reference, score_current)
        report["score"] = {
            "psi": _num(value),
            "ks": _num(ks_statistic(score_reference, score_current)),
            "status": status(value),
        }
        if report["score"]["status"] == "alert":
            report["alerts"].append("model score")
    return report


def _num(x):
    return float(x) if x is not None and np.isfinite(x) else None


# --------------------------------------------------------------------------- project report


def lookback_years(feature: str, index_year: int) -> list[int]:
    """Calendar years a feature reads for a given index year, from its name."""
    if feature.endswith("13_24m"):
        return [index_year - 2]
    if feature.endswith("_12m"):
        return [index_year - 1]
    return [index_year - 1, index_year - 2]


def known_cause(feature: str, index_year: int) -> str | None:
    for year in lookback_years(feature, index_year):
        if year in CALENDAR_SHOCKS:
            return CALENDAR_SHOCKS[year]
    return None


def simulate_cost_feed_change(rows: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Dollar inputs for the last 12 months arrive multiplied by `factor`.

    This is what a claims feed switching from paid to billed amounts would look like.
    The year-over-year change is recomputed from the altered value. Everything else,
    including the cost in months 13 to 24, is untouched.
    """
    out = rows.copy()
    for name in ("cost_prior_12m", "medication_cost_prior_12m", "max_encounter_cost_12m"):
        if name in out.columns:
            out[name] = out[name].astype(float) * factor
    if {"cost_change_12m", "cost_prior_12m", "cost_prior_13_24m"} <= set(out.columns):
        out["cost_change_12m"] = out["cost_prior_12m"] - out["cost_prior_13_24m"].astype(float)
    return out


def simulate_lab_feed_stops(rows: pd.DataFrame, lab: str, flag: str | None) -> pd.DataFrame:
    """One lab stops being recorded: the value goes missing and its 'measured' flag goes to 0."""
    out = rows.copy()
    out[lab] = np.nan
    if flag and flag in out.columns:
        out[flag] = 0
    return out


def _simulation(
    sc: Scored,
    reference,
    score_reference,
    latest,
    title,
    description,
    changed,
    altered: pd.DataFrame,
) -> dict:
    """Before and after numbers for one simulated change to the latest year's inputs."""
    feats = sc.features
    before = {r["feature"]: r["psi"] for r in compare(reference, latest, feats)}
    after = {r["feature"]: r["psi"] for r in compare(reference, altered, feats)}
    raw_after = sc.scorer.raw(sc.X(altered))
    p_before = latest["p_raw"].to_numpy()
    p_after = sigmoid(raw_after)
    cal_before = latest["p"].to_numpy()
    cal_after = sc.scorer.calibrate(raw_after)
    edges = np.linspace(0, 1, 21)

    def hist(v):
        return (np.histogram(v, edges)[0] / max(1, len(v))).tolist()

    y, cost, tie = (
        latest[config.LABEL].to_numpy(dtype=float),
        latest[config.COST_NEXT].to_numpy(dtype=float),
        latest["tie_key"].to_numpy(),
    )
    key = f"recall@{config.HEADLINE_K:.2f}"
    recall_before = float(M.ranking_metrics(latest["raw"].to_numpy(), y, cost, tie)[key][0])
    recall_after = float(M.ranking_metrics(raw_after, y, cost, tie)[key][0])
    moved = [f for f in feats if abs((after[f] or 0) - (before[f] or 0)) > 1e-12]
    high = sc.scorer.tiers["high"]
    return {
        "title": title,
        "description": description,
        "changed_feature": changed,
        "changed_features": moved,
        "feature_psi_before": before[changed],
        "feature_psi_after": after[changed],
        "score_psi_before": _num(psi(score_reference, p_before)),
        "score_psi_after": _num(psi(score_reference, p_after)),
        "score_ks_before_vs_after": _num(ks_statistic(p_before, p_after)),
        "alerts_fired": [
            f for f in feats if status(after[f]) == "alert" and status(before[f]) != "alert"
        ],
        "watch_fired": [
            f for f in feats if status(after[f]) == "watch" and status(before[f]) == "ok"
        ],
        "mean_probability_before": float(cal_before.mean()),
        "mean_probability_after": float(cal_after.mean()),
        "share_high_tier_before": float((cal_before >= high).mean()),
        "share_high_tier_after": float((cal_after >= high).mean()),
        # Known here only because the data is retrospective. A live monitor could not see it.
        "recall_at_10_before": recall_before,
        "recall_at_10_after": recall_after,
        "score_hist_before": hist(cal_before),
        "score_hist_after": hist(cal_after),
        "bin_edges": edges.tolist(),
    }


def run() -> dict:
    from .evaluate import by_year as performance_by_year

    paths = config.paths()
    sc = load_scored(paths)
    frame, feats = sc.frame, sc.features
    label = dict(zip(sc.dictionary["name"], sc.dictionary["label"], strict=True))
    years = sorted(int(y) for y in frame["index_year"].unique())
    latest_year = years[-1]

    reference = frame[frame["split"] == "train"]
    # Scores on training rows are optimistic, so the score reference is the holdout
    # members in the training years: same period, never seen by the model.
    held = reference[reference["holdout"]]
    score_source = "holdout members in the training years"
    if len(held) < 200:
        held, score_source = reference, "all rows in the training years"
    score_reference = held["p_raw"].to_numpy()

    matrix: dict[int, dict[str, float]] = {}
    by_year = []
    for year in years:
        rows = frame[frame["index_year"] == year]
        matrix[year] = {r["feature"]: r["psi"] for r in compare(reference, rows, feats)}
        by_year.append(
            {
                "year": year,
                "split": sc.plan.split_of(year),
                "in_reference": year in sc.plan.train_years,
                "n": len(rows),
                "score_psi": _num(psi(score_reference, rows["p_raw"].to_numpy())),
                "score_ks": _num(ks_statistic(score_reference, rows["p_raw"].to_numpy())),
                "labels_mature": year < latest_year,
            }
        )

    # The same few features in every year, so they can be drawn as lines: the ones
    # that moved most in any year.
    peak = {f: max((matrix[y][f] or 0) for y in years) for f in feats}
    chart = sorted(feats, key=lambda f: -peak[f])[:5]
    for entry in by_year:
        entry["top_features"] = [
            {"feature": f, "label": label[f], "psi": matrix[entry["year"]][f]} for f in chart
        ]
        entry["n_watch"] = sum(status(matrix[entry["year"]][f]) == "watch" for f in feats)
        entry["n_alert"] = sum(status(matrix[entry["year"]][f]) == "alert" for f in feats)

    shifts = []
    for year in years:
        for f in feats:
            value = matrix[year][f]
            if status(value) != "ok":
                shifts.append(
                    {
                        "year": year,
                        "feature": f,
                        "label": label[f],
                        "psi": value,
                        "status": status(value),
                        "in_reference": year in sc.plan.train_years,
                        "likely_cause": known_cause(f, year),
                    }
                )
    shifts.sort(key=lambda r: -r["psi"])

    latest = frame[frame["index_year"] == latest_year]
    feature_psi_latest = [
        {
            "feature": f,
            "label": label[f],
            "psi": matrix[latest_year][f],
            "status": status(matrix[latest_year][f]),
        }
        for f in sorted(feats, key=lambda f: -(matrix[latest_year][f] or 0))
    ]

    # The first simulation is the one the site shows. The second is kept in the report
    # because its result is useful: a 15% error in the cost feed stays under the alert
    # line for the main cost feature.
    simulations = []
    if "egfr" in feats:
        simulations.append(
            _simulation(
                sc,
                reference,
                score_reference,
                latest,
                "Simulated: the eGFR lab feed stops",
                f"Every eGFR result for index year "
                f"{latest_year} is set to missing and its 'measured' flag to 0, as if the lab "
                "interface went down. Nothing else changes.",
                "egfr",
                simulate_lab_feed_stops(latest, "egfr", "egfr_measured"),
            )
        )
    if config.PRIOR_COST in feats:
        simulations.append(
            _simulation(
                sc,
                reference,
                score_reference,
                latest,
                "Simulated: last year's cost arrives 15% too high",
                f"The dollar inputs for the last 12 "
                f"months of index year {latest_year} are multiplied by 1.15, as if a claims "
                "feed switched from paid to billed amounts. Nothing else changes.",
                config.PRIOR_COST,
                simulate_cost_feed_change(latest, 1.15),
            )
        )

    report = {
        "reference_years": list(sc.plan.train_years),
        "reference_note": "Feature reference: every member-year in the training years. Score "
        f"reference: {score_source}, because scores on rows the model trained on run high.",
        "score_note": "Score drift is measured on the uncalibrated probability. The calibrated "
        "probability has ties, which makes quantile bins unstable.",
        "thresholds": {"watch": config.PSI_WATCH, "alert": config.PSI_ALERT},
        "label_lag_note": LABEL_LAG_NOTE,
        "labels_mature_as_of": f"{latest_year}-01-01",
        "by_year": by_year,
        "chart_features": chart,
        "real_shifts": shifts,
        "feature_psi_latest": feature_psi_latest,
        "feature_psi_by_year": {str(y): matrix[y] for y in years},
        "performance_by_year": [
            {**row, "labels_mature": row["year"] < latest_year} for row in performance_by_year(sc)
        ],
        "simulations": simulations,
    }
    dataset.write_json(paths.reports_dir / "drift.json", report)
    worst = shifts[0] if shifts else None
    print(
        f"drift: {len(shifts)} feature-years at watch or alert"
        + (
            f", largest {worst['feature']} in {worst['year']} at {worst['psi']:.3f}"
            if worst
            else ""
        )
    )
    return report
