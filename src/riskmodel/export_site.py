"""Write the JSON files the results page reads (docs/data/), from the saved reports.

Every number on the page comes from one of these files, and every one of these
files comes from artifacts/reports/ or from scoring the mart with the saved model.
Values are rounded to 5 significant digits to keep the files small. The exemplar
input rows are left exact, so the page and the API can score the same inputs.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from . import config, dataset
from . import metrics as M
from .scoring import load_scored

MODEL_ORDER = ("lgbm", "logreg", "prior_cost")
HK = f"{config.HEADLINE_K:.2f}"
N_FIELD = 2000


def rounded(obj, digits: int = 5):
    """Round every float in a nested structure to `digits` significant digits."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return 0.0 if obj == 0 else float(f"{obj:.{digits}g}")
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, digits) for v in obj]
    if isinstance(obj, np.generic):
        return rounded(obj.item(), digits)
    raise TypeError(f"cannot write {type(obj)} to JSON")


def _flat(v: float) -> dict:
    return {"v": v, "lo": v, "hi": v}


def build_summary(sc, metrics: dict, profile: dict | None) -> dict:
    frame, test = sc.frame, sc.test
    plan = sc.plan
    thresholds = (
        frame[frame[config.LABEL] == 1].groupby("index_year")[config.COST_NEXT].min().to_dict()
    )
    models = metrics["models"]
    d = metrics["deltas"]["lgbm_minus_prior_cost"]

    rng = np.random.default_rng(config.SEED)
    take = np.sort(rng.choice(len(test), size=min(N_FIELD, len(test)), replace=False))
    sample = test.iloc[take]
    tie = sample["tie_key"].to_numpy()
    n = len(sample)

    def ranks(score):
        r = np.empty(n, dtype=int)
        r[M.rank_order(score, tie)] = np.arange(n)
        return r

    r_model = ranks(sample["raw"].to_numpy())
    r_prior = ranks(sample["prior_cost"].to_numpy())
    r_random = rng.permutation(n)
    field = [
        {
            "id": f"{str(m)[:8]}-{int(yr)}",
            "y": int(y),
            "cost_next": round(float(c)),
            "p": float(f"{float(p):.4g}"),
            "rank_model": int(a),
            "rank_prior": int(b),
            "rank_random": int(r),
        }
        for m, yr, y, c, p, a, b, r in zip(
            sample["member_id"],
            sample["index_year"],
            sample[config.LABEL],
            sample[config.COST_NEXT],
            sample["p"],
            r_model,
            r_prior,
            r_random,
            strict=True,
        )
    ]
    band = max(1, int(np.round(config.HEADLINE_K * n)))
    y_s = sample[config.LABEL].to_numpy()

    return {
        "fixture": False,
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": dataset.git_sha(),
        "model_version": sc.scorer.metadata["model_version"],
        "data": {
            "n_patients": int(profile["n_patients"])
            if profile
            else int(frame["member_id"].nunique()),
            "n_members": int(frame["member_id"].nunique()),
            "n_member_years": len(frame),
            "n_features": len(sc.features),
            "index_years": sorted(int(y) for y in frame["index_year"].unique()),
            "train_years": list(plan.train_years),
            "tune_year": plan.tune_year,
            "calib_year": plan.calib_year,
            "test_years": list(plan.test_years),
            "n_test": len(test),
            "n_test_members": int(test["member_id"].nunique()),
            "n_test_pos": int(test[config.LABEL].sum()),
            "prevalence_test": float(test[config.LABEL].mean()),
            "member_holdout_share": config.HOLDOUT_SHARE,
            "member_holdout_share_actual": float(
                frame.groupby("member_id")["holdout"].first().mean()
            ),
            "high_cost_threshold_usd_by_year": {
                str(int(y)): float(round(v, 2)) for y, v in thresholds.items()
            },
            "n_bootstrap": int(metrics["n_bootstrap"]),
        },
        "headline": {
            "k": config.HEADLINE_K,
            "recall_at_k": {
                **{k: models[k]["at"][HK]["recall"] for k in MODEL_ORDER},
                "random": config.HEADLINE_K,
            },
            "dollars_at_k": {
                **{k: models[k]["at"][HK]["dollars"] for k in MODEL_ORDER},
                "random": config.HEADLINE_K,
            },
            "auroc": {k: models[k]["auroc"] for k in MODEL_ORDER},
            "auprc": {k: models[k]["auprc"] for k in MODEL_ORDER},
            "brier": models["lgbm"]["brier"]["v"],
            "ece": models["lgbm"]["ece"]["v"],
            "delta_recall_at_k_vs_prior": d[f"recall@{HK}"],
            "delta_auprc_vs_prior": d["auprc"],
            "delta_dollars_at_k_vs_prior": d[f"dollars@{HK}"],
            "delta_auroc_vs_prior": d["auroc"],
        },
        # The page counts from this sample, so its counts will differ a little from the
        # full test set. These are the sample's own numbers for the top 10% band.
        "members_field_note": {
            "n": n,
            "band": band,
            "n_high_cost": int(y_s.sum()),
            "in_band_model": int(y_s[r_model < band].sum()),
            "in_band_prior": int(y_s[r_prior < band].sum()),
            "in_band_random": int(y_s[r_random < band].sum()),
            "how": "simple random sample of test rows, fixed seed",
        },
        "members_field": field,
    }


def build_performance(metrics: dict, capacity: dict, sanity: dict) -> dict:
    out_models = []
    for key in MODEL_ORDER:
        m = metrics["models"][key]
        out_models.append(
            {
                "key": key,
                "label": m["label"],
                "metrics": {
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "brier": m["brier"],
                    "ece": m["ece"],
                    "at": m["at"],
                },
                "roc": [[float(f"{a:.4g}"), float(f"{b:.4g}")] for a, b in m["roc"]],
                "pr": [[float(f"{a:.4g}"), float(f"{b:.4g}")] for a, b in m["pr"]],
                "calibration": m["calibration"],
            }
        )
    rnd = metrics["random"]
    out_models.append(
        {
            "key": "random",
            "label": rnd["label"],
            "metrics": {
                "auroc": _flat(rnd["auroc"]),
                "auprc": _flat(rnd["auprc"]),
                "brier": None,
                "ece": None,
                "at": {
                    k: {name: _flat(v) for name, v in row.items()} for k, row in rnd["at"].items()
                },
            },
            "roc": [[0, 0], [1, 1]],
            "pr": [[0, rnd["auprc"]], [1, rnd["auprc"]]],
            "calibration": None,
        }
    )
    cba = metrics["calibration_before_after"]

    def cal(block):
        return {
            "brier": block["brier"]["v"],
            "ece": block["ece"]["v"],
            "bins": block["bins"],
            "brier_ci": block["brier"],
            "ece_ci": block["ece"],
            "slope": block["calibration_slope"],
            "intercept": block["calibration_intercept"],
        }

    npc = sanity["no_prior_cost_features"]
    return {
        "fixture": False,
        "models": out_models,
        "deltas_vs_prior_cost": {
            "lgbm": metrics["deltas"]["lgbm_minus_prior_cost"],
            "logreg": metrics["deltas"]["logreg_minus_prior_cost"],
        },
        "calibration_before_after": {
            "note": cba["note"],
            "raw": cal(cba["raw"]),
            "calibrated": cal(cba["calibrated"]),
            "change": cba["change"],
        },
        "capacity": [
            {
                "pct": row["pct"],
                "n_contacted": row["n_contacted"],
                **{
                    k: {x: row[k][x] for x in ("found", "dollars", "precision")}
                    for k in ("lgbm", "prior_cost", "logreg", "random")
                },
            }
            for row in capacity["capacity"]
        ],
        "by_year": metrics["by_year"],
        "member_holdout": metrics["member_holdout"],
        "within_year_ranking_recall_at_10": metrics["within_year_ranking_recall_at_10"],
        "sanity": {
            "shuffled_label_auroc": sanity["shuffled_labels"]["test_auroc"],
            "no_prior_cost_features": {
                "auroc": npc["auroc"],
                "auprc": npc["auprc"],
                "recall_at_10": npc["recall_at_10"],
                "recall_at_10_ci": npc["recall_at_10_ci"],
                "dropped": npc["dropped"],
            },
            "max_single_feature_auroc": sanity["single_feature_auroc"]["max"],
            "all_pass": sanity["all_pass"],
        },
        "sensitivity": metrics["sensitivity"],
    }


def build_explain(report: dict) -> dict:
    out = rounded({k: v for k, v in report.items() if k != "note"})
    for slim, full in zip(out["exemplars"], report["exemplars"], strict=True):
        slim["row"] = full["row"]  # exact inputs, never rounded
        slim["percentile"] = float(f"{full['percentile']:.4g}")
        slim["cost_next"] = round(full["cost_next"])
    return {"fixture": False, **out}


def build_drift(report: dict) -> dict:
    keep = (
        "reference_years",
        "reference_note",
        "score_note",
        "thresholds",
        "label_lag_note",
        "labels_mature_as_of",
        "by_year",
        "chart_features",
        "real_shifts",
        "feature_psi_latest",
        "performance_by_year",
    )
    out = {"fixture": False, **{k: report[k] for k in keep}}
    sims = report.get("simulations") or []
    out["simulation"] = sims[0] if sims else None
    out["other_simulations"] = [
        {
            k: v
            for k, v in s.items()
            if k not in ("score_hist_before", "score_hist_after", "bin_edges")
        }
        for s in sims[1:]
    ]
    return out


def run() -> dict:
    paths = config.paths()
    reports = paths.reports_dir
    sc = load_scored(paths)
    metrics = dataset.read_json(reports / "metrics.json")
    sanity = dataset.read_json(reports / "sanity.json")
    capacity = dataset.read_json(reports / "capacity.json")
    profile_path = reports / "data_profile.json"
    profile = dataset.read_json(profile_path) if profile_path.exists() else None

    files = {
        "summary.json": build_summary(sc, metrics, profile),
        "performance.json": build_performance(metrics, capacity, sanity),
        "explain.json": build_explain(dataset.read_json(reports / "explain.json")),
        "fairness.json": {"fixture": False, **dataset.read_json(reports / "fairness.json")},
        "drift.json": build_drift(dataset.read_json(reports / "drift.json")),
    }
    pipeline_path = reports / "pipeline.json"
    if pipeline_path.exists():
        # Written by the warehouse export. Passed through unchanged apart from the flag.
        files["pipeline.json"] = {**dataset.read_json(pipeline_path), "fixture": False}

    sizes = {}
    for name, obj in files.items():
        if name == "explain.json":
            payload = obj  # already rounded, with exact rows
        elif name == "pipeline.json":
            payload = obj
        else:
            payload = rounded(obj)
        dataset.write_json(paths.site_data_dir / name, payload, compact=True)
        sizes[name] = (paths.site_data_dir / name).stat().st_size
    total = sum(sizes.values())
    print(
        "site data: "
        + ", ".join(f"{k} {v / 1024:.0f} KB" for k, v in sizes.items())
        + f". Total {total / 1024:.0f} KB (limit 600 KB, model_browser.json not counted)"
    )
    if total > 600_000:
        raise AssertionError(f"site data is {total:,} bytes, over the 600 KB budget")
    return sizes
