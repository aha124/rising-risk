"""Write the trained trees as JSON for the page's scorer, plus rows to test it against.

docs/data/model_browser.json  the trees, calibrator, tiers and input descriptions
docs/data/parity.json         200 test rows with the raw score and calibrated
                              probability Python gives, checked by docs/tests/parity.mjs

Thresholds, leaf values and calibrator points are written at full double precision.

Missing values. The page's rule is: a missing value goes left when `d` is true,
otherwise right, and any other value goes left when value <= threshold. LightGBM
has two cases. A split trained on a feature with missing values has missing_type
"NaN" and sends them where default_left says. A split trained on a feature with no
missing values has missing_type "None", and LightGBM then treats a missing value as
0.0 and compares it with the threshold. So `d` is default_left in the first case
and (0 <= threshold) in the second. `check_export` walks the exported trees in
Python with the page's rule and compares with LightGBM on rows with extra nulls.
"""

from __future__ import annotations

import math

import numpy as np

from . import config, dataset
from .scoring import Scored, load_scored

N_PARITY_ROWS = 200
# Inputs offered as controls on the page. A judgment call, checked against the measured
# importance ranks that run() writes to browser_model.json: the four inputs people ask
# about first, then utilization, medication and condition inputs that carry weight in the
# model and that a person can set by hand. Left out on purpose: inputs computed from other
# inputs, labs that are missing for most members (a slider cannot say "not recorded"),
# payer flags, and the months 13 to 24 twin of an input already shown.
INTERACTIVE = (
    "cost_prior_12m",
    "age",
    "sex_female",
    "pregnant_at_index",
    "procedures_12m",
    "outpatient_visits_12m",
    "medication_cost_prior_12m",
    "active_medications",
    "has_chronic_kidney_disease",
    "has_diabetes",
)


def export_tree(node: dict) -> dict:
    if "leaf_value" in node:
        return {"v": node["leaf_value"]}
    if node["decision_type"] != "<=":
        raise ValueError(f"unhandled decision type {node['decision_type']}")
    missing_type = node["missing_type"]
    if missing_type == "NaN":
        d = bool(node["default_left"])
    elif missing_type == "None":
        d = bool(0.0 <= node["threshold"])  # LightGBM compares 0.0 with the threshold here
    else:
        raise ValueError(f"unhandled missing_type {missing_type}")
    return {
        "f": node["split_feature"],
        "t": node["threshold"],
        "d": d,
        "l": export_tree(node["left_child"]),
        "r": export_tree(node["right_child"]),
    }


def tree_value(node: dict, row) -> float:
    """The page's evaluator, in Python."""
    while "v" not in node:
        x = row[node["f"]]
        missing = x is None or (isinstance(x, float) and math.isnan(x))
        node = node["l"] if (node["d"] if missing else x <= node["t"]) else node["r"]
    return node["v"]


def raw_score(trees: list[dict], row, init_score: float = 0.0) -> float:
    s = init_score
    for tree in trees:
        s += tree_value(tree, row)
    return s


def check_export(trees: list[dict], sc: Scored, X: np.ndarray, tol: float = 1e-9) -> float:
    """Largest gap between the exported trees and LightGBM on the given rows."""
    expected = sc.scorer.raw(X)
    got = np.array(
        [raw_score(trees, [None if np.isnan(v) else float(v) for v in row]) for row in X]
    )
    worst = float(np.max(np.abs(got - expected))) if len(X) else 0.0
    if worst > tol:
        raise AssertionError(f"exported trees differ from LightGBM by {worst:.3e}")
    return worst


def with_extra_nulls(X: np.ndarray, rng: np.random.Generator, n_rows: int) -> np.ndarray:
    """Blank one random feature in each of the first n_rows rows, including features
    that are never missing in training."""
    X = X.copy()
    n_rows = min(n_rows, len(X))
    X[np.arange(n_rows), rng.integers(0, X.shape[1], n_rows)] = np.nan
    return X


def derived_inputs(sc: Scored) -> list[dict]:
    """Inputs that are exact functions of other inputs, checked against the data."""
    frame, feats = sc.frame, set(sc.features)
    out = []
    if {"cost_change_12m", "cost_prior_12m", "cost_prior_13_24m"} <= feats:
        gap = frame["cost_change_12m"] - (frame["cost_prior_12m"] - frame["cost_prior_13_24m"])
        if float(gap.abs().max()) < 0.011:
            out.append(
                {
                    "feature": "cost_change_12m",
                    "op": "difference",
                    "of": ["cost_prior_12m", "cost_prior_13_24m"],
                }
            )
    flags = [
        str(n)
        for n, g in zip(sc.dictionary["name"], sc.dictionary["group"], strict=True)
        if g == "conditions" and str(n).startswith("has_")
    ]
    if (
        config.CHRONIC_COUNT in feats
        and flags
        and (frame[flags].sum(axis=1) == frame[config.CHRONIC_COUNT]).all()
    ):
        out.append({"feature": config.CHRONIC_COUNT, "op": "sum", "of": flags})
    return out


def control_range(f: dict) -> tuple[float, float, float, float | None]:
    """min, max, step and default for one input, from its 1st, 50th and 99th percentiles."""
    if f["type"] == "binary":
        default = None if f["median"] is None else float(round(f["median"]))
        return 0.0, 1.0, 1.0, default
    lo, hi, med = f["p01"], f["p99"], f["median"]
    if lo is None or hi is None:
        return 0.0, 1.0, 1.0, None
    if f["unit"] == "USD":
        step = 100.0 if hi <= 20_000 else 500.0 if hi <= 100_000 else 1000.0
        lo = 0.0 if lo >= 0 else math.floor(lo / step) * step
        hi = math.ceil(hi / step) * step
    elif f["dtype"] == "int" or f.get("integer_valued"):
        step, lo, hi = 1.0, math.floor(lo), math.ceil(hi)
    elif f["unit"] == "share":
        step, lo, hi = 0.01, 0.0, 1.0
    else:
        step = 0.1
        lo, hi = math.floor(lo * 10) / 10, math.ceil(hi * 10) / 10
    if hi <= lo:
        hi = lo + step
    # A typical member has no value for an input that is missing more often than not.
    default = None if f["missing_rate"] > 0.5 else min(hi, max(lo, round(med / step) * step))
    digits = 2 if step < 0.1 else 1 if step < 1 else 0
    return (
        float(lo),
        float(hi),
        step,
        None if default is None else round(float(default), digits),
    )


def run() -> dict:
    paths = config.paths()
    sc = load_scored(paths)
    scorer = sc.scorer
    spec = dataset.read_json(paths.model_dir / "feature_spec.json")["features"]
    explain_path = paths.reports_dir / "explain.json"
    if explain_path.exists():
        importance = {
            f["name"]: f["importance"] for f in dataset.read_json(explain_path)["features"]
        }
    else:
        gain = scorer.booster.feature_importance(importance_type="gain")
        importance = dict(zip(sc.features, gain.astype(float), strict=True))

    dump = scorer.booster.dump_model()
    trees = [export_tree(t["tree_structure"]) for t in dump["tree_info"]]
    n_leaves = int(sum(t["num_leaves"] for t in dump["tree_info"]))

    derived = derived_inputs(sc)
    derived_names = {d["feature"] for d in derived}
    interactive = {n for n in INTERACTIVE if n in sc.features and n not in derived_names}
    rank = {n: i + 1 for i, n in enumerate(sorted(importance, key=lambda n: -importance[n]))}
    features = []
    defaults = {}
    for f in spec:
        lo, hi, step, default = control_range(f)
        defaults[f["name"]] = default
        features.append(
            {
                "name": f["name"],
                "label": f["label"],
                "group": f["group"],
                "unit": "" if f["type"] == "binary" else f["unit"],
                "type": f["type"],
                "min": lo,
                "max": hi,
                "step": step,
                "default": default,
                "interactive": f["name"] in interactive,
                "nullable": bool(f["nullable"]),
            }
        )
    # Defaults of derived inputs follow from the defaults of their parts.
    for d in derived:
        parts = [defaults[n] or 0.0 for n in d["of"]]
        value = parts[0] - parts[1] if d["op"] == "difference" else float(sum(parts))
        next(f for f in features if f["name"] == d["feature"])["default"] = value

    test = sc.test
    rng = np.random.default_rng(config.SEED)
    take = rng.choice(len(test), size=min(N_PARITY_ROWS, len(test)), replace=False)
    X = with_extra_nulls(sc.X(test.iloc[take]), rng, 20)
    raw = scorer.raw(X)
    p = scorer.calibrate(raw)
    worst = check_export(trees, sc, X)
    # A wider check than the 200 published rows: 2,000 rows, 500 of them with an extra null.
    wide = rng.choice(len(test), size=min(2000, len(test)), replace=False)
    worst = max(worst, check_export(trees, sc, with_extra_nulls(sc.X(test.iloc[wide]), rng, 500)))

    model = {
        "fixture": False,
        "model_version": scorer.metadata["model_version"],
        "n_trees": len(trees),
        "n_leaves": n_leaves,
        "init_score": 0.0,
        "init_score_note": "LightGBM folds the starting score into the first tree, so the raw "
        "score is the plain sum of leaf values.",
        "missing_rule": "A null or NaN input goes left when d is true, otherwise right. Any "
        "other value goes left when value <= t.",
        "features": features,
        "derived": derived,
        "interactive_note": "Inputs without a control stay at their training median. Some "
        "combinations a visitor can set (a high yearly cost with a small largest encounter) "
        "are ones the model never saw in training.",
        "trees": trees,
        "calibrator": {"x": scorer.calib_x.tolist(), "y": scorer.calib_y.tolist()},
        "calibrator_input": "probability",
        "calibrator_note": "The calibrator takes the sigmoid of the raw score. Interpolate "
        "linearly between points and clip outside them.",
        "tiers": {"high": scorer.tiers["high"], "rising": scorer.tiers["rising"]},
        "score_percentiles": scorer.score_percentiles.tolist(),
        "score_percentiles_source": f"calibrated probability, test years "
        f"{list(sc.plan.test_years)}, percentiles 0 to 100",
    }
    parity = {
        "fixture": False,
        "note": "Test rows in model feature order. The first 20 have one extra null in a "
        "random feature, including features that are never missing in training.",
        "rows": [[None if np.isnan(v) else float(v) for v in row] for row in X],
        "raw": raw.tolist(),
        "p": p.tolist(),
    }
    dataset.write_json(paths.site_data_dir / "model_browser.json", model, compact=True)
    dataset.write_json(paths.site_data_dir / "parity.json", parity, compact=True)
    size = (paths.site_data_dir / "model_browser.json").stat().st_size
    if size > config.MAX_BROWSER_MODEL_BYTES:
        raise AssertionError(
            f"model_browser.json is {size:,} bytes, over the {config.MAX_BROWSER_MODEL_BYTES:,} "
            "limit. Lower the tree or leaf cap in the search space and retrain."
        )
    summary = {
        "n_trees": len(trees),
        "n_leaves": n_leaves,
        "bytes": int(size),
        "max_bytes": config.MAX_BROWSER_MODEL_BYTES,
        "largest_gap_vs_lightgbm": worst,
        "interactive": [
            {"feature": f["name"], "importance_rank": rank.get(f["name"])}
            for f in features
            if f["interactive"]
        ],
        "derived": derived,
        "rows_with_nulls": int(sum(any(v is None for v in row) for row in parity["rows"])),
    }
    dataset.write_json(paths.reports_dir / "browser_model.json", summary)
    print(
        f"browser model: {len(trees)} trees, {n_leaves} leaves, {size:,} bytes, "
        f"largest gap vs LightGBM {worst:.1e}"
    )
    return summary
