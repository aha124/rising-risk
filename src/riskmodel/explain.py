"""Explanations from LightGBM's own TreeSHAP (pred_contrib=True), in log-odds.

Three outputs, all on the test years: global importance (mean absolute
contribution), dependence curves for the top 8 features, and 12 example members.

The examples are picked by rule, not by looking for good stories. Members are put
into groups by what happened (flagged and high cost, flagged and not, missed,
routine), each group is sorted by raw score, and the members at fixed positions in
that order are taken:

  flagged_right (4)  two the prior-cost rule would have missed (outside its top 10%):
                     the median-score member among women 18 to 49, and the median-score
                     member among everyone else, because those are the two kinds of
                     member the model adds. Then two the rule would also have called,
                     at one third and two thirds of the way down that list.
  flagged_wrong (2)  in the model's top 10%, not high cost: one third and two thirds.
  missed (2)         high cost, outside the model's top 10%: the median-score member
                     among those the prior-cost rule did catch, and the median-score
                     member among those neither approach caught.
  routine (4)        not flagged, not high cost: 20%, 40%, 60% and 80% of the way down.

If a group is too small to fill its slots the remaining picks come from evenly
spaced positions in the wider group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, dataset
from . import metrics as M
from .scoring import Scored, load_scored

TOP_DEPENDENCE = 8
TOP_CONTRIBS = 8
FACT_FEATURES = (
    "cost_prior_12m",
    "cost_prior_13_24m",
    "chronic_group_count",
    "inpatient_admits_12m",
    "outpatient_visits_12m",
    "active_medications",
    "pregnant_at_index",
)


def display_value(value, unit: str, dtype: str) -> str:
    """A feature value the way a person would write it."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not recorded"
    v = float(value)
    if dtype == "binary":
        return "yes" if v == 1 else "no"
    if unit == "USD":
        return f"${v:,.0f}" if v >= 0 else f"-${-v:,.0f}"
    if unit == "share":
        return f"{v * 100:.0f}%"
    if dtype == "int" or v == round(v):
        return f"{v:,.0f}"
    return f"{v:.1f}"


def global_importance(sc: Scored, contrib: np.ndarray) -> list[dict]:
    imp = np.abs(contrib[:, :-1]).mean(axis=0)
    order = np.argsort(-imp, kind="stable")
    d = sc.dictionary.set_index("name")
    return [
        {
            "name": sc.features[j],
            "label": str(d.loc[sc.features[j], "label"]),
            "group": str(d.loc[sc.features[j], "group"]),
            "description": str(d.loc[sc.features[j], "description"]),
            "unit": str(d.loc[sc.features[j], "unit"]),
            "importance": float(imp[j]),
            "rank": rank + 1,
        }
        for rank, j in enumerate(order)
    ]


def dependence_curve(x: np.ndarray, c: np.ndarray, max_points: int = 12) -> tuple[list, dict]:
    """Mean contribution by feature value: one point per value for flags and small
    counts, otherwise one point per quantile bin placed at the bin's median value.
    Missing values are summarized separately."""
    ok = ~np.isnan(x)
    points = []
    distinct = np.unique(x[ok])
    if len(distinct) <= max_points:
        for u in distinct:
            m = ok & (x == u)
            if m.sum() >= 20:
                points.append(
                    {"x": float(u), "mean_contrib": float(c[m].mean()), "n": int(m.sum())}
                )
    elif ok.any():
        cuts = np.unique(np.quantile(x[ok], np.linspace(0, 1, max_points + 1)[1:-1]))
        which = np.searchsorted(cuts, x[ok], side="right")
        xs, cs = x[ok], c[ok]
        for b in np.unique(which):
            m = which == b
            points.append(
                {
                    "x": float(np.median(xs[m])),
                    "mean_contrib": float(cs[m].mean()),
                    "n": int(m.sum()),
                }
            )
    missing = {"mean_contrib": float(c[~ok].mean()), "n": int((~ok).sum())} if (~ok).any() else None
    return points, missing


def _positions(order: np.ndarray, fractions) -> list[int]:
    if len(order) == 0:
        return []
    picks = []
    for f in fractions:
        i = int(order[min(len(order) - 1, int(f * len(order)))])
        if i not in picks:
            picks.append(i)
    return picks


def pick_exemplars(test: pd.DataFrame) -> list[tuple[int, str]]:
    """Row positions and kinds for the example members. See the module docstring."""
    y = test[config.LABEL].to_numpy() == 1
    raw = test["raw"].to_numpy()
    tie = test["tie_key"].to_numpy()
    in_model = M.top_k_mask(raw, tie, config.HEADLINE_K)
    in_prior = M.top_k_mask(test["prior_cost"].to_numpy(), tie, config.HEADLINE_K)
    women = (test[config.SEX_FEMALE].to_numpy(dtype=float) == 1) & (
        test[config.AGE].to_numpy(dtype=float) <= config.CHILDBEARING_MAX_AGE
    )
    by_score = M.rank_order(raw, tie)

    def ordered(mask):
        return by_score[mask[by_score]]

    chosen: list[tuple[int, str]] = []
    used: set[int] = set()

    def take(kind, want, primary: list[int], fallback_mask):
        got = [i for i in primary if i not in used][:want]
        if len(got) < want:
            pool = [int(i) for i in ordered(fallback_mask) if i not in used and i not in got]
            need = want - len(got)
            step = max(1, len(pool) // (need + 1))
            got += pool[step - 1 :: step][:need] if pool else []
        for i in got:
            used.add(i)
            chosen.append((i, kind))

    rising = y & in_model & ~in_prior
    take(
        "flagged_right",
        2,
        _positions(ordered(rising & women), [0.5]) + _positions(ordered(rising & ~women), [0.5]),
        rising,
    )
    take(
        "flagged_right",
        2,
        _positions(ordered(y & in_model & in_prior), [1 / 3, 2 / 3]),
        y & in_model,
    )
    take("flagged_wrong", 2, _positions(ordered(~y & in_model), [1 / 3, 2 / 3]), ~y & in_model)
    missed = y & ~in_model
    take(
        "missed",
        2,
        _positions(ordered(missed & in_prior), [0.5])
        + _positions(ordered(missed & ~in_prior), [0.5]),
        missed,
    )
    routine = ~y & ~in_model
    take("routine", 4, _positions(ordered(routine), [0.2, 0.4, 0.6, 0.8]), routine)
    return chosen


def build_exemplars(sc: Scored, test: pd.DataFrame, contrib: np.ndarray) -> list[dict]:
    d = sc.dictionary.set_index("name")
    feats = sc.features
    X = sc.X(test)
    raw = test["raw"].to_numpy()
    tie = test["tie_key"].to_numpy()
    pct_model = M.rank_pct(raw, tie)
    pct_prior = M.rank_pct(test["prior_cost"].to_numpy(), tie)

    def shown(j, i):
        return display_value(X[i, j], str(d.loc[feats[j], "unit"]), str(d.loc[feats[j], "dtype"]))

    out = []
    for num, (i, kind) in enumerate(pick_exemplars(test)):
        row = test.iloc[i]
        c = contrib[i, :-1]
        top = np.argsort(-np.abs(c), kind="stable")[:TOP_CONTRIBS]
        age = int(row[config.AGE])
        who = "woman" if row[config.SEX_FEMALE] == 1 else "man"
        lead = next((j for j in top if feats[j] not in (config.AGE, config.SEX_FEMALE)), top[0])
        lead_label = str(d.loc[feats[lead], "label"])
        lead_text = f"{lead_label[0].lower()}{lead_label[1:]}: {shown(lead, i)}"
        facts = [
            {"label": "Age", "value": str(age)},
            {"label": "Sex", "value": "Female" if row[config.SEX_FEMALE] == 1 else "Male"},
        ]
        for name in FACT_FEATURES:
            if name in feats:
                j = feats.index(name)
                facts.append({"label": str(d.loc[name, "label"]), "value": shown(j, i)})
        p = float(row["p"])
        out.append(
            {
                "id": f"{str(row['member_id'])[:8]}-{int(row['index_year'])}",
                "kind": kind,
                "title": f"Member {chr(65 + num)}, {age}-year-old {who}, {lead_text}",
                "index_year": int(row["index_year"]),
                "facts": facts,
                "p": p,
                "raw": float(raw[i]),
                "percentile": float((1 - pct_model[i]) * 100),
                "tier": sc.scorer.tier(p),
                "y": int(row[config.LABEL]),
                "cost_next": float(row[config.COST_NEXT]),
                "rank_prior_pct": float(pct_prior[i]),
                "in_model_top_10": bool(pct_model[i] < config.HEADLINE_K),
                "in_prior_top_10": bool(pct_prior[i] < config.HEADLINE_K),
                "contribs": [
                    {
                        "feature": feats[j],
                        "label": str(d.loc[feats[j], "label"]),
                        "value_display": shown(j, i),
                        "contrib": float(c[j]),
                    }
                    for j in top
                ],
                "other_contrib": float(c.sum() - c[top].sum()),
                "row": [None if np.isnan(v) else float(v) for v in X[i]],
            }
        )
    return out


def run() -> dict:
    paths = config.paths()
    sc = load_scored(paths)
    test = sc.test.reset_index(drop=True)
    X = sc.X(test)
    contrib = sc.scorer.contributions(X)
    if not np.allclose(contrib.sum(axis=1), test["raw"].to_numpy(), atol=1e-6):
        raise AssertionError("contributions do not add up to the raw score")

    features = global_importance(sc, contrib)
    d = sc.dictionary.set_index("name")
    dependence = []
    for f in features[:TOP_DEPENDENCE]:
        j = sc.features.index(f["name"])
        points, missing = dependence_curve(X[:, j], contrib[:, j])
        dependence.append(
            {
                "feature": f["name"],
                "label": f["label"],
                "unit": str(d.loc[f["name"], "unit"]),
                "points": points,
                "missing": missing,
            }
        )
    report = {
        "note": "Contributions are TreeSHAP values in log-odds from LightGBM pred_contrib, "
        "computed on the test years. base_value plus all contributions equals the raw score.",
        "base_value": float(contrib[0, -1]),
        "features": features,
        "dependence": dependence,
        "exemplars": build_exemplars(sc, test, contrib),
    }
    dataset.write_json(paths.reports_dir / "explain.json", report)
    print("explain: top features " + ", ".join(f["name"] for f in features[:5]))
    return report
