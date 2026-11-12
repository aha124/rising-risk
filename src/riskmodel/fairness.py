"""Subgroup results and the cost-proxy check.

Race and ethnicity are never model inputs. They are read here, after scoring, and
nowhere else. Sex and age are model inputs, and their rows show what that does to
who gets selected.

Selection is the pooled top 10% of the test set by raw score, the same list the
headline uses. A group's recall is the share of its high-cost members on that list.

The cost-proxy check follows Obermeyer et al. (Science, 2019). The model predicts
cost. If one group is sicker than another at the same score, cost is standing in
poorly for need in that group. Synthetic data can show whether the pipeline would
detect that. It cannot show whether it happens in a real population.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, dataset
from . import metrics as M
from .scoring import load_scored

AGE_BANDS = (
    ("18 to 34", 18, 34),
    ("35 to 49", 35, 49),
    ("50 to 64", 50, 64),
    ("65 to 79", 65, 79),
    ("80 and over", 80, 200),
)
SMALL_RULE = (
    f"Groups with fewer than {config.SMALL_GROUP_ROWS} test rows or fewer than "
    f"{config.SMALL_GROUP_POSITIVES} high-cost members are marked small. Their numbers are "
    "shown for completeness and are too noisy to read."
)


def group_masks(test: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    age = test[config.AGE].to_numpy(dtype=float)
    female = test[config.SEX_FEMALE].to_numpy(dtype=float) == 1
    cuts = [("sex", "Female", female), ("sex", "Male", ~female)]
    cuts += [("age band", name, (age >= lo) & (age <= hi)) for name, lo, hi in AGE_BANDS]
    for attribute in config.AUDIT_COLUMNS:
        values = test[attribute].astype(str).to_numpy()
        levels = pd.Series(values).value_counts().index.tolist()  # largest first
        cuts += [(attribute, level, values == level) for level in levels]
    return cuts


def _ratio_interval(num: np.ndarray, den: np.ndarray, W: np.ndarray, point: float) -> dict:
    """Interval for sum(w * num) / sum(w * den) over member resamples."""
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = (W @ num) / (W @ den)
    return M.interval(point, samples)


def subgroup_table(test: pd.DataFrame, W: np.ndarray) -> list[dict]:
    y = test[config.LABEL].to_numpy(dtype=float)
    p = test["p"].to_numpy()
    tie = test["tie_key"].to_numpy()
    selected = M.top_k_mask(test["raw"].to_numpy(), tie, config.HEADLINE_K).astype(float)
    selected_prior = M.top_k_mask(test["prior_cost"].to_numpy(), tie, config.HEADLINE_K).astype(
        float
    )
    rows = []
    for attribute, level, mask in group_masks(test):
        g = mask.astype(float)
        n, n_pos = int(mask.sum()), int(y[mask].sum())
        if n == 0:
            continue
        has_both = 0 < n_pos < n
        recall = float((selected * y * g).sum() / n_pos) if n_pos else None
        oe = float(n_pos / p[mask].sum()) if p[mask].sum() > 0 else None
        row = {
            "attribute": attribute,
            "level": level,
            "n": n,
            "n_pos": n_pos,
            "prevalence": float(n_pos / n),
            "auroc": float(roc_auc_score(y[mask], test["raw"].to_numpy()[mask]))
            if has_both
            else None,
            "obs_over_expected": oe,
            "selection_rate_at_10": float(selected[mask].mean()),
            "recall_at_10": recall,
            "precision_at_10": float((selected * y * g).sum() / max(1.0, (selected * g).sum())),
            "prior_cost_selection_rate_at_10": float(selected_prior[mask].mean()),
            "prior_cost_recall_at_10": float((selected_prior * y * g).sum() / n_pos)
            if n_pos
            else None,
            "small": bool(n < config.SMALL_GROUP_ROWS or n_pos < config.SMALL_GROUP_POSITIVES),
        }
        if n_pos:
            row["recall_at_10_ci"] = _ratio_interval(selected * y * g, y * g, W, recall)
            row["obs_over_expected_ci"] = _ratio_interval(y * g, p * g, W, oe)
        rows.append(row)
    return rows


def proxy_audit(test: pd.DataFrame, W: np.ndarray, attribute: str = "race") -> dict:
    """Mean number of chronic condition groups by score decile and race.

    Decile 10 holds the highest scores. The summary number for a group is its mean
    chronic count minus the reference group's, compared within each decile and then
    averaged over deciles with the group's own row counts as weights.
    """
    chronic = test[config.CHRONIC_COUNT].to_numpy(dtype=float)
    age = test[config.AGE].to_numpy(dtype=float)
    values = test[attribute].astype(str).to_numpy()
    pct = M.rank_pct(test["raw"].to_numpy(), test["tie_key"].to_numpy())
    decile = 10 - np.minimum(9, np.floor(pct * 10).astype(int))  # 10 = highest scores
    levels = pd.Series(values).value_counts().index.tolist()
    reference = levels[0]
    y = test[config.LABEL].to_numpy()

    rows = []
    for d in range(1, 11):
        for level in levels:
            m = (decile == d) & (values == level)
            if m.sum() == 0:
                continue
            rows.append(
                {
                    "score_decile": d,
                    "level": level,
                    "mean_chronic_conditions": float(chronic[m].mean()),
                    "mean_age": float(age[m].mean()),
                    "n": int(m.sum()),
                }
            )

    def gap(weights: np.ndarray, level: str, deciles=range(1, 11)) -> np.ndarray:
        """Decile-matched difference in mean chronic count, level minus reference."""
        num = np.zeros(weights.shape[0])
        den = np.zeros(weights.shape[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            for d in deciles:
                g = ((decile == d) & (values == level)).astype(float)
                r = ((decile == d) & (values == reference)).astype(float)
                wg, wr = weights @ g, weights @ r
                diff = (weights @ (g * chronic)) / wg - (weights @ (r * chronic)) / wr
                ok = (wg > 0) & (wr > 0)
                num += np.where(ok, wg * diff, 0.0)
                den += np.where(ok, wg, 0.0)
            return num / den

    ones = np.ones((1, len(test)))
    comparisons = []
    for level in levels[1:]:
        m = values == level
        small = bool(m.sum() < config.SMALL_GROUP_ROWS or y[m].sum() < config.SMALL_GROUP_POSITIVES)
        comparisons.append(
            {
                "level": level,
                "reference": reference,
                "n": int(m.sum()),
                "small": small,
                "chronic_gap_all_deciles": M.interval(gap(ones, level)[0], gap(W, level)),
                "chronic_gap_top_decile": M.interval(
                    gap(ones, level, [10])[0], gap(W, level, [10])
                ),
            }
        )
    return {
        "attribute": attribute,
        "reference_level": reference,
        "rows": rows,
        "comparisons": comparisons,
        "finding": proxy_finding(comparisons, reference, float(chronic.mean())),
    }


def proxy_finding(comparisons: list[dict], reference: str, mean_chronic: float) -> str:
    """Two sentences and a caveat that say only what the comparison table supports.

    A group is described as different only when its interval excludes zero. Small
    groups are counted, not interpreted.
    """
    usable = [
        c for c in comparisons if not c["small"] and c["chronic_gap_all_deciles"]["lo"] is not None
    ]
    caveat = " Synthetic data cannot settle this for a real population."
    skipped = len(comparisons) - len(usable)
    if not usable:
        return (
            "Every race group other than the largest is too small in the test set to compare, "
            "so this check says nothing either way." + caveat
        )

    def name(level):
        return str(level).capitalize()

    def numbers(c):
        iv = c["chronic_gap_all_deciles"]
        return f"{iv['v']:+.2f}, interval {iv['lo']:+.2f} to {iv['hi']:+.2f}"

    apart = [
        c
        for c in usable
        if c["chronic_gap_all_deciles"]["lo"] > 0 or c["chronic_gap_all_deciles"]["hi"] < 0
    ]
    same = [c for c in usable if c not in apart]
    if apart:
        parts = []
        for c in apart:
            more = c["chronic_gap_all_deciles"]["v"] > 0
            parts.append(
                f"{name(c['level'])} members carry {'more' if more else 'fewer'} chronic "
                f"condition groups than {name(reference)} members ({numbers(c)}, against a "
                f"test-set average of {mean_chronic:.1f})"
            )
        first = "At the same risk score, " + "; ".join(parts)
        if any(c["chronic_gap_all_deciles"]["v"] > 0 for c in apart):
            first += ", the direction Obermeyer et al. reported, though the gap is small"
        first += "."
    else:
        first = (
            "At the same risk score, no race group large enough to measure carries a clearly "
            f"different number of chronic condition groups than {name(reference)} members."
        )
    rest = []
    if same:
        rest.append(
            " and ".join(f"{name(c['level'])} members ({numbers(c)})" for c in same)
            + " show no clear gap"
        )
    if skipped:
        rest.append(
            f"{skipped} smaller group{'s are' if skipped != 1 else ' is'} too small to compare"
        )
    second = (" " + ", and ".join(rest) + ".") if rest else ""
    return first + second + caveat


def run() -> dict:
    paths = config.paths()
    s = config.settings()
    sc = load_scored(paths)
    test = sc.test.reset_index(drop=True)
    _, member_pos = np.unique(test["member_id"].to_numpy(), return_inverse=True)
    counts = M.member_resamples(int(member_pos.max()) + 1, s.n_bootstrap, config.SEED)
    W = counts[:, member_pos].astype(np.float32)

    report = {
        "excluded_from_model": list(config.AUDIT_COLUMNS),
        "model_inputs_in_this_table": [config.SEX_FEMALE, config.AGE],
        "small_group_rule": SMALL_RULE,
        "selection_note": "Selection is the pooled top 10% of test rows by raw score. "
        "obs_over_expected is observed high-cost members over the sum of calibrated "
        "probabilities, so 1.0 is perfectly calibrated for the group.",
        "n_bootstrap": s.n_bootstrap,
        "groups": subgroup_table(test, W),
        "proxy_audit": proxy_audit(test, W),
    }
    dataset.write_json(paths.reports_dir / "fairness.json", report)
    print(f"fairness: {len(report['groups'])} groups. {report['proxy_audit']['finding']}")
    return report
