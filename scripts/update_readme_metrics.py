"""Write the measured numbers into README.md and MODEL_CARD.md.

    .venv/bin/python scripts/update_readme_metrics.py           rewrite the blocks
    .venv/bin/python scripts/update_readme_metrics.py --check   exit 1 if a block is stale

Every number in those two documents sits between a pair of markers,

    <!-- name:start -->
    ...
    <!-- name:end -->

and this script fills each block from artifacts/reports/*.json. Nothing between the
markers is typed by hand, so the documents cannot disagree with the reports. CI runs
the --check form.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
DOCUMENTS = ("README.md", "MODEL_CARD.md")
K = "0.10"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


# ------------------------------------------------------------------------------ formatting


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def pct_iv(iv: dict) -> str:
    """56.3% (54.2% to 58.1%)"""
    return f"{pct(iv['v'])} ({pct(iv['lo'])} to {pct(iv['hi'])})"


def points_iv(iv: dict) -> str:
    """Signed difference in percentage points: +7.6 (+6.0 to +9.0)"""
    return f"{iv['v'] * 100:+.1f} ({iv['lo'] * 100:+.1f} to {iv['hi'] * 100:+.1f})"


def num_iv(iv: dict, digits: int = 3) -> str:
    return f"{iv['v']:.{digits}f} ({iv['lo']:.{digits}f} to {iv['hi']:.{digits}f})"


def diff_iv(iv: dict, digits: int = 3) -> str:
    return f"{iv['v']:+.{digits}f} ({iv['lo']:+.{digits}f} to {iv['hi']:+.{digits}f})"


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    rule = ["---" if a == "l" else "---:" for a in align]
    lines = [header, rule, *rows]
    return "\n".join("| " + " | ".join(str(c) for c in line) + " |" for line in lines)


# ---------------------------------------------------------------------------------- blocks


def block_metrics(m: dict, **_) -> str:
    models, test = m["models"], m["test"]
    lgbm, prior = models["lgbm"]["at"][K], models["prior_cost"]["at"][K]
    delta = m["deltas"]["lgbm_minus_prior_cost"][f"recall@{K}"]
    names = [
        ("lgbm", "Gradient boosted trees"),
        ("logreg", "Logistic regression"),
        ("prior_cost", "Last year's cost (the rule to beat)"),
    ]
    rows = [
        [
            label,
            pct_iv(models[key]["at"][K]["recall"]),
            pct_iv(models[key]["at"][K]["dollars"]),
            num_iv(models[key]["auprc"]),
            num_iv(models[key]["auroc"]),
        ]
        for key, label in names
    ]
    prevalence = f"{test['prevalence']:.3f}"
    rows.append(["Random order (expected)", pct(0.10), pct(0.10), prevalence, "0.500"])
    years = " and ".join(str(y) for y in test["years"])
    return "\n\n".join(
        [
            (
                f"Calling the top 10% of members by model score reaches "
                f"**{pct(lgbm['recall']['v'])}** of the members who ended up high cost "
                f"(95% interval {pct(lgbm['recall']['lo'])} to {pct(lgbm['recall']['hi'])}). "
                f"Calling the top 10% by last year's cost reaches "
                f"**{pct(prior['recall']['v'])}** ({pct(prior['recall']['lo'])} to "
                f"{pct(prior['recall']['hi'])}). The paired difference is "
                f"**{delta['v'] * 100:.1f} points** ({delta['lo'] * 100:.1f} to "
                f"{delta['hi'] * 100:.1f})."
            ),
            table(
                [
                    "Approach",
                    "High-cost members reached at 10%",
                    "Next-year dollars reached at 10%",
                    "AUPRC",
                    "AUROC",
                ],
                rows,
            ),
            (
                f"Test set: index years {years}, {test['n']:,} member-years from "
                f"{test['n_members']:,} members, {test['n_pos']:,} high cost. Intervals are 95%, "
                f"from {m['n_bootstrap']:,} bootstrap resamples of members."
            ),
        ]
    )


def block_gain(m: dict, **_) -> str:
    s = m["sensitivity"]
    g = s["gain_decomposition"]
    strata = s["by_childbearing_stratum"]
    women, rest = strata["women_18_49"], strata["everyone_else"]
    only = g["found_by_model_only"]
    rest_only = g["found_by_model_only_everyone_else"]
    return "\n\n".join(
        [
            (
                f"At 10% outreach the model reached {only} high-cost member-years the cost rule "
                f"missed, and missed {g['found_by_prior_only']} the rule reached. "
                f"{g['found_by_model_only_pregnancy_in_outcome_year']} of those {only} "
                f"({pct(s['gain_share_pregnancy_in_outcome_year'], 0)}) had a pregnancy during the "
                f"outcome year, and for "
                f"{g['found_by_model_only_pregnancy_starting_in_outcome_year']} of them it began "
                f"after January 1. In this synthetic population "
                f"{pct(s['test_pos_share_pregnancy_in_outcome_year'], 0)} of high-cost member-years "
                f"include a pregnancy, and the model anticipates one from age and sex. The flag for "
                f"a pregnancy already under way on January 1 accounts for "
                f"{g['found_by_model_only_pregnant']} of the {only}, because the cost rule already "
                f"calls nearly all of those members."
            ),
            (
                f"The model beats the rule on both sides of that line. Among women 18 to 49 it "
                f"reaches {pct(women['lgbm']['recall_at_10']['v'])} against "
                f"{pct(women['prior_cost']['recall_at_10']['v'])}. Among everyone else it reaches "
                f"{pct(rest['lgbm']['recall_at_10']['v'])} against "
                f"{pct(rest['prior_cost']['recall_at_10']['v'])}, a difference of "
                f"{rest['delta_recall_at_10_vs_prior']['v'] * 100:.1f} points "
                f"({rest['delta_recall_at_10_vs_prior']['lo'] * 100:.1f} to "
                f"{rest['delta_recall_at_10_vs_prior']['hi'] * 100:.1f}). Of the {rest_only} extra "
                f"finds in that second group, "
                f"{g['found_by_model_only_everyone_else_with_chronic_kidney_disease']} have chronic "
                f"kidney disease and {g['found_by_model_only_everyone_else_with_diabetes']} have "
                f"diabetes."
            ),
        ]
    )


def block_evaluation(m: dict, **_) -> str:
    models = m["models"]
    names = [
        ("lgbm", "Gradient boosted trees"),
        ("logreg", "Logistic regression"),
        ("prior_cost", "Last year's cost"),
    ]
    rows = []
    for key, label in names:
        v = models[key]
        rows.append(
            [
                label,
                pct_iv(v["at"][K]["recall"]),
                pct_iv(v["at"]["0.05"]["recall"]),
                pct_iv(v["at"]["0.20"]["recall"]),
                pct_iv(v["at"][K]["dollars"]),
                num_iv(v["auprc"]),
                num_iv(v["auroc"]),
            ]
        )
    d = m["deltas"]["lgbm_minus_prior_cost"]
    d2 = m["deltas"]["lgbm_minus_logreg"]
    d3 = m["deltas"]["logreg_minus_prior_cost"]
    paired = table(
        ["Paired difference", "Recall at 10%, points", "Dollars at 10%, points", "AUPRC", "AUROC"],
        [
            [
                "Trees minus last year's cost",
                points_iv(d[f"recall@{K}"]),
                points_iv(d[f"dollars@{K}"]),
                diff_iv(d["auprc"]),
                diff_iv(d["auroc"]),
            ],
            [
                "Logistic regression minus last year's cost",
                points_iv(d3[f"recall@{K}"]),
                points_iv(d3[f"dollars@{K}"]),
                diff_iv(d3["auprc"]),
                diff_iv(d3["auroc"]),
            ],
            [
                "Trees minus logistic regression",
                points_iv(d2[f"recall@{K}"]),
                points_iv(d2[f"dollars@{K}"]),
                diff_iv(d2["auprc"]),
                diff_iv(d2["auroc"]),
            ],
        ],
    )
    main = table(
        [
            "Approach",
            "Recall at 10%",
            "Recall at 5%",
            "Recall at 20%",
            "Dollars at 10%",
            "AUPRC",
            "AUROC",
        ],
        rows,
    )
    test = m["test"]
    note = (
        f"{test['n']:,} member-years, {test['n_pos']:,} high cost, index years "
        f"{' and '.join(str(y) for y in test['years'])}. 95% intervals from {m['n_bootstrap']:,} "
        f"bootstrap resamples of members. The same resamples are used for every approach, so "
        f"the differences are paired."
    )
    return f"{main}\n\n{paired}\n\n{note}"


def block_strata(m: dict, **_) -> str:
    strata = m["sensitivity"]["by_childbearing_stratum"]
    rows = []
    for key, label in (("women_18_49", "Women 18 to 49"), ("everyone_else", "Everyone else")):
        s = strata[key]
        rows.append(
            [
                label,
                f"{s['n']:,}",
                f"{s['n_pos']:,}",
                pct(s["share_of_high_cost"], 0),
                pct_iv(s["lgbm"]["recall_at_10"]),
                pct_iv(s["prior_cost"]["recall_at_10"]),
                points_iv(s["delta_recall_at_10_vs_prior"]),
            ]
        )
    return (
        table(
            [
                "Stratum",
                "Member-years",
                "High cost",
                "Share of all high cost",
                "Model, recall at 10%",
                "Last year's cost, recall at 10%",
                "Difference, points",
            ],
            rows,
        )
        + "\n\nEach stratum is ranked on its own, and the top 10% of that stratum is called."
    )


def block_decomposition(m: dict, **_) -> str:
    s = m["sensitivity"]
    g = s["gain_decomposition"]
    only = g["found_by_model_only"]
    rows = [
        ["Reached by both", g["found_by_both"]],
        ["Reached only by the cost rule", g["found_by_prior_only"]],
        ["Reached only by the model", only],
        [
            "of those: pregnancy during the outcome year",
            (
                f"{g['found_by_model_only_pregnancy_in_outcome_year']} "
                f"({pct(s['gain_share_pregnancy_in_outcome_year'], 0)})"
            ),
        ],
        [
            "of those: pregnancy that began after January 1",
            (
                f"{g['found_by_model_only_pregnancy_starting_in_outcome_year']} "
                f"({pct(s['gain_share_pregnancy_starting_in_outcome_year'], 0)})"
            ),
        ],
        [
            "of those: pregnant on January 1",
            f"{g['found_by_model_only_pregnant']} ({pct(s['gain_share_pregnant'], 0)})",
        ],
        [
            "of those: not a woman 18 to 49",
            (
                f"{g['found_by_model_only_everyone_else']} "
                f"({pct(g['found_by_model_only_everyone_else'] / only, 0)})"
            ),
        ],
        [
            "of those: chronic kidney disease",
            g["found_by_model_only_everyone_else_with_chronic_kidney_disease"],
        ],
        ["of those: diabetes", g["found_by_model_only_everyone_else_with_diabetes"]],
    ]
    for i in (3, 4, 5, 6):
        rows[i][0] = "&nbsp;&nbsp;" + rows[i][0]
    for i in (7, 8):
        rows[i][0] = "&nbsp;&nbsp;&nbsp;&nbsp;" + rows[i][0]
    rank = g["model_only_prior_cost_rank"]
    return "\n\n".join(
        [
            table(["High-cost member-years in the test set, at 10% outreach", "Count"], rows),
            (
                f"{g['high_cost_pregnancy_in_outcome_year']:,} of the "
                f"{m['test']['n_pos']:,} high-cost member-years "
                f"({pct(s['test_pos_share_pregnancy_in_outcome_year'], 0)}) include a pregnancy "
                f"during the outcome year. The members only the model reaches are not near misses "
                f"for the cost rule: their median position in the prior-cost ranking is the "
                f"{rank['median'] * 100:.0f}th percentile from the top, and "
                f"{pct(rank['share_beyond_0.50'], 0)} are in the bottom half."
            ),
        ]
    )


def block_sensitivity(m: dict, sanity: dict, **_) -> str:
    s = m["sensitivity"]
    ex, wo = s["excluding_pregnant"], s["without_pregnancy_feature"]
    rule = s["prior_cost_plus_pregnancy_rule"]
    no_cost = sanity["no_prior_cost_features"]
    full = m["models"]["lgbm"]["at"][K]["recall"]
    prior = m["models"]["prior_cost"]["at"][K]["recall"]
    rows = [
        [
            "Full model, full test set",
            pct_iv(full),
            pct_iv(prior),
            points_iv(m["deltas"]["lgbm_minus_prior_cost"][f"recall@{K}"]),
        ],
        [
            f"Test set without members pregnant on January 1 ({ex['n_excluded']} rows dropped)",
            pct_iv(ex["lgbm"]["recall_at_10"]),
            pct_iv(ex["prior_cost"]["recall_at_10"]),
            points_iv(ex["delta_recall_at_10_vs_prior"]),
        ],
        [
            "Model refit without the pregnancy feature",
            pct_iv(wo["lgbm"]["recall_at_10"]),
            pct_iv(prior),
            points_iv(wo["delta_recall_at_10_vs_prior"]),
        ],
        [
            f"Model refit without the {len(no_cost['dropped'])} prior-cost features",
            pct_iv(no_cost["recall_at_10_ci"]),
            pct_iv(prior),
            "",
        ],
        [
            "Full model against a stronger rule: pregnant members first, then by cost",
            pct_iv(full),
            pct_iv(rule["prior_cost_plus_pregnancy"]["recall_at_10"]),
            points_iv(rule["lgbm_minus_rule_recall_at_10"]),
        ],
    ]
    return table(
        ["Check", "Model, recall at 10%", "Rule, recall at 10%", "Difference, points"], rows
    )


def block_calibration(m: dict, **_) -> str:
    c = m["calibration_before_after"]
    rows = [
        [
            label,
            num_iv(c[key]["brier"], 4),
            num_iv(c[key]["ece"], 4),
            num_iv(c[key]["calibration_slope"], 2),
            num_iv(c[key]["calibration_intercept"], 2),
        ]
        for key, label in (("raw", "Before (sigmoid of the raw score)"), ("calibrated", "After"))
    ]
    change = c["change"]
    rows.append(
        ["Change", diff_iv(change["brier"], 4), diff_iv(change["ece"], 4), "", ""],
    )
    return table(
        ["Isotonic calibration", "Brier score", "Expected calibration error", "Slope", "Intercept"],
        rows,
    )


def block_holdout(m: dict, **_) -> str:
    rows = []
    for key, label in (
        ("seen", "Members with rows in training"),
        ("unseen", "Members never seen in training"),
    ):
        h = m["member_holdout"][key]
        rows.append(
            [
                label,
                f"{h['n']:,}",
                f"{h['n_pos']:,}",
                pct_iv(h["recall_at_10_ci"]),
                pct(h["prior_cost"]["recall_at_10"]),
                points_iv(h["delta_recall_at_10_vs_prior"]),
                f"{h['auroc']:.3f}",
            ]
        )
    return table(
        [
            "Test rows",
            "Member-years",
            "High cost",
            "Model, recall at 10%",
            "Last year's cost",
            "Difference, points",
            "Model AUROC",
        ],
        rows,
    )


def block_subgroups(fairness: dict, **_) -> str:
    rows = []
    for g in fairness["groups"]:
        gap = (g["recall_at_10"] - g["prior_cost_recall_at_10"]) * 100
        rows.append(
            [
                f"{g['attribute'].capitalize()}: {g['level']}" + (" (small)" if g["small"] else ""),
                f"{g['n']:,}",
                f"{g['n_pos']:,}",
                f"{g['auroc']:.3f}",
                f"{g['obs_over_expected']:.2f}",
                pct(g["selection_rate_at_10"]),
                pct_iv(g["recall_at_10_ci"]),
                pct(g["prior_cost_recall_at_10"]),
                f"{gap:+.1f}",
            ]
        )
    return (
        table(
            [
                "Group",
                "Member-years",
                "High cost",
                "AUROC",
                "Observed / expected",
                "Share called at 10%",
                "Model, recall at 10%",
                "Last year's cost",
                "Difference, points",
            ],
            rows,
        )
        + "\n\nOne ranking of the whole test set, top 10% called. Recall is the share of a "
        "group's high-cost members who were in that top 10%. " + fairness["small_group_rule"]
    )


def block_proxy(fairness: dict, **_) -> str:
    audit = fairness["proxy_audit"]
    rows = [
        [
            f"{c['level'].capitalize()} minus {c['reference']}"
            + (" (small)" if c["small"] else ""),
            f"{c['n']:,}",
            diff_iv(c["chronic_gap_all_deciles"], 2),
        ]
        for c in audit["comparisons"]
    ]
    return table(
        [
            "Comparison at the same risk score decile",
            "Member-years",
            "Gap in active chronic condition groups",
        ],
        rows,
    )


def block_drift(drift: dict, **_) -> str:
    perf = {p["year"]: p for p in drift["performance_by_year"]}
    by_feature = drift["feature_psi_by_year"]
    rows = []
    for y in drift["by_year"]:
        name, value = max(by_feature[str(y["year"])].items(), key=lambda kv: kv[1] or 0)
        p = perf[y["year"]]
        held = p.get("holdout_members")
        rows.append(
            [
                y["year"],
                y["split"],
                y["n_alert"],
                y["n_watch"],
                f"`{name}` ({value:.2f})",
                f"{y['score_psi']:.3f}",
                pct(held["lgbm"]["recall_at_10"]) if held else pct(p["lgbm"]["recall_at_10"]),
                (
                    pct(held["prior_cost"]["recall_at_10"])
                    if held
                    else pct(p["prior_cost"]["recall_at_10"])
                ),
                f"{held['n']:,}" if held else f"{p['n']:,} (all members)",
                "yes" if y["labels_mature"] else "no",
            ]
        )
    t = drift["thresholds"]
    return (
        table(
            [
                "Index year",
                "Split",
                "Inputs at alert",
                "Inputs at watch",
                "Largest input PSI",
                "Score PSI",
                "Model, recall at 10%",
                "Last year's cost",
                "Member-years",
                "Labels mature",
            ],
            rows,
            align="llrrlrrrrl",
        )
        + f"\n\nReference: index years {drift['reference_years'][0]} to "
        f"{drift['reference_years'][-1]}. Watch at PSI {t['watch']:.2f}, alert at "
        f"{t['alert']:.2f}. Recall before the test years is on held-out members only, because "
        f"the model trained on the others. In the test years it is on everyone."
    )


BLOCKS = {
    "metrics": block_metrics,
    "gain": block_gain,
    "evaluation": block_evaluation,
    "strata": block_strata,
    "decomposition": block_decomposition,
    "sensitivity": block_sensitivity,
    "calibration": block_calibration,
    "holdout": block_holdout,
    "subgroups": block_subgroups,
    "proxy": block_proxy,
    "drift": block_drift,
}

MARKER = re.compile(r"<!-- (\w+):start -->\n.*?<!-- \1:end -->", re.DOTALL)


def render(text: str, reports: dict) -> str:
    def fill(match: re.Match) -> str:
        name = match.group(1)
        if name not in BLOCKS:
            raise SystemExit(f"unknown block <!-- {name}:start -->")
        body = BLOCKS[name](**reports)
        return f"<!-- {name}:start -->\n{body}\n<!-- {name}:end -->"

    return MARKER.sub(fill, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="do not write, exit 1 if stale")
    args = parser.parse_args(argv)

    reports = {
        "m": load("metrics.json"),
        "sanity": load("sanity.json"),
        "fairness": load("fairness.json"),
        "drift": load("drift.json"),
    }
    stale = []
    for name in DOCUMENTS:
        path = ROOT / name
        if not path.exists():
            print(f"{name}: not found", file=sys.stderr)
            return 1
        old = path.read_text()
        new = render(old, reports)
        blocks = len(MARKER.findall(old))
        status = "up to date"
        if new != old:
            stale.append(name)
            status = "stale" if args.check else "updated"
            if not args.check:
                path.write_text(new)
        print(f"{name}: {blocks} blocks, {status}")
    if args.check and stale:
        print(f"run scripts/update_readme_metrics.py to refresh: {', '.join(stale)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
