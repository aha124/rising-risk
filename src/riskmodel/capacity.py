"""What each approach finds at every outreach capacity from 1% to 100% of members.

Both test years are ranked together by the raw score, ties broken by member id.
`found` and `dollars` are shares of all high-cost members and of all next-year
dollars in the test set. Random is the expected value, not a draw.
"""

from __future__ import annotations

import numpy as np

from . import config, dataset
from . import metrics as M
from .scoring import RANK_COLUMN, load_scored


def capacity_table(test, models=("lgbm", "prior_cost", "logreg")) -> list[dict]:
    y = test[config.LABEL].to_numpy(dtype=float)
    cost = test[config.COST_NEXT].to_numpy(dtype=float)
    tie = test["tie_key"].to_numpy()
    n = len(test)
    prevalence = float(y.mean())
    cum = {}
    for key in models:
        order = M.rank_order(test[RANK_COLUMN[key]].to_numpy(), tie)
        cum[key] = (np.cumsum(y[order]), np.cumsum(cost[order]))
    rows = []
    for pct in range(1, 101):
        m = max(1, int(np.round(pct / 100 * n)))
        row = {"pct": pct / 100, "n_contacted": m}
        for key in models:
            found, dollars = cum[key][0][m - 1], cum[key][1][m - 1]
            row[key] = {
                "found": float(found / y.sum()),
                "dollars": float(dollars / cost.sum()),
                "precision": float(found / m),
                "n_found": int(found),
            }
        row["random"] = {
            "found": pct / 100,
            "dollars": pct / 100,
            "precision": prevalence,
            "n_found": float(prevalence * m),
        }
        rows.append(row)
    return rows


def run() -> dict:
    paths = config.paths()
    sc = load_scored(paths)
    test = sc.test
    report = {
        "note": "Test years ranked together by the raw score, ties broken by member id. "
        "found and dollars are shares of the test-set totals.",
        "n_test": len(test),
        "n_high_cost": int(test[config.LABEL].sum()),
        "total_dollars": float(test[config.COST_NEXT].sum()),
        "capacity": capacity_table(test),
    }
    dataset.write_json(paths.reports_dir / "capacity.json", report)
    at10 = report["capacity"][9]
    print(
        f"capacity: at 10% ({at10['n_contacted']} members) lgbm finds "
        f"{at10['lgbm']['n_found']}, prior cost {at10['prior_cost']['n_found']}"
    )
    return report
