"""Turn validated feature objects into scores.

All arithmetic happens in riskmodel.scoring.Scorer, the same object batch evaluation
and the site export use. This module only arranges inputs into the model's feature
order and formats what comes back, so the API cannot drift from batch scoring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from riskmodel import config, dataset
from riskmodel.explain import display_value
from riskmodel.scoring import Scorer

from . import audit

TOP_FACTORS = 5


class ScoringService:
    def __init__(self, model_dir: Path | None = None, reports_dir: Path | None = None):
        paths = config.paths()
        self.model_dir = Path(model_dir or paths.model_dir)
        self.reports_dir = Path(reports_dir or paths.reports_dir)
        self.scorer = Scorer.load(self.model_dir)
        self.spec = dataset.read_json(self.model_dir / "feature_spec.json")["features"]
        self.features = self.scorer.features
        self.model_version = str(self.scorer.metadata["model_version"])
        self._lo = np.array([f["min"] for f in self.spec], dtype=float)
        self._hi = np.array([f["max"] for f in self.spec], dtype=float)

    def row(self, features: dict) -> list[float | None]:
        """Feature values in model order, None where the request left one out."""
        return [features.get(name) for name in self.features]

    def score(self, rows: list[list[float | None]]) -> list[dict]:
        X = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=np.float64)
        raw = self.scorer.raw(X)
        p = self.scorer.calibrate(raw)
        contrib = self.scorer.contributions(X)[:, :-1]
        out = []
        for i in range(len(rows)):
            prob = float(p[i])
            top = np.argsort(-np.abs(contrib[i]), kind="stable")[:TOP_FACTORS]
            outside = np.flatnonzero((X[i] < self._lo) | (X[i] > self._hi))
            out.append(
                {
                    "probability": prob,
                    "percentile": self.scorer.percentile(prob),
                    "tier": self.scorer.tier(prob),
                    "top_factors": [self._factor(j, X[i, j], contrib[i, j]) for j in top],
                    "n_missing": int(np.isnan(X[i]).sum()),
                    "warnings": [
                        f"{self.features[j]} = {X[i, j]:g} is outside the training range "
                        f"({self._lo[j]:g} to {self._hi[j]:g})"
                        for j in outside
                    ],
                    "input_sha256": audit.input_sha256(rows[i]),
                }
            )
        return out

    def _factor(self, j: int, value: float, contribution: float) -> dict:
        f = self.spec[j]
        return {
            "feature": f["name"],
            "label": f["label"],
            "value": display_value(value, f["unit"], f["dtype"]),
            "contribution": float(contribution),
            "direction": "raises" if contribution > 0 else "lowers",
        }

    def model_info(self) -> dict:
        """Metadata, the feature spec and the test metrics the model shipped with."""
        info = {
            "metadata": self.scorer.metadata,
            "tiers": {k: self.scorer.tiers[k] for k in ("high", "rising")},
            "features": self.spec,
            "test_metrics": None,
        }
        path = self.reports_dir / "metrics.json"
        if path.exists():
            m = dataset.read_json(path)
            lgbm, prior = m["models"]["lgbm"], m["models"]["prior_cost"]
            k = f"{config.HEADLINE_K:.2f}"
            info["test_metrics"] = {
                "test": m["test"],
                "n_bootstrap": m["n_bootstrap"],
                "recall_at_10": {
                    "model": lgbm["at"][k]["recall"],
                    "prior_cost_rule": prior["at"][k]["recall"],
                },
                "delta_recall_at_10_vs_prior_cost_rule": m["deltas"]["lgbm_minus_prior_cost"][
                    f"recall@{k}"
                ],
                "auroc": lgbm["auroc"],
                "auprc": lgbm["auprc"],
                "brier": lgbm["brier"],
                "ece": lgbm["ece"],
            }
        return info
