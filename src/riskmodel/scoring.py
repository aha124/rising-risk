"""Load the saved model and score rows. Batch evaluation, the site export and the
API all go through this module, so there is one definition of a score.

raw          sum of leaf values (log-odds), used for every ranking
p_raw        sigmoid of raw, before calibration
p            p_raw passed through the isotonic calibrator (linear interpolation
             between its thresholds, clipped at both ends)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, dataset
from .baselines import LogRegBaseline


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def apply_calibrator(p_raw, x, y):
    """Isotonic calibrator as exported: interpolate between thresholds, clip outside."""
    return np.interp(np.asarray(p_raw, dtype=float), x, y)


@dataclass
class Scorer:
    booster: lgb.Booster
    features: list[str]
    calib_x: np.ndarray
    calib_y: np.ndarray
    tiers: dict
    score_percentiles: np.ndarray | None
    metadata: dict

    @classmethod
    def load(cls, model_dir: Path | None = None) -> Scorer:
        model_dir = Path(model_dir or config.paths().model_dir)
        spec = dataset.read_json(model_dir / "feature_spec.json")
        cal = dataset.read_json(model_dir / "calibrator.json")
        tiers = dataset.read_json(model_dir / "tiers.json")
        ref_path = model_dir / "score_reference.json"
        ref = dataset.read_json(ref_path)["percentiles"] if ref_path.exists() else None
        booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        features = [f["name"] for f in spec["features"]]
        if booster.feature_name() != features:
            raise ValueError("model.txt and feature_spec.json disagree on feature order")
        return cls(
            booster=booster,
            features=features,
            calib_x=np.asarray(cal["x"], dtype=float),
            calib_y=np.asarray(cal["y"], dtype=float),
            tiers=tiers,
            score_percentiles=None if ref is None else np.asarray(ref, dtype=float),
            metadata=dataset.read_json(model_dir / "metadata.json"),
        )

    def raw(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(np.asarray(X, dtype=np.float64), raw_score=True, num_threads=1)

    def calibrate(self, raw) -> np.ndarray:
        return apply_calibrator(sigmoid(raw), self.calib_x, self.calib_y)

    def probability(self, X: np.ndarray) -> np.ndarray:
        return self.calibrate(self.raw(X))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """TreeSHAP values in log-odds, one column per feature plus the base value last."""
        return self.booster.predict(
            np.asarray(X, dtype=np.float64), pred_contrib=True, num_threads=1
        )

    def tier(self, p: float) -> str:
        if p >= self.tiers["high"]:
            return "high"
        if p >= self.tiers["rising"]:
            return "rising"
        return "routine"

    def percentile(self, p: float) -> float | None:
        """Where a calibrated probability sits in the reference distribution, 0 to 100.

        Same lookup as percentileOf in docs/js/scorer.js.
        """
        q = self.score_percentiles
        if q is None or len(q) < 2:
            return None
        last = len(q) - 1
        if p <= q[0]:
            return 0.0
        if p >= q[last]:
            return 100.0
        lo = int(np.searchsorted(q, p, side="right")) - 1
        span = q[lo + 1] - q[lo]
        frac = 0.0 if span == 0 else (p - q[lo]) / span
        return float((lo + frac) / last * 100)


@dataclass
class Scored:
    frame: pd.DataFrame  # mart rows with split columns and score columns
    features: list[str]
    dictionary: pd.DataFrame
    scorer: Scorer
    logreg: LogRegBaseline
    plan: dataset.YearPlan

    @property
    def test(self) -> pd.DataFrame:
        return self.frame[self.frame["split"] == "test"]

    def X(self, rows: pd.DataFrame) -> np.ndarray:
        return dataset.feature_matrix(rows, self.features)


# Score column used to rank under each approach.
RANK_COLUMN = {"lgbm": "raw", "logreg": "logreg", "prior_cost": "prior_cost"}


def load_scored(paths: config.Paths | None = None) -> Scored:
    """The whole mart scored by the saved models."""
    paths = paths or config.paths()
    frame = dataset.load_mart(paths.mart)
    dictionary = dataset.load_feature_dictionary(paths.feature_dictionary)
    scorer = Scorer.load(paths.model_dir)
    features = dataset.feature_names(dictionary, frame)
    if features != scorer.features:
        raise ValueError("the saved model was trained on a different feature list")
    logreg = LogRegBaseline.from_json(dataset.read_json(paths.model_dir / "logreg.json"))
    X = dataset.feature_matrix(frame, features)
    frame = frame.copy()
    frame["raw"] = scorer.raw(X)
    frame["p_raw"] = sigmoid(frame["raw"].to_numpy())
    frame["p"] = scorer.calibrate(frame["raw"].to_numpy())
    frame["logreg"] = logreg.predict_proba(X)
    frame["prior_cost"] = frame[config.PRIOR_COST].to_numpy(dtype=float)
    plan = dataset.year_plan(frame["index_year"].unique())
    return Scored(frame, features, dictionary, scorer, logreg, plan)
