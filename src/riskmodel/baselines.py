"""The logistic regression baseline.

Median imputation, a signed log1p on dollar and count columns, standard scaling,
then L2 logistic regression. The fitted pieces are plain arrays, so the model saves
to a small JSON file and predicts with numpy alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

LOG_UNITS = ("USD", "count", "days")


def log_feature_mask(units: list[str]) -> np.ndarray:
    return np.array([u in LOG_UNITS for u in units], dtype=bool)


@dataclass
class LogRegBaseline:
    medians: np.ndarray
    log_mask: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    C: float

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        Z = prepare(X, self.medians, self.log_mask)
        return (Z - self.mean) / self.scale

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._prepare(X) @ self.coef + self.intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision_function(X)))

    def to_json(self) -> dict:
        return {
            "medians": self.medians.tolist(),
            "log_mask": [bool(v) for v in self.log_mask],
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
            "C": float(self.C),
        }

    @classmethod
    def from_json(cls, d: dict) -> LogRegBaseline:
        return cls(
            medians=np.asarray(d["medians"], dtype=float),
            log_mask=np.asarray(d["log_mask"], dtype=bool),
            mean=np.asarray(d["mean"], dtype=float),
            scale=np.asarray(d["scale"], dtype=float),
            coef=np.asarray(d["coef"], dtype=float),
            intercept=float(d["intercept"]),
            C=float(d["C"]),
        )


def prepare(X: np.ndarray, medians: np.ndarray, log_mask: np.ndarray) -> np.ndarray:
    """Fill missing values with the training median, then sign(x) * log1p(|x|) where flagged."""
    Z = np.array(X, dtype=np.float64, copy=True)
    bad = np.isnan(Z)
    if bad.any():
        Z[bad] = np.take(medians, np.where(bad)[1])
    cols = np.flatnonzero(log_mask)
    Z[:, cols] = np.sign(Z[:, cols]) * np.log1p(np.abs(Z[:, cols]))
    return Z


def fit_logreg(X: np.ndarray, y: np.ndarray, log_mask: np.ndarray, C: float) -> LogRegBaseline:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)  # a column that is all missing
    Z = prepare(X, medians, log_mask)
    mean = Z.mean(axis=0)
    scale = Z.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit((Z - mean) / scale, y)
    return LogRegBaseline(
        medians=medians,
        log_mask=log_mask,
        mean=mean,
        scale=scale,
        coef=model.coef_[0].astype(float),
        intercept=float(model.intercept_[0]),
        C=float(C),
    )
