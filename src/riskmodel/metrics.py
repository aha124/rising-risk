"""Metric functions, written so one pass gives the value on the data and on every
bootstrap resample.

A resample is a vector of non-negative integer weights, one per row (how many times
the row was drawn). Every function here takes an optional weight matrix with one
row per resample and returns one value per resample. With no weights it returns a
single value, the plain metric, and the tests check that against scikit-learn.

Ranking uses the score in descending order with ties broken by `tie_key` ascending.
A top-k cut that falls inside a row drawn several times takes only as many copies
of that row as fit, which is what cutting the expanded resample would do.
"""

from __future__ import annotations

import numpy as np

CHUNK = 200  # resamples handled at once, keeps the weight matrices near 30 MB


def rank_order(score: np.ndarray, tie_key: np.ndarray) -> np.ndarray:
    """Row positions from highest score to lowest, ties by tie_key ascending."""
    return np.lexsort((np.asarray(tie_key), -np.asarray(score, dtype=float)))


def interval(point: float, samples: np.ndarray) -> dict:
    """Point estimate with the 2.5 and 97.5 percentiles of the bootstrap values."""
    samples = np.asarray(samples, dtype=float)
    ok = samples[np.isfinite(samples)]
    if len(ok) == 0:
        return {"v": _num(point), "lo": None, "hi": None}
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return {"v": _num(point), "lo": float(lo), "hi": float(hi)}


def _num(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _prefix_at(cum_w, cum_v, per_unit, m):
    """Sum of a value over the first m units of each weighted sequence.

    cum_w, cum_v: (B, n) running totals of weight and weight x value in rank order.
    per_unit: (n,) the value of one copy of each row. m: (B,) number of units.
    """
    n = cum_w.shape[1]
    rows = np.arange(cum_w.shape[0])
    full = (cum_w <= m[:, None]).sum(axis=1)  # rows that fit completely
    last = np.maximum(full - 1, 0)
    w_full = np.where(full > 0, cum_w[rows, last], 0.0)
    v_full = np.where(full > 0, cum_v[rows, last], 0.0)
    part = np.clip(m - w_full, 0.0, None)
    nxt = np.minimum(full, n - 1)
    return v_full + part * per_unit[nxt]


def _weight_chunks(weights, order, n):
    if weights is None:
        yield np.ones((1, n))
        return
    for start in range(0, weights.shape[0], CHUNK):
        yield weights[start : start + CHUNK][:, order].astype(np.float64)


def ranking_metrics(score, y, cost, tie_key, weights=None, ks=(0.01, 0.05, 0.10, 0.20)) -> dict:
    """AUROC, average precision, and precision, recall, lift and dollar share at each k.

    Returns {name: array of length B}. Names: "auroc", "auprc", and for each k
    "precision@0.10", "recall@0.10", "lift@0.10", "dollars@0.10".
    """
    score = np.asarray(score, dtype=float)
    order = rank_order(score, tie_key)
    s = score[order]
    yy = np.asarray(y, dtype=float)[order]
    cc = np.asarray(cost, dtype=float)[order]
    n = len(s)
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])  # last row of each tied block
    out: dict[str, list] = {}

    def put(name, values):
        out.setdefault(name, []).append(values)

    with np.errstate(divide="ignore", invalid="ignore"):
        for w in _weight_chunks(weights, order, n):
            cum_w = np.cumsum(w, axis=1)
            cum_tp = np.cumsum(w * yy, axis=1)
            cum_cost = np.cumsum(w * cc, axis=1)
            total = cum_w[:, -1]
            pos = cum_tp[:, -1]
            neg = total - pos
            dollars = cum_cost[:, -1]

            tp = cum_tp[:, ends]
            fp = cum_w[:, ends] - tp
            tp_prev = np.c_[np.zeros(len(w)), tp[:, :-1]]
            fp_prev = np.c_[np.zeros(len(w)), fp[:, :-1]]
            put("auroc", ((fp - fp_prev) * (tp + tp_prev) / 2).sum(axis=1) / (pos * neg))
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            put("auprc", ((tp - tp_prev) * precision).sum(axis=1) / pos)

            for k in ks:
                m = np.maximum(1.0, np.round(k * total))
                found = _prefix_at(cum_w, cum_tp, yy, m)
                spent = _prefix_at(cum_w, cum_cost, cc, m)
                key = f"{k:.2f}"
                put(f"precision@{key}", found / m)
                put(f"recall@{key}", found / pos)
                put(f"lift@{key}", (found / m) / (pos / total))
                put(f"dollars@{key}", spent / dollars)
    return {name: np.concatenate(parts) for name, parts in out.items()}


def probability_metrics(p, y, tie_key, weights=None, bins: int = 10) -> dict:
    """Brier score and expected calibration error (equal-count bins).

    Rows are ordered by predicted probability (ties by tie_key) and cut into `bins`
    groups of equal size, the first groups taking any remainder. ECE is the
    size-weighted mean of |mean predicted - observed rate| over the groups.
    """
    p = np.asarray(p, dtype=float)
    order = np.lexsort((np.asarray(tie_key), p))
    pp = p[order]
    yy = np.asarray(y, dtype=float)[order]
    n = len(pp)
    sq = (pp - yy) ** 2
    out: dict[str, list] = {"brier": [], "ece": []}
    for w in _weight_chunks(weights, order, n):
        cum_w = np.cumsum(w, axis=1)
        cum_p = np.cumsum(w * pp, axis=1)
        cum_y = np.cumsum(w * yy, axis=1)
        total = cum_w[:, -1]
        out["brier"].append((w * sq).sum(axis=1) / total)
        base, extra = np.divmod(total, bins)
        gap = np.zeros(len(w))
        prev_p = np.zeros(len(w))
        prev_y = np.zeros(len(w))
        for j in range(1, bins + 1):
            m = base * j + np.minimum(extra, j)
            sp = _prefix_at(cum_w, cum_p, pp, m)
            sy = _prefix_at(cum_w, cum_y, yy, m)
            gap += np.abs((sp - prev_p) - (sy - prev_y))
            prev_p, prev_y = sp, sy
        out["ece"].append(gap / total)
    return {name: np.concatenate(parts) for name, parts in out.items()}


def calibration_bins(p, y, tie_key, bins: int = 10) -> list[dict]:
    """Mean predicted probability and observed rate in equal-count bins, lowest first."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.lexsort((np.asarray(tie_key), p))
    return [
        {"mean_pred": float(p[i].mean()), "obs_rate": float(y[i].mean()), "n": len(i)}
        for i in np.array_split(order, bins)
        if len(i)
    ]


def logit(p, eps: float = 1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_line(p, y, weights=None, eps: float = 1e-4) -> dict:
    """Slope and intercept of a logistic regression of the outcome on logit(p).

    A well calibrated score gives slope 1 and intercept 0. Probabilities are clipped
    to [eps, 1 - eps] first, because isotonic calibration can output exactly 0 or 1.
    """
    x = logit(p, eps)
    y = np.asarray(y, dtype=float)
    rows = [None] if weights is None else weights
    slopes, intercepts = [], []
    for w in rows:
        a, b = _logistic_1d(x, y, None if w is None else np.asarray(w, dtype=float))
        intercepts.append(a)
        slopes.append(b)
    return {"slope": np.array(slopes), "intercept": np.array(intercepts)}


def _logistic_1d(x, y, w=None, iters: int = 50):
    """Newton's method for y ~ a + b x. Returns (a, b), or NaN if it does not converge."""
    w = np.ones_like(x) if w is None else w
    if (w * y).sum() <= 0 or (w * (1 - y)).sum() <= 0:
        return np.nan, np.nan
    a, b = 0.0, 1.0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-np.clip(a + b * x, -500, 500)))
        g0 = (w * (y - mu)).sum()
        g1 = (w * (y - mu) * x).sum()
        v = w * mu * (1 - mu)
        h00, h01, h11 = v.sum(), (v * x).sum(), (v * x * x).sum()
        det = h00 * h11 - h01 * h01
        if not np.isfinite(det) or det <= 0:
            return np.nan, np.nan
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-10:
            break
    return float(a), float(b)


def top_k_mask(score, tie_key, k: float) -> np.ndarray:
    """True for the rows contacted at capacity k (a share of rows)."""
    order = rank_order(score, tie_key)
    m = max(1, int(np.round(k * len(order))))
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:m]] = True
    return mask


def rank_pct(score, tie_key) -> np.ndarray:
    """Position in the ranking as a fraction: 0 is the highest score."""
    order = rank_order(score, tie_key)
    pct = np.empty(len(order))
    pct[order] = np.arange(len(order)) / len(order)
    return pct


def member_resamples(n_members: int, n_resamples: int, seed: int) -> np.ndarray:
    """How many times each member is drawn in each resample, shape (B, n_members).

    Members are drawn with replacement, so both years of a member move together.
    """
    rng = np.random.default_rng(seed)
    probs = np.full(n_members, 1.0 / n_members)
    return rng.multinomial(n_members, probs, size=n_resamples).astype(np.int16)
