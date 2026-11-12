"""Metric functions against scikit-learn and against values worked out by hand."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from riskmodel import metrics as M


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 4000
    score = np.round(rng.normal(size=n), 1)  # rounded so there are ties
    y = (rng.random(n) < 1 / (1 + np.exp(-score - 2))).astype(int)
    cost = rng.exponential(1000, n) * (1 + 3 * y)
    return score, y, cost, rng.permutation(n)


def test_auroc_and_auprc_match_sklearn_with_ties(data):
    score, y, cost, tie = data
    r = M.ranking_metrics(score, y, cost, tie)
    assert r["auroc"][0] == pytest.approx(roc_auc_score(y, score), abs=1e-12)
    assert r["auprc"][0] == pytest.approx(average_precision_score(y, score), abs=1e-12)


def test_top_k_by_hand():
    # 10 members, capacity 20% = 2 members. Ranked by score: rows 0 and 1.
    score = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0.0])
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    cost = np.array([50, 10, 20, 0, 0, 0, 20, 0, 0, 0.0])
    r = M.ranking_metrics(score, y, cost, np.arange(10), ks=(0.2,))
    assert r["precision@0.20"][0] == pytest.approx(0.5)
    assert r["recall@0.20"][0] == pytest.approx(1 / 3)
    assert r["lift@0.20"][0] == pytest.approx(0.5 / 0.3)
    assert r["dollars@0.20"][0] == pytest.approx(60 / 100)


def test_ties_are_broken_by_tie_key():
    score = np.array([1.0, 1.0, 1.0, 0.0])
    y = np.array([0, 0, 1, 0])
    # capacity 25% = 1 member. With the positive row holding the lowest key it is picked.
    first = M.ranking_metrics(score, y, np.ones(4), np.array([2, 1, 0, 3]), ks=(0.25,))
    last = M.ranking_metrics(score, y, np.ones(4), np.array([0, 1, 2, 3]), ks=(0.25,))
    assert first["recall@0.25"][0] == 1.0
    assert last["recall@0.25"][0] == 0.0
    assert list(M.rank_order(score, np.array([2, 1, 0, 3]))) == [2, 1, 0, 3]


def test_weights_equal_the_expanded_resample(data):
    score, y, cost, tie = data
    n = len(y)
    w = np.random.default_rng(1).multinomial(n, np.full(n, 1 / n), size=4)
    weighted = M.ranking_metrics(score, y, cost, tie, weights=w)
    for b in range(4):
        idx = np.repeat(np.arange(n), w[b])
        expanded = M.ranking_metrics(score[idx], y[idx], cost[idx], tie[idx] * n + np.arange(n))
        for name, values in weighted.items():
            assert values[b] == pytest.approx(expanded[name][0], abs=1e-10), name
        assert weighted["auroc"][b] == pytest.approx(roc_auc_score(y[idx], score[idx]), abs=1e-10)


def test_brier_and_ece():
    rng = np.random.default_rng(2)
    n = 3000
    p = rng.random(n)  # continuous, so bins do not depend on tie order
    y = (rng.random(n) < p).astype(int)
    tie = np.arange(n)
    out = M.probability_metrics(p, y, tie)
    assert out["brier"][0] == pytest.approx(brier_score_loss(y, p), abs=1e-12)
    parts = np.array_split(np.argsort(p), 10)
    by_hand = sum(len(i) / n * abs(p[i].mean() - y[i].mean()) for i in parts)
    assert out["ece"][0] == pytest.approx(by_hand, abs=1e-12)

    w = rng.multinomial(n, np.full(n, 1 / n), size=3)
    weighted = M.probability_metrics(p, y, tie, weights=w)
    for b in range(3):
        idx = np.repeat(np.arange(n), w[b])
        parts = np.array_split(idx[np.argsort(p[idx], kind="stable")], 10)
        by_hand = sum(len(i) / n * abs(p[i].mean() - y[i].mean()) for i in parts)
        assert weighted["ece"][b] == pytest.approx(by_hand, abs=1e-10)
        assert weighted["brier"][b] == pytest.approx(np.mean((p[idx] - y[idx]) ** 2), abs=1e-12)


def test_ece_by_hand_two_bins():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1])
    # bins: {0.1, 0.1} observed 0.5, gap 0.4. {0.9, 0.9} observed 1.0, gap 0.1. Mean 0.25.
    assert M.probability_metrics(p, y, np.arange(4), bins=2)["ece"][0] == pytest.approx(0.25)


def test_calibration_line_recovers_a_known_distortion():
    rng = np.random.default_rng(3)
    true_logit = rng.normal(-2, 1.5, 200_000)
    y = (rng.random(len(true_logit)) < 1 / (1 + np.exp(-true_logit))).astype(int)
    overconfident = 1 / (1 + np.exp(-(2 * true_logit + 1)))  # logit(p) = 2 * truth + 1
    line = M.calibration_line(overconfident, y, eps=1e-12)
    assert line["slope"][0] == pytest.approx(0.5, abs=0.02)
    assert line["intercept"][0] == pytest.approx(-0.5, abs=0.03)


def test_member_resamples_keep_member_count():
    counts = M.member_resamples(50, 20, seed=42)
    assert counts.shape == (20, 50)
    assert (counts.sum(axis=1) == 50).all()
    assert (counts == M.member_resamples(50, 20, seed=42)).all()


def test_interval_and_rank_helpers():
    iv = M.interval(0.5, np.arange(101) / 100)
    assert iv == {"v": 0.5, "lo": pytest.approx(0.025), "hi": pytest.approx(0.975)}
    score = np.array([3.0, 1.0, 2.0, 0.0])
    assert list(M.rank_pct(score, np.arange(4))) == [0.0, 0.5, 0.25, 0.75]
    assert list(M.top_k_mask(score, np.arange(4), 0.5)) == [True, False, True, False]
