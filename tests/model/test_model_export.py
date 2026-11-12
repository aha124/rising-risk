"""Calibrator round trip, tree export and the saved model's scoring path."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from riskmodel import config, dataset
from riskmodel.export_browser_model import export_tree, raw_score, with_extra_nulls
from riskmodel.scoring import Scorer, apply_calibrator, load_scored, sigmoid
from riskmodel.train import fit_calibrator

ROOT = Path(__file__).resolve().parents[2]


def test_calibrator_round_trip_through_json():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p**2).astype(int)
    x, yy = fit_calibrator(p, y)
    saved = json.loads(json.dumps({"x": x.tolist(), "y": yy.tolist()}))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    grid = np.concatenate([np.linspace(-0.2, 1.2, 2001), p[:500]])
    ours = apply_calibrator(grid, np.array(saved["x"]), np.array(saved["y"]))
    assert np.max(np.abs(ours - iso.predict(grid))) < 1e-12
    assert (np.diff(saved["y"]) >= 0).all() and min(saved["y"]) >= 0 and max(saved["y"]) <= 1


def test_saved_calibrator_matches_the_scorer():
    paths = config.paths()
    scorer = Scorer.load(paths.model_dir)
    cal = dataset.read_json(paths.model_dir / "calibrator.json")
    assert cal["input"] == "probability"
    raw = np.linspace(-8, 4, 500)
    expected = np.interp(sigmoid(raw), cal["x"], cal["y"])
    assert np.array_equal(scorer.calibrate(raw), expected)
    assert (np.diff(scorer.calibrate(raw)) >= 0).all()  # calibration never reorders members
    tiers = dataset.read_json(paths.model_dir / "tiers.json")
    assert 0 < tiers["rising"] < tiers["high"] < 1
    assert scorer.tier(tiers["high"]) == "high"
    assert scorer.tier(tiers["rising"]) == "rising"
    assert scorer.tier(tiers["rising"] - 1e-9) == "routine"


def test_tree_export_follows_lightgbm_on_missing_values():
    """One feature has missing values in training (splits get missing_type NaN), the
    others have none (missing_type None, where LightGBM reads a null as 0.0)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 4))
    X[:, 1] = rng.integers(-2, 3, 4000)  # thresholds on both sides of zero
    y = ((X[:, 0] + X[:, 1] - X[:, 2] + rng.normal(0, 0.5, 4000)) > 0).astype(int)
    X[rng.random(4000) < 0.3, 0] = np.nan
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 8, "verbose": -1, "seed": 1, "num_threads": 1},
        lgb.Dataset(X, y),
        num_boost_round=30,
    )
    dump = booster.dump_model()
    kinds = set()

    def walk(node):
        if "leaf_value" not in node:
            kinds.add(node["missing_type"])
            walk(node["left_child"])
            walk(node["right_child"])

    for t in dump["tree_info"]:
        walk(t["tree_structure"])
    assert kinds == {"NaN", "None"}

    trees = [export_tree(t["tree_structure"]) for t in dump["tree_info"]]
    probe = with_extra_nulls(X[:600], rng, 600)  # a null in a random feature of every row
    expected = booster.predict(probe, raw_score=True)
    got = [raw_score(trees, [None if np.isnan(v) else float(v) for v in row]) for row in probe]
    assert np.max(np.abs(np.array(got) - expected)) < 1e-12


def test_published_browser_model_matches_the_saved_model():
    site = config.paths().site_data_dir
    model = dataset.read_json(site / "model_browser.json")
    parity = dataset.read_json(site / "parity.json")
    scorer = Scorer.load()
    assert [f["name"] for f in model["features"]] == scorer.features
    assert model["n_trees"] == len(model["trees"]) == scorer.booster.num_trees()
    assert model["calibrator"]["x"] == scorer.calib_x.tolist()  # full precision, no rounding
    assert model["calibrator"]["y"] == scorer.calib_y.tolist()
    X = np.array([[np.nan if v is None else v for v in row] for row in parity["rows"]])
    assert len(X) == 200 and np.isnan(X).any(axis=1).sum() >= 20
    raw = scorer.raw(X)
    assert np.max(np.abs(raw - np.array(parity["raw"]))) < 1e-12
    assert np.max(np.abs(scorer.calibrate(raw) - np.array(parity["p"]))) < 1e-12
    walked = [raw_score(model["trees"], row, model["init_score"]) for row in parity["rows"]]
    assert np.max(np.abs(np.array(walked) - raw)) < 1e-9


def test_node_parity_script_passes():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    out = subprocess.run(
        [node, str(ROOT / "docs" / "tests" / "parity.mjs")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout and "fixture" not in out.stdout


def test_exemplar_rows_reproduce_their_scores():
    explain = dataset.read_json(config.paths().site_data_dir / "explain.json")
    scorer = Scorer.load()
    for ex in explain["exemplars"]:
        row = np.array([[np.nan if v is None else v for v in ex["row"]]])
        p = float(scorer.probability(row)[0])
        assert p == pytest.approx(ex["p"], rel=1e-4)
        assert scorer.tier(p) == ex["tier"]
        total = ex["other_contrib"] + sum(c["contrib"] for c in ex["contribs"])
        assert explain["base_value"] + total == pytest.approx(float(scorer.raw(row)[0]), abs=1e-3)


def test_headline_can_be_rederived_from_the_parquet_and_the_saved_model():
    """An independent path to recall at 10%: pandas sort, no package metric code."""
    import pandas as pd

    paths = config.paths()
    mart = pd.read_parquet(paths.mart)
    names = [
        f["name"] for f in dataset.read_json(paths.model_dir / "feature_spec.json")["features"]
    ]
    test = mart[mart["index_year"].isin([2024, 2025])].copy()
    booster = lgb.Booster(model_file=str(paths.model_dir / "model.txt"))
    test["score"] = booster.predict(
        test[names].to_numpy(dtype=float, na_value=np.nan), raw_score=True
    )
    k = round(0.10 * len(test))
    positives = test["label_high_cost"].sum()

    def recall(column):
        top = test.sort_values([column, "member_id"], ascending=[False, True]).head(k)
        return top["label_high_cost"].sum() / positives

    metrics = dataset.read_json(paths.reports_dir / "metrics.json")
    for key, column in (("lgbm", "score"), ("prior_cost", "cost_prior_12m")):
        reported = metrics["models"][key]["at"]["0.10"]["recall"]["v"]
        assert recall(column) == pytest.approx(reported, abs=2 / positives), key


def test_scored_frame_is_consistent():
    sc = load_scored()
    f = sc.frame
    assert np.array_equal(f["p"].to_numpy(), sc.scorer.calibrate(f["raw"].to_numpy()))
    assert np.array_equal(f["prior_cost"].to_numpy(), f[config.PRIOR_COST].to_numpy(dtype=float))
    assert f["tie_key"].is_unique
