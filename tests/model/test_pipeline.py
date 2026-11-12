"""The whole pipeline in fast mode on a small mart, written to temporary folders."""

from __future__ import annotations

import json

from riskmodel import config

from . import site_schema


def test_fast_run_writes_every_artifact(fast_run):
    model = fast_run.base / "model"
    reports = fast_run.base / "reports"
    for name in (
        "model.txt",
        "calibrator.json",
        "feature_spec.json",
        "tiers.json",
        "metadata.json",
        "logreg.json",
        "score_reference.json",
    ):
        assert (model / name).stat().st_size > 0, name
    for name in (
        "metrics.json",
        "sanity.json",
        "fairness.json",
        "drift.json",
        "capacity.json",
        "explain.json",
        "tuning.json",
        "run_summary.json",
    ):
        assert (reports / name).stat().st_size > 0, name
    meta = json.loads((model / "metadata.json").read_text())
    assert meta["fast_mode"] is True and meta["n_trees"] <= 40
    assert meta["tuning"]["n_trials"] == 0
    metrics = json.loads((reports / "metrics.json").read_text())
    assert metrics["n_bootstrap"] == 50


def test_fast_run_site_data_has_the_contract_shape(fast_run):
    # pipeline.json comes from the warehouse export, which this run does not include
    site_schema.check_all(fast_run.base / "site", expect_pipeline=False)


def test_fast_run_split_checks_pass(fast_run):
    sanity = json.loads((fast_run.base / "reports" / "sanity.json").read_text())
    assert sanity["split"]["pass"] is True
    assert sanity["single_feature_auroc"]["pass"] is True


def test_fast_mode_never_defaults_to_the_committed_folders(monkeypatch):
    for key in ("RISKMODEL_MODEL_DIR", "RISKMODEL_REPORTS_DIR", "RISKMODEL_SITE_DATA_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RISKMODEL_FAST", "1")
    fast = config.paths()
    monkeypatch.delenv("RISKMODEL_FAST")
    full = config.paths()
    assert fast.mart == full.mart
    for attr in ("model_dir", "reports_dir", "site_data_dir"):
        assert getattr(fast, attr) != getattr(full, attr)
        assert "build" in getattr(fast, attr).parts
