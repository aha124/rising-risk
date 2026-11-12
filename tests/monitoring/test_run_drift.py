"""The drift CLI: exit status, report contents, agreement with the project drift report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoring import run_drift
from riskmodel import config, dataset

ROOT = Path(__file__).resolve().parents[2]
MART = ROOT / "data" / "marts" / "member_year.parquet"
TRAIN_YEARS = "2015-2021"


def run(tmp_path, *extra, current=MART, years="2025"):
    out = tmp_path / "report.json"
    argv = ["--reference", str(MART), "--reference-years", TRAIN_YEARS]
    argv += ["--current", str(current), "--output", str(out)]
    if years:
        argv += ["--current-years", years]
    code = run_drift.main([*argv, *extra])
    return code, (json.loads(out.read_text()) if out.exists() else None)


def test_parse_years():
    assert run_drift.parse_years("2015-2017") == [2015, 2016, 2017]
    assert run_drift.parse_years("2024,2025") == [2024, 2025]
    assert run_drift.parse_years(None) is None


def test_quiet_year_exits_zero(tmp_path, capsys):
    code, report = run(tmp_path)
    assert code == 0
    assert report["alerts"] == [] and report["failed"] is False
    assert len(report["features"]) == 59
    assert report["thresholds"] == {"watch": config.PSI_WATCH, "alert": config.PSI_ALERT}
    assert "12 months" in report["label_lag_note"]
    assert "model score" in capsys.readouterr().out


def test_shifted_year_exits_one_and_names_the_feature(tmp_path):
    """Index year 2023 looks back on the 2021 vaccination encounters."""
    code, report = run(tmp_path, years="2023")
    assert code == 1 and report["failed"] is True
    assert report["alerts"] == ["outpatient_visits_13_24m"]
    assert report["features"][0]["feature"] == "outpatient_visits_13_24m"


def test_cli_agrees_with_the_project_drift_report(tmp_path):
    project = dataset.read_json(config.paths().reports_dir / "drift.json")
    for year in (2023, 2025):
        _, report = run(tmp_path, years=str(year))
        entry = next(e for e in project["by_year"] if e["year"] == year)
        assert report["score"]["psi"] == pytest.approx(entry["score_psi"], abs=1e-12)
        mine = {r["feature"]: r["psi"] for r in report["features"]}
        for top in entry["top_features"]:
            assert mine[top["feature"]] == pytest.approx(top["psi"], abs=1e-12)


def test_fail_on_controls_the_exit_status(tmp_path):
    assert run(tmp_path, "--fail-on", "never", years="2023")[0] == 0
    # 2022 and 2023 both have features between the watch and alert lines
    code, report = run(tmp_path, "--fail-on", "watch", years="2023")
    assert code == 1 and report["watch"]


def test_a_lab_that_stops_arriving_raises_an_alert(tmp_path):
    current = pd.read_parquet(MART)
    current = current[current["index_year"] == 2025].copy()
    current["hba1c"] = float("nan")
    path = tmp_path / "current.parquet"
    current.to_parquet(path, index=False)
    code, report = run(tmp_path, current=path, years=None)
    assert code == 1 and "hba1c" in report["alerts"]


def test_bad_input_exits_two(tmp_path, capsys):
    assert run(tmp_path, current=tmp_path / "missing.parquet")[0] == 2
    narrow = tmp_path / "narrow.parquet"
    pd.DataFrame({"age": [40, 50]}).to_parquet(narrow, index=False)
    assert run(tmp_path, current=narrow, years=None)[0] == 2
    assert run(tmp_path, years="1999")[0] == 2
    assert "error" in capsys.readouterr().err
