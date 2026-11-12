"""Shared fixtures: a small mart cut from the real one, and one fast pipeline run on it."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
MART = ROOT / "data" / "marts" / "member_year.parquet"
DICTIONARY = ROOT / "data" / "marts" / "feature_dictionary.parquet"

ENV_KEYS = (
    "RISKMODEL_MART_DIR",
    "RISKMODEL_MODEL_DIR",
    "RISKMODEL_REPORTS_DIR",
    "RISKMODEL_SITE_DATA_DIR",
    "RISKMODEL_MLRUNS_DIR",
    "RISKMODEL_FAST",
)


def write_small_mart(folder: Path, n_members: int = 700, seed: int = 0) -> Path:
    """Every row of a random set of members, plus the feature dictionary."""
    mart = pd.read_parquet(MART)
    members = np.sort(mart["member_id"].unique())
    keep = np.random.default_rng(seed).choice(members, n_members, replace=False)
    folder.mkdir(parents=True, exist_ok=True)
    mart[mart["member_id"].isin(keep)].to_parquet(folder / "member_year.parquet", index=False)
    pd.read_parquet(DICTIONARY).to_parquet(folder / "feature_dictionary.parquet", index=False)
    return folder


class Workspace:
    """Points the pipeline at temporary folders for the length of a `with` block."""

    def __init__(self, base: Path, mart_dir: Path):
        self.base = base
        self.values = {
            "RISKMODEL_MART_DIR": str(mart_dir),
            "RISKMODEL_MODEL_DIR": str(base / "model"),
            "RISKMODEL_REPORTS_DIR": str(base / "reports"),
            "RISKMODEL_SITE_DATA_DIR": str(base / "site"),
            "RISKMODEL_MLRUNS_DIR": str(base / "mlruns"),
            "RISKMODEL_FAST": "1",
        }

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in ENV_KEYS}
        os.environ.update(self.values)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="session")
def small_mart_dir(tmp_path_factory) -> Path:
    return write_small_mart(tmp_path_factory.mktemp("mart"))


@pytest.fixture(scope="session")
def fast_run(tmp_path_factory, small_mart_dir) -> Workspace:
    """The whole pipeline in fast mode on the small mart. Returns where it wrote."""
    from riskmodel import run

    ws = Workspace(tmp_path_factory.mktemp("fast_run"), small_mart_dir)
    with ws:
        assert run.main(["all"]) == 0
    return ws
