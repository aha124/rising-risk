"""Paths and settings for the modeling pipeline.

Everything is read at call time, so a test or a CI job can point the pipeline at a
different mart or output folder with environment variables:

    RISKMODEL_MART_DIR       folder holding member_year.parquet and feature_dictionary.parquet
    RISKMODEL_MODEL_DIR      where model files go            (default artifacts/model)
    RISKMODEL_REPORTS_DIR    where metric reports go         (default artifacts/reports)
    RISKMODEL_SITE_DATA_DIR  where the site JSON goes        (default docs/data)
    RISKMODEL_MLRUNS_DIR     MLflow file store               (default mlruns)
    RISKMODEL_FAST=1         few trees, no Optuna, 50 bootstrap resamples

A fast run is a smoke test, not a result. Unless the folders above are set, it
writes under build/fast/ so it can never replace the committed model, reports or
site data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SEED = 42
MODEL_VERSION = "2026.11.0"

HOLDOUT_SHARE = 0.20

KEY_COLUMNS = ("member_id", "index_date", "index_year")
AUDIT_COLUMNS = ("race", "ethnicity")
PREGNANCY_IN_OUTCOME_YEAR = "pregnancy_in_outcome_year"
OUTCOME_COLUMNS = (
    "cost_next_12m",
    "label_high_cost",
    "died_in_outcome_year",
    PREGNANCY_IN_OUTCOME_YEAR,
)
# Nothing in this tuple may ever be a model input.
FORBIDDEN_AS_FEATURES = KEY_COLUMNS + AUDIT_COLUMNS + OUTCOME_COLUMNS

LABEL = "label_high_cost"
COST_NEXT = "cost_next_12m"
PRIOR_COST = "cost_prior_12m"
PREGNANCY = "pregnant_at_index"
CHRONIC_COUNT = "chronic_group_count"
SEX_FEMALE = "sex_female"
AGE = "age"
# Oldest age band with a meaningful share of pregnancies in this data.
CHILDBEARING_MAX_AGE = 49

K_GRID = (0.01, 0.05, 0.10, 0.20)
HEADLINE_K = 0.10

PSI_WATCH = 0.10
PSI_ALERT = 0.25

TIER_HIGH_SHARE = 0.05
TIER_RISING_SHARE = 0.15

SMALL_GROUP_ROWS = 300
SMALL_GROUP_POSITIVES = 30

MAX_BROWSER_MODEL_BYTES = 2_500_000


@dataclass(frozen=True)
class Paths:
    mart_dir: Path
    model_dir: Path
    reports_dir: Path
    site_data_dir: Path
    mlruns_dir: Path

    @property
    def mart(self) -> Path:
        return self.mart_dir / "member_year.parquet"

    @property
    def feature_dictionary(self) -> Path:
        return self.mart_dir / "feature_dictionary.parquet"


def _dir(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).resolve() if value else default


def paths() -> Paths:
    fast = os.environ.get("RISKMODEL_FAST", "") == "1"
    out = ROOT / "build" / "fast" if fast else ROOT
    return Paths(
        mart_dir=_dir("RISKMODEL_MART_DIR", ROOT / "data" / "marts"),
        model_dir=_dir("RISKMODEL_MODEL_DIR", out / "artifacts" / "model"),
        reports_dir=_dir("RISKMODEL_REPORTS_DIR", out / "artifacts" / "reports"),
        site_data_dir=_dir("RISKMODEL_SITE_DATA_DIR", out / "docs" / "data"),
        mlruns_dir=_dir("RISKMODEL_MLRUNS_DIR", ROOT / "mlruns"),
    )


@dataclass(frozen=True)
class Settings:
    fast: bool
    n_bootstrap: int
    n_trials: int
    max_trees: int
    early_stopping_rounds: int
    max_leaves: int = 31


def settings() -> Settings:
    fast = os.environ.get("RISKMODEL_FAST", "") == "1"
    if fast:
        return Settings(
            fast=True, n_bootstrap=50, n_trials=0, max_trees=40, early_stopping_rounds=10
        )
    return Settings(
        fast=False, n_bootstrap=1000, n_trials=30, max_trees=400, early_stopping_rounds=50
    )
