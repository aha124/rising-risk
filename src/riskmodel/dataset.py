"""Load the member-year mart and decide which rows each step may see.

The split has two parts. Time: the last two index years are the test set, the year
before them fits the calibrator, the year before that picks hyperparameters, and
everything earlier trains. Members: 20% of members, chosen by a hash of the member
id, are kept out of training, tuning and calibration in every year. They only ever
appear in evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class YearPlan:
    train_years: tuple[int, ...]
    tune_year: int
    calib_year: int
    test_years: tuple[int, ...]

    def split_of(self, year: int) -> str:
        if year in self.test_years:
            return "test"
        if year == self.calib_year:
            return "calibrate"
        if year == self.tune_year:
            return "tune"
        return "train"


def year_plan(years) -> YearPlan:
    """Last two years test, then calibrate, then tune, the rest train."""
    years = sorted(int(y) for y in set(years))
    if len(years) < 5:
        raise ValueError(f"need at least 5 index years to split by time, got {years}")
    return YearPlan(
        train_years=tuple(years[:-4]),
        tune_year=years[-4],
        calib_year=years[-3],
        test_years=tuple(years[-2:]),
    )


def holdout_fraction(member_id: str) -> float:
    """A number in [0, 1) that depends only on the member id."""
    digest = hashlib.sha256(str(member_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def is_holdout(member_id: str) -> bool:
    return holdout_fraction(member_id) < config.HOLDOUT_SHARE


def load_feature_dictionary(path: Path | None = None) -> pd.DataFrame:
    path = path or config.paths().feature_dictionary
    return pd.read_parquet(path)


def feature_names(dictionary: pd.DataFrame, mart: pd.DataFrame) -> list[str]:
    """Model inputs, in the order the feature dictionary lists them.

    The dictionary is the only source of feature names. Anything else in the mart
    (keys, race, ethnicity, next-year cost, the label, death in the outcome year)
    stays out, and this raises if the dictionary ever names one of those.
    """
    names = [str(n) for n in dictionary["name"]]
    if len(set(names)) != len(names):
        raise ValueError("feature dictionary has duplicate names")
    banned = sorted(set(names) & set(config.FORBIDDEN_AS_FEATURES))
    if banned:
        raise ValueError(f"not allowed as model inputs: {banned}")
    missing = [n for n in names if n not in mart.columns]
    if missing:
        raise ValueError(f"features missing from the mart: {missing}")
    non_numeric = [n for n in names if not pd.api.types.is_numeric_dtype(mart[n])]
    if non_numeric:
        raise ValueError(f"features must be numeric: {non_numeric}")
    return names


def load_mart(path: Path | None = None) -> pd.DataFrame:
    """The mart with split columns added, sorted by year then member."""
    path = path or config.paths().mart
    df = pd.read_parquet(path)
    df["member_id"] = df["member_id"].astype(str)
    df["index_year"] = df["index_year"].astype(int)
    df = df.sort_values(["index_year", "member_id"], kind="stable").reset_index(drop=True)
    if df.duplicated(["member_id", "index_year"]).any():
        raise ValueError("mart has more than one row for a member-year")
    return add_split_columns(df)


def add_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `holdout`, `split`, `role`, `seen` and `tie_key`.

    split  train, tune, calibrate or test, from the index year alone.
    role   "fit" for rows a fitting step may read, "holdout" for holdout members
           before the test years, "test" for every row in the test years.
    seen   test rows only: the member has at least one fit row.
    tie_key  rank of (member_id, index_year), used to break score ties the same way everywhere.
    """
    plan = year_plan(df["index_year"].unique())
    members = pd.Series(df["member_id"].unique())
    flag = dict(zip(members, members.map(is_holdout), strict=True))
    df = df.copy()
    df["holdout"] = df["member_id"].map(flag).astype(bool)
    df["split"] = df["index_year"].map(plan.split_of)
    df["role"] = np.where(df["split"] == "test", "test", np.where(df["holdout"], "holdout", "fit"))
    fit_members = set(df.loc[df["role"] == "fit", "member_id"])
    df["seen"] = df["member_id"].isin(fit_members)
    order = np.lexsort((df["index_year"].to_numpy(), df["member_id"].to_numpy().astype("U")))
    tie = np.empty(len(df), dtype=np.int64)
    tie[order] = np.arange(len(df))
    df["tie_key"] = tie
    return df


def fit_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows a fitting step may read for one of train, tune or calibrate."""
    if split not in ("train", "tune", "calibrate"):
        raise ValueError(f"{split!r} is not a fitting split")
    return df[(df["split"] == split) & (df["role"] == "fit")]


def feature_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Float matrix of model inputs, missing values as NaN."""
    banned = sorted(set(features) & set(config.FORBIDDEN_AS_FEATURES))
    if banned:
        raise ValueError(f"not allowed as model inputs: {banned}")
    return df[features].to_numpy(dtype=np.float64, na_value=np.nan)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    """Short commit hash of the working tree, or "unknown" outside a git checkout."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_json(path: Path, obj, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    else:
        text = json.dumps(obj, indent=2, allow_nan=False)
    path.write_text(text + ("" if compact else "\n"))


def read_json(path: Path):
    return json.loads(Path(path).read_text())
