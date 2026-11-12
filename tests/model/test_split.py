"""The split is what it claims to be, and fitting cannot see what it must not see."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmodel import config, dataset, train

from .conftest import Workspace, write_small_mart


@pytest.fixture(scope="module")
def frame():
    return dataset.load_mart()


def test_year_plan_has_no_overlap():
    plan = dataset.year_plan(range(2015, 2026))
    assert plan.train_years == tuple(range(2015, 2022))
    assert (plan.tune_year, plan.calib_year, plan.test_years) == (2022, 2023, (2024, 2025))
    groups = [set(plan.train_years), {plan.tune_year}, {plan.calib_year}, set(plan.test_years)]
    assert sum(len(g) for g in groups) == len(set().union(*groups)) == 11
    assert max(plan.train_years) < plan.tune_year < plan.calib_year < min(plan.test_years)
    with pytest.raises(ValueError):
        dataset.year_plan([2020, 2021, 2022])


def test_holdout_hash_is_stable_and_near_20_percent(frame):
    assert dataset.is_holdout("abc") == dataset.is_holdout("abc")
    # fixed values, so a change to the hash is caught
    assert dataset.holdout_fraction("member-1") == pytest.approx(0.5940244, abs=1e-6)
    share = frame.groupby("member_id")["holdout"].first().mean()
    assert 0.17 < share < 0.23
    # a member is in or out for every one of their years
    assert (frame.groupby("member_id")["holdout"].nunique() == 1).all()


def test_fit_rows_contain_no_holdout_member_and_no_test_year(frame):
    plan = dataset.year_plan(frame["index_year"].unique())
    holdout_members = set(frame.loc[frame["holdout"], "member_id"])
    seen_years = set()
    for split in ("train", "tune", "calibrate"):
        rows = dataset.fit_rows(frame, split)
        assert len(rows) > 0
        assert not (set(rows["member_id"]) & holdout_members)
        assert not (set(rows["index_year"]) & set(plan.test_years))
        assert not (set(rows["index_year"]) & seen_years)  # no year in two fitting splits
        seen_years |= set(rows["index_year"])
    with pytest.raises(ValueError):
        dataset.fit_rows(frame, "test")


def test_every_row_has_exactly_one_role(frame):
    assert set(frame["role"]) == {"fit", "holdout", "test"}
    test = frame[frame["role"] == "test"]
    assert set(test["index_year"]) == {2024, 2025}
    assert test["holdout"].any() and (~test["holdout"]).any()
    # "seen" means the member has a fit row. Holdout members never do.
    assert not test.loc[test["holdout"], "seen"].any()


def test_features_exclude_keys_audit_and_outcome_columns(frame):
    features = dataset.feature_names(dataset.load_feature_dictionary(), frame)
    assert len(features) == 59
    assert not (set(features) & set(config.FORBIDDEN_AS_FEATURES))
    X = dataset.feature_matrix(frame.head(50), features)
    assert X.shape == (50, 59) and X.dtype == np.float64


@pytest.mark.parametrize("column", config.FORBIDDEN_AS_FEATURES)
def test_forbidden_columns_cannot_be_requested(frame, column):
    with pytest.raises(ValueError, match="not allowed"):
        dataset.feature_matrix(frame.head(5), ["age", column])
    bad = pd.DataFrame({"name": ["age", column]})
    with pytest.raises(ValueError, match="not allowed"):
        dataset.feature_names(bad, frame)


def test_saved_model_uses_the_dictionary_features_only():
    spec = dataset.read_json(config.paths().model_dir / "feature_spec.json")
    names = [f["name"] for f in spec["features"]]
    assert not (set(names) & set(config.FORBIDDEN_AS_FEATURES))
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(config.paths().model_dir / "model.txt"))
    assert booster.feature_name() == names


def _fit_files(ws: Workspace) -> dict:
    with ws:
        train.run()
    model = ws.base / "model"
    return {
        name: (model / name).read_text()
        for name in ("model.txt", "calibrator.json", "tiers.json", "logreg.json")
    }


def test_fitting_ignores_holdout_members_test_years_and_outcome_columns(tmp_path):
    """Train twice. The second time, everything fitting must not read is scrambled:
    every feature and label of holdout members and of test-year rows, and next-year
    cost, death, pregnancy in the outcome year and the audit columns of every row. The fitted trees, the calibrator,
    the tier cut points and the logistic baseline must come out identical.
    """
    clean_dir = write_small_mart(tmp_path / "clean", n_members=500, seed=1)
    before = _fit_files(Workspace(tmp_path / "a", clean_dir))

    mart = pd.read_parquet(clean_dir / "member_year.parquet")
    dictionary = pd.read_parquet(clean_dir / "feature_dictionary.parquet")
    features = [str(n) for n in dictionary["name"]]
    rng = np.random.default_rng(7)
    plan = dataset.year_plan(mart["index_year"].unique())
    hidden = mart["member_id"].map(dataset.is_holdout) | mart["index_year"].isin(plan.test_years)
    assert hidden.sum() > 500 and (~hidden).sum() > 500
    for name in features:
        mart[name] = mart[name].astype(float)
        mart.loc[hidden, name] = rng.normal(1e6, 1e6, int(hidden.sum()))
    mart[config.LABEL] = mart[config.LABEL].astype("int64")
    mart.loc[hidden, config.LABEL] = rng.integers(0, 2, int(hidden.sum()))
    mart[config.COST_NEXT] = rng.exponential(5e4, len(mart))
    mart["died_in_outcome_year"] = rng.integers(0, 2, len(mart))
    mart[config.PREGNANCY_IN_OUTCOME_YEAR] = rng.integers(0, 2, len(mart))
    mart["index_date"] = "1900-01-01"
    mart["race"] = rng.choice(["x", "y"], len(mart))
    mart["ethnicity"] = rng.choice(["x", "y"], len(mart))
    scrambled_dir = tmp_path / "scrambled"
    scrambled_dir.mkdir()
    mart.to_parquet(scrambled_dir / "member_year.parquet", index=False)
    dictionary.to_parquet(scrambled_dir / "feature_dictionary.parquet", index=False)

    after = _fit_files(Workspace(tmp_path / "b", scrambled_dir))
    for name in before:
        assert before[name] == after[name], f"{name} changed, so fitting read hidden data"


def test_fitting_does_react_to_fit_rows(tmp_path):
    """The mirror of the test above, so that one cannot pass by accident: change the
    labels of fit rows and the model must change."""
    clean_dir = write_small_mart(tmp_path / "clean", n_members=500, seed=1)
    before = _fit_files(Workspace(tmp_path / "a", clean_dir))
    mart = pd.read_parquet(clean_dir / "member_year.parquet")
    plan = dataset.year_plan(mart["index_year"].unique())
    fit = ~mart["member_id"].map(dataset.is_holdout) & mart["index_year"].isin(plan.train_years)
    mart.loc[fit, config.LABEL] = np.random.default_rng(3).permutation(
        mart.loc[fit, config.LABEL].to_numpy()
    )
    other = tmp_path / "changed"
    other.mkdir()
    mart.to_parquet(other / "member_year.parquet", index=False)
    pd.read_parquet(clean_dir / "feature_dictionary.parquet").to_parquet(
        other / "feature_dictionary.parquet", index=False
    )
    after = _fit_files(Workspace(tmp_path / "b", other))
    assert before["model.txt"] != after["model.txt"]
