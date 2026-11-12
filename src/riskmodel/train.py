"""Fit the logistic baseline, the gradient boosted model and its calibrator.

Only rows with role "fit" are read here: non-holdout members, training years for
the trees, the tune year for early stopping and hyperparameters, the calibration
year for the isotonic calibrator and the tier cut points. Test years and holdout
members are never read for fitting.
"""

from __future__ import annotations

import datetime as dt
import platform

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config, dataset, tracking
from .baselines import fit_logreg, log_feature_mask
from .scoring import apply_calibrator, sigmoid

LOGREG_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

# Used as is in fast mode, and as the fixed part of every Optuna trial.
BASE_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": config.SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}


def fit_lgbm(X_train, y_train, X_tune, y_tune, params: dict, features: list[str]):
    """Train with early stopping on tune-year average precision.

    Returns (booster cut at its best iteration, best iteration, tune-year AUPRC).
    No LightGBM categorical features are used, so every split is `value <= threshold`.
    """
    s = config.settings()
    params = {**BASE_PARAMS, **params}
    if params["num_leaves"] > s.max_leaves:
        raise ValueError(f"num_leaves is capped at {s.max_leaves}")
    train = lgb.Dataset(X_train, y_train, feature_name=features, categorical_feature=[])
    tune = lgb.Dataset(X_tune, y_tune, reference=train)
    booster = lgb.train(
        params,
        train,
        num_boost_round=s.max_trees,
        valid_sets=[tune],
        valid_names=["tune"],
        callbacks=[lgb.early_stopping(s.early_stopping_rounds, verbose=False)],
    )
    best = booster.best_iteration or booster.current_iteration()
    booster = lgb.Booster(model_str=booster.model_to_string(num_iteration=best))
    auprc = float(average_precision_score(y_tune, booster.predict(X_tune, raw_score=True)))
    return booster, int(best), auprc


def suggest_params(trial) -> dict:
    """Search space. The 31-leaf and 400-tree caps also bound the browser model's size."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, config.settings().max_leaves),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
    }


def tune_lgbm(X_train, y_train, X_tune, y_tune, features, log=None) -> tuple[dict, list[dict]]:
    """Optuna TPE with a fixed seed. Returns the best parameters and every trial."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[dict] = []

    def objective(trial):
        params = suggest_params(trial)
        booster, best, auprc = fit_lgbm(X_train, y_train, X_tune, y_tune, params, features)
        leaves = int(sum(t["num_leaves"] for t in booster.dump_model()["tree_info"]))
        trials.append(
            {"number": trial.number, "tune_auprc": auprc, "n_trees": best, "n_leaves": leaves}
            | params
        )
        if log is not None:
            log.metrics({"trial_tune_auprc": auprc, "trial_n_trees": best}, step=trial.number)
        return auprc

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED)
    )
    study.optimize(objective, n_trials=config.settings().n_trials)
    return dict(study.best_params), trials


def choose_logreg(X_train, y_train, X_tune, y_tune, log_mask):
    """Fit one model per C on the training years, keep the best tune-year AUPRC."""
    grid = []
    best = None
    for C in LOGREG_C_GRID:
        model = fit_logreg(X_train, y_train, log_mask, C)
        auprc = float(average_precision_score(y_tune, model.predict_proba(X_tune)))
        grid.append({"C": C, "tune_auprc": auprc})
        if best is None or auprc > best[1]:
            best = (model, auprc)
    return best[0], grid


def fit_calibrator(p_raw: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic regression of the outcome on the uncalibrated probability."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_raw, y)
    return iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float)


def feature_spec(frame: pd.DataFrame, dictionary: pd.DataFrame, features: list[str]) -> dict:
    """Per-feature label, unit and summary numbers from the training rows."""
    train = dataset.fit_rows(frame, "train")
    rows = []
    for name in features:
        d = dictionary[dictionary["name"] == name].iloc[0]
        col = train[name].to_numpy(dtype=float, na_value=np.nan)
        ok = col[~np.isnan(col)]
        has = len(ok) > 0
        q = np.percentile(ok, [0, 1, 50, 99, 100]) if has else [None] * 5
        rows.append(
            {
                "name": name,
                "label": str(d["label"]),
                "group": str(d["group"]),
                "unit": str(d["unit"]),
                "type": "binary" if d["dtype"] == "binary" else "numeric",
                "dtype": str(d["dtype"]),
                "description": str(d["description"]),
                "min": _f(q[0]),
                "p01": _f(q[1]),
                "median": _f(q[2]),
                "p99": _f(q[3]),
                "max": _f(q[4]),
                "missing_rate": float(np.isnan(col).mean()) if len(col) else 0.0,
                "integer_valued": bool(has and np.all(ok == np.round(ok))),
                "nullable": bool(frame[name].isna().any()),
            }
        )
    return {"features": rows}


def _f(x):
    return None if x is None else float(x)


def run() -> dict:
    paths = config.paths()
    s = config.settings()
    frame = dataset.load_mart(paths.mart)
    dictionary = dataset.load_feature_dictionary(paths.feature_dictionary)
    features = dataset.feature_names(dictionary, frame)
    plan = dataset.year_plan(frame["index_year"].unique())

    parts = {}
    for split in ("train", "tune", "calibrate"):
        rows = dataset.fit_rows(frame, split)
        if rows["holdout"].any() or (rows["split"] == "test").any():
            raise AssertionError("a holdout or test row reached a fitting step")
        parts[split] = (
            dataset.feature_matrix(rows, features),
            rows[config.LABEL].to_numpy(dtype=int),
        )
    (X_tr, y_tr), (X_tu, y_tu), (X_ca, y_ca) = parts["train"], parts["tune"], parts["calibrate"]
    data_hash = dataset.file_sha256(paths.mart)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    # Logistic regression baseline.
    units = [str(dictionary.loc[dictionary["name"] == f, "unit"].iloc[0]) for f in features]
    with tracking.run("logreg", {"C_grid": LOGREG_C_GRID}) as log:
        logreg, c_grid = choose_logreg(X_tr, y_tr, X_tu, y_tu, log_feature_mask(units))
        dataset.write_json(paths.model_dir / "logreg.json", logreg.to_json())
        log.params({"C": logreg.C})
        log.metrics({"tune_auprc": max(g["tune_auprc"] for g in c_grid)})
        log.artifact(paths.model_dir / "logreg.json")

    # Gradient boosted trees.
    trials: list[dict] = []
    with tracking.run("lgbm", {"n_trials": s.n_trials, "max_trees": s.max_trees}) as log:
        if s.n_trials > 0:
            best_params, trials = tune_lgbm(X_tr, y_tr, X_tu, y_tu, features, log)
        else:
            best_params = {}
        booster, n_trees, tune_auprc = fit_lgbm(X_tr, y_tr, X_tu, y_tu, best_params, features)
        params = {**BASE_PARAMS, **best_params}
        booster.save_model(str(paths.model_dir / "model.txt"))

        raw_tune = booster.predict(X_tu, raw_score=True)
        raw_cal = booster.predict(X_ca, raw_score=True)
        cx, cy = fit_calibrator(sigmoid(raw_cal), y_ca)
        p_cal = apply_calibrator(sigmoid(raw_cal), cx, cy)
        dataset.write_json(
            paths.model_dir / "calibrator.json",
            {
                "kind": "isotonic",
                "input": "probability",
                "note": "x is the uncalibrated probability (sigmoid of the raw score). "
                "Interpolate linearly between points and clip outside them.",
                "fit_on": f"index year {plan.calib_year}, non-holdout members",
                "n_fit": len(y_ca),
                "x": cx.tolist(),
                "y": cy.tolist(),
            },
        )
        high = float(np.quantile(p_cal, 1 - config.TIER_HIGH_SHARE))
        rising = float(np.quantile(p_cal, 1 - config.TIER_HIGH_SHARE - config.TIER_RISING_SHARE))
        dataset.write_json(
            paths.model_dir / "tiers.json",
            {
                "high": high,
                "rising": rising,
                "rule": "calibrated probability at or above `high` is the high tier, at or above "
                "`rising` is the rising tier, below that is routine",
                "frozen_from": f"index year {plan.calib_year}, non-holdout members",
                "target_share": {
                    "high": config.TIER_HIGH_SHARE,
                    "rising": config.TIER_RISING_SHARE,
                },
                # Isotonic output has ties, so the realized shares are not exactly 5% and 15%.
                "share_in_calibration_year": {
                    "high": float((p_cal >= high).mean()),
                    "rising": float(((p_cal >= rising) & (p_cal < high)).mean()),
                },
            },
        )

        # Reference distribution for "higher than x% of members": test-year inputs, no labels.
        test = frame[frame["split"] == "test"]
        p_test = apply_calibrator(
            sigmoid(booster.predict(dataset.feature_matrix(test, features), raw_score=True)),
            cx,
            cy,
        )
        dataset.write_json(
            paths.model_dir / "score_reference.json",
            {
                "source": f"calibrated probability, index years {list(plan.test_years)}, "
                "all members. Inputs only, no outcomes.",
                "percentiles": np.percentile(p_test, np.arange(101)).tolist(),
            },
        )

        dataset.write_json(
            paths.model_dir / "feature_spec.json", feature_spec(frame, dictionary, features)
        )
        dump = booster.dump_model()
        n_leaves = int(sum(t["num_leaves"] for t in dump["tree_info"]))
        metadata = {
            "model_version": config.MODEL_VERSION,
            "trained_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": dataset.git_sha(),
            "data_sha256": data_hash,
            "fast_mode": s.fast,
            "seed": config.SEED,
            "n_features": len(features),
            "n_trees": n_trees,
            "n_leaves": n_leaves,
            "params": {k: v for k, v in params.items() if k not in ("verbose",)},
            "tuning": {
                "n_trials": s.n_trials,
                "sampler": "optuna TPE, seed 42" if s.n_trials else "none (default parameters)",
                "selection_metric": "tune-year AUPRC",
                "tune_auprc": tune_auprc,
                "tune_auroc": _auc(y_tu, raw_tune),
            },
            "logreg": {"C": logreg.C, "grid": c_grid},
            "split": {
                "train_years": list(plan.train_years),
                "tune_year": plan.tune_year,
                "calib_year": plan.calib_year,
                "test_years": list(plan.test_years),
                "member_holdout_share": config.HOLDOUT_SHARE,
                "n_train": len(y_tr),
                "n_tune": len(y_tu),
                "n_calibrate": len(y_ca),
                "n_holdout_rows_before_test": int((frame["role"] == "holdout").sum()),
                "n_test": int((frame["split"] == "test").sum()),
            },
            "ranking": "Members are ranked by the raw score, ties broken by member id ascending. "
            "The calibrated probability is for reading, not ranking, because isotonic "
            "calibration creates ties.",
            "versions": {
                "python": platform.python_version(),
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
            },
        }
        dataset.write_json(paths.model_dir / "metadata.json", metadata)
        paths.reports_dir.mkdir(parents=True, exist_ok=True)
        dataset.write_json(
            paths.reports_dir / "tuning.json",
            {
                "note": "Every trial is scored on tune-year AUPRC with early stopping, capped at "
                f"{s.max_trees} trees and {s.max_leaves} leaves per tree.",
                "best_params": best_params,
                "best_tune_auprc": tune_auprc,
                "n_trees": n_trees,
                "trials": trials,
            },
        )
        log.params({f"best_{k}": v for k, v in best_params.items()})
        log.metrics({"tune_auprc": tune_auprc, "n_trees": n_trees, "n_leaves": n_leaves})
        for name in (
            "model.txt",
            "calibrator.json",
            "tiers.json",
            "feature_spec.json",
            "metadata.json",
        ):
            log.artifact(paths.model_dir / name)

    print(
        f"train: {len(y_tr)} rows, tune {len(y_tu)}, calibrate {len(y_ca)}. "
        f"lgbm {n_trees} trees, {n_leaves} leaves, tune AUPRC {tune_auprc:.4f}. "
        f"logreg C={logreg.C}"
    )
    return metadata


def _auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else None
