"""Evaluate on the test years and write metrics.json and sanity.json.

Everything here reads the saved model. The only fitting is for the checks that need
a second model (shuffled labels, no prior-cost features, no pregnancy feature), and
those use the same fit rows and the same hyperparameters as the main model.

Intervals come from 1,000 bootstrap resamples of members (not rows), so a member's
two test years stay together. The same resamples are used for every approach, which
makes the differences against the prior-cost rule paired. The intervals describe
sampling noise in the test set for this one fitted model. They do not include the
noise from refitting the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from . import config, dataset, tracking
from . import metrics as M
from .scoring import RANK_COLUMN, Scored, load_scored
from .train import fit_lgbm

MODELS = ("lgbm", "logreg", "prior_cost")
LABELS = {
    "lgbm": "Gradient boosted trees",
    "logreg": "Logistic regression",
    "prior_cost": "Last year's cost",
    "random": "Random",
}
HK = f"{config.HEADLINE_K:.2f}"


# --------------------------------------------------------------------------- helpers


def _arrays(df: pd.DataFrame):
    return (
        df[config.LABEL].to_numpy(dtype=float),
        df[config.COST_NEXT].to_numpy(dtype=float),
        df["tie_key"].to_numpy(),
    )


def ranked(df: pd.DataFrame, score: np.ndarray, weights: np.ndarray | None):
    """Ranking metrics on the rows and on each resample: (point dict, bootstrap dict)."""
    y, cost, tie = _arrays(df)
    point = {k: float(v[0]) for k, v in M.ranking_metrics(score, y, cost, tie).items()}
    boot = M.ranking_metrics(score, y, cost, tie, weights=weights) if weights is not None else {}
    return point, boot


def intervals(point: dict, boot: dict) -> dict:
    return {k: M.interval(point[k], boot[k]) for k in point}


def delta(point_a, boot_a, point_b, boot_b) -> dict:
    """Paired difference a minus b for every ranking metric."""
    return {k: M.interval(point_a[k] - point_b[k], boot_a[k] - boot_b[k]) for k in point_a}


def at_k_block(iv: dict) -> dict:
    return {
        f"{k:.2f}": {m: iv[f"{m}@{k:.2f}"] for m in ("precision", "recall", "lift", "dollars")}
        for k in config.K_GRID
    }


def short(point: dict) -> dict:
    """The three numbers reported for subsets."""
    return {
        "auroc": _nan_to_none(point["auroc"]),
        "auprc": _nan_to_none(point["auprc"]),
        "recall_at_10": _nan_to_none(point[f"recall@{HK}"]),
    }


def _nan_to_none(x):
    return float(x) if x is not None and np.isfinite(x) else None


def thin_curve(a, b, n: int = 200) -> list[list[float]]:
    keep = (
        np.arange(len(a))
        if len(a) <= n
        else np.unique(np.linspace(0, len(a) - 1, n).round().astype(int))
    )
    return [[float(a[i]), float(b[i])] for i in keep]


def probability_block(p, df, weights) -> tuple[dict, dict]:
    """Brier, ECE, calibration slope and intercept with intervals, plus the 10 bins.

    Also returns the per-resample Brier and ECE, so two scores can be compared on
    the same resamples.
    """
    y, _, tie = _arrays(df)
    point = M.probability_metrics(p, y, tie)
    boot = M.probability_metrics(p, y, tie, weights=weights)
    line = M.calibration_line(p, y)
    line_boot = M.calibration_line(p, y, weights=weights)
    block = {
        "brier": M.interval(point["brier"][0], boot["brier"]),
        "ece": M.interval(point["ece"][0], boot["ece"]),
        "calibration_slope": M.interval(line["slope"][0], line_boot["slope"]),
        "calibration_intercept": M.interval(line["intercept"][0], line_boot["intercept"]),
        "bins": M.calibration_bins(p, y, tie),
    }
    return block, boot


# --------------------------------------------------------------------------- pieces


def by_year(sc: Scored) -> list[dict]:
    """Metrics for every index year, ranking within the year.

    For years before the test years the model has seen most of the rows, so the
    all-member numbers are in-sample. `holdout_members` repeats the numbers on the
    20% of members the model never saw, which is the fair comparison for those years.
    """
    out = []
    for year, rows in sc.frame.groupby("index_year"):
        entry = {
            "year": int(year),
            "split": sc.plan.split_of(int(year)),
            "n": len(rows),
            "prevalence": float(rows[config.LABEL].mean()),
        }
        for key in MODELS:
            entry[key] = short(ranked(rows, rows[RANK_COLUMN[key]].to_numpy(), None)[0])
        held = rows[rows["holdout"]]
        if entry["split"] != "test" and held[config.LABEL].nunique() == 2:
            entry["holdout_members"] = {
                "n": len(held),
                **{
                    key: short(ranked(held, held[RANK_COLUMN[key]].to_numpy(), None)[0])
                    for key in MODELS
                },
            }
        out.append(entry)
    return out


def seen_unseen(test: pd.DataFrame, W: np.ndarray) -> dict:
    """Test rows split by whether the member contributed any fit row.

    Unseen members are the 20% holdout plus anyone whose first eligible year is a
    test year. Each group is ranked on its own, top 10% of that group.
    """
    out = {}
    for name, mask in (("seen", test["seen"].to_numpy()), ("unseen", ~test["seen"].to_numpy())):
        rows = test[mask]
        w = W[:, mask]
        pt_m, bt_m = ranked(rows, rows["raw"].to_numpy(), w)
        pt_p, bt_p = ranked(rows, rows["prior_cost"].to_numpy(), w)
        d = delta(pt_m, bt_m, pt_p, bt_p)
        out[name] = {
            "n": len(rows),
            "n_pos": int(rows[config.LABEL].sum()),
            "n_holdout_members_rows": int(rows["holdout"].sum()),
            **short(pt_m),
            "recall_at_10_ci": M.interval(pt_m[f"recall@{HK}"], bt_m[f"recall@{HK}"]),
            "prior_cost": short(pt_p),
            "delta_recall_at_10_vs_prior": d[f"recall@{HK}"],
        }
    return out


def within_year_ranking(test: pd.DataFrame) -> dict:
    """Recall at 10% when each test year is ranked on its own, then pooled.

    The headline ranks both test years together. This is the check that the choice
    does not matter.
    """
    out = {}
    for key in MODELS:
        found = 0
        for _, rows in test.groupby("index_year"):
            mask = M.top_k_mask(
                rows[RANK_COLUMN[key]].to_numpy(), rows["tie_key"].to_numpy(), config.HEADLINE_K
            )
            found += int(rows[config.LABEL].to_numpy()[mask].sum())
        out[key] = found / max(1, int(test[config.LABEL].sum()))
    return out


def ablation(sc: Scored, drop: list[str], params: dict, shuffle: bool = False):
    """Refit with some features removed (or labels shuffled) and score every test row."""
    keep = [f for f in sc.features if f not in set(drop)]
    frame = sc.frame
    parts = {}
    for split in ("train", "tune"):
        rows = dataset.fit_rows(frame, split)
        parts[split] = (
            dataset.feature_matrix(rows, keep),
            rows[config.LABEL].to_numpy(dtype=int),
        )
    (X_tr, y_tr), (X_tu, y_tu) = parts["train"], parts["tune"]
    if shuffle:
        rng = np.random.default_rng(config.SEED)
        y_tr, y_tu = rng.permutation(y_tr), rng.permutation(y_tu)
    booster, n_trees, _ = fit_lgbm(X_tr, y_tr, X_tu, y_tu, params, keep)
    score = booster.predict(dataset.feature_matrix(sc.test, keep), raw_score=True)
    return score, n_trees


def pregnancy_sensitivity(sc: Scored, test, W, params, full_pt, full_bt, prior_pt, prior_bt):
    """Is the gain over the prior-cost rule only pregnancy?

    Five views:
    1. excluding_pregnant: drop test rows with pregnant_at_index = 1 and rank what is left.
    2. without_pregnancy_feature: refit without the feature, evaluate on the full test set.
    3. prior_cost_plus_pregnancy_rule: a two-line rule (pregnant members first, then by
       last year's cost) as a stronger simple baseline.
    4. by_childbearing_stratum: women aged 18 to 49 and everyone else, each ranked on its
       own. A pregnancy that starts during the outcome year is not visible on January 1,
       but age and sex are, so this is where a model can pick up maternity cost that the
       pregnancy flag does not explain.
    5. gain decomposition at 10%: who the model finds that the prior-cost rule misses.
    """
    y = test[config.LABEL].to_numpy(dtype=int)
    preg = test[config.PREGNANCY].to_numpy(dtype=float) == 1
    tie = test["tie_key"].to_numpy()

    keep = ~preg
    rows = test[keep]
    w = W[:, keep]
    pt_m, bt_m = ranked(rows, rows["raw"].to_numpy(), w)
    pt_p, bt_p = ranked(rows, rows["prior_cost"].to_numpy(), w)
    d = delta(pt_m, bt_m, pt_p, bt_p)

    def block(pt, bt):
        return {
            "auroc": pt["auroc"],
            "auprc": pt["auprc"],
            "recall_at_10": M.interval(pt[f"recall@{HK}"], bt[f"recall@{HK}"]),
            "auroc_ci": M.interval(pt["auroc"], bt["auroc"]),
            "auprc_ci": M.interval(pt["auprc"], bt["auprc"]),
            "dollars_at_10": M.interval(pt[f"dollars@{HK}"], bt[f"dollars@{HK}"]),
        }

    excluding = {
        "n": len(rows),
        "n_pos": int(rows[config.LABEL].sum()),
        "n_excluded": int(preg.sum()),
        "n_pos_excluded": int(y[preg].sum()),
        "lgbm": block(pt_m, bt_m),
        "prior_cost": block(pt_p, bt_p),
        "delta_recall_at_10_vs_prior": d[f"recall@{HK}"],
        "delta_auprc_vs_prior": d["auprc"],
        "delta_auroc_vs_prior": d["auroc"],
        "delta_dollars_at_10_vs_prior": d[f"dollars@{HK}"],
    }

    score_np, n_trees_np = ablation(sc, [config.PREGNANCY], params)
    pt_np, bt_np = ranked(test, score_np, W)
    d_np = delta(pt_np, bt_np, prior_pt, prior_bt)
    d_full = delta(full_pt, full_bt, pt_np, bt_np)
    without = {
        "n_trees": n_trees_np,
        "lgbm": block(pt_np, bt_np),
        "delta_recall_at_10_vs_prior": d_np[f"recall@{HK}"],
        "delta_auprc_vs_prior": d_np["auprc"],
        "full_model_minus_this_recall_at_10": d_full[f"recall@{HK}"],
        "full_model_minus_this_auprc": d_full["auprc"],
    }

    rule = np.where(preg, 1e12, 0.0) + test["prior_cost"].to_numpy()
    pt_r, bt_r = ranked(test, rule, W)
    d_rule = delta(full_pt, full_bt, pt_r, bt_r)
    rule_block = {
        "rule": "pregnant on the index date first, then by last year's cost",
        "prior_cost_plus_pregnancy": block(pt_r, bt_r),
        "lgbm_minus_rule_recall_at_10": d_rule[f"recall@{HK}"],
        "lgbm_minus_rule_auprc": d_rule["auprc"],
        "lgbm_minus_rule_dollars_at_10": d_rule[f"dollars@{HK}"],
    }

    women = (test[config.SEX_FEMALE].to_numpy(dtype=float) == 1) & (
        test[config.AGE].to_numpy(dtype=float) <= config.CHILDBEARING_MAX_AGE
    )
    strata = {}
    for name, mask in (("women_18_49", women), ("everyone_else", ~women)):
        sub = test[mask]
        if sub[config.LABEL].nunique() < 2:
            continue
        ws = W[:, mask]
        s_m, sb_m = ranked(sub, sub["raw"].to_numpy(), ws)
        s_p, sb_p = ranked(sub, sub["prior_cost"].to_numpy(), ws)
        sd = delta(s_m, sb_m, s_p, sb_p)
        strata[name] = {
            "n": len(sub),
            "n_pos": int(sub[config.LABEL].sum()),
            "prevalence": float(sub[config.LABEL].mean()),
            "share_of_test_rows": float(mask.mean()),
            "share_of_high_cost": float(y[mask].sum() / max(1, y.sum())),
            "lgbm": block(s_m, sb_m),
            "prior_cost": block(s_p, sb_p),
            "delta_recall_at_10_vs_prior": sd[f"recall@{HK}"],
            "delta_auprc_vs_prior": sd["auprc"],
            "delta_auroc_vs_prior": sd["auroc"],
        }

    in_model = M.top_k_mask(test["raw"].to_numpy(), tie, config.HEADLINE_K)
    in_prior = M.top_k_mask(test["prior_cost"].to_numpy(), tie, config.HEADLINE_K)
    pos = y == 1
    model_only = pos & in_model & ~in_prior
    prior_only = pos & in_prior & ~in_model
    both = pos & in_model & in_prior
    net = int(model_only.sum()) - int(prior_only.sum())
    net_pregnant = int((model_only & preg).sum()) - int((prior_only & preg).sum())
    decomposition = {
        "k": config.HEADLINE_K,
        "found_by_both": int(both.sum()),
        "found_by_model_only": int(model_only.sum()),
        "found_by_model_only_pregnant": int((model_only & preg).sum()),
        "found_by_prior_only": int(prior_only.sum()),
        "found_by_prior_only_pregnant": int((prior_only & preg).sum()),
        "net_gain": net,
        "net_gain_pregnant": net_pregnant,
        "net_gain_not_pregnant": net - net_pregnant,
        "pregnant_test_rows": int(preg.sum()),
        "pregnant_high_cost": int((pos & preg).sum()),
        "pregnant_in_model_top": int((in_model & preg).sum()),
        "pregnant_in_prior_top": int((in_prior & preg).sum()),
        # Women 18 to 49 who were not pregnant on the index date. In this data their
        # high-cost years are mostly pregnancies that begin during the outcome year.
        "found_by_model_only_women_18_49_not_pregnant": int((model_only & women & ~preg).sum()),
        "found_by_prior_only_women_18_49_not_pregnant": int((prior_only & women & ~preg).sum()),
        "found_by_both_women_18_49": int((both & women).sum()),
        "women_18_49_in_model_top": int((in_model & women).sum()),
        "women_18_49_in_prior_top": int((in_prior & women).sum()),
        "women_18_49_test_rows": int(women.sum()),
        "women_18_49_high_cost": int((pos & women).sum()),
    }
    prior_pct = M.rank_pct(test["prior_cost"].to_numpy(), tie)
    if model_only.any():
        q = np.quantile(prior_pct[model_only], [0.25, 0.5, 0.75])
        decomposition["model_only_prior_cost_rank"] = {
            "note": "Where the members found only by the model sit in the prior-cost ranking "
            "(0 is the highest prior cost, the rule's cut is at 0.10).",
            "p25": float(q[0]),
            "median": float(q[1]),
            "p75": float(q[2]),
            "share_beyond_0.20": float((prior_pct[model_only] >= 0.20).mean()),
            "share_beyond_0.50": float((prior_pct[model_only] >= 0.50).mean()),
        }
    # Pregnancy during the outcome year is an outcome column in the mart. It is read here,
    # after ranking, to say what the extra finds are. It is never a model input.
    preg_year = test[config.PREGNANCY_IN_OUTCOME_YEAR].to_numpy(dtype=float) == 1
    preg_starts = preg_year & ~preg
    decomposition.update(
        {
            "high_cost_pregnancy_in_outcome_year": int((pos & preg_year).sum()),
            "found_by_model_only_pregnancy_in_outcome_year": int((model_only & preg_year).sum()),
            "found_by_model_only_pregnancy_starting_in_outcome_year": int(
                (model_only & preg_starts).sum()
            ),
            "found_by_prior_only_pregnancy_in_outcome_year": int((prior_only & preg_year).sum()),
        }
    )
    # What the extra finds outside women 18 to 49 have in common.
    rest_only = model_only & ~women
    decomposition["found_by_model_only_everyone_else"] = int(rest_only.sum())
    for group in ("chronic_kidney_disease", "diabetes"):
        column = f"has_{group}"
        if column in test.columns:
            has = test[column].to_numpy(dtype=float) == 1
            decomposition[f"found_by_model_only_everyone_else_with_{group}"] = int(
                (rest_only & has).sum()
            )
    n_only = int(model_only.sum())
    share_women = float((model_only & women & ~preg).sum() / n_only) if n_only > 0 else None
    share = float((model_only & preg).sum() / model_only.sum()) if model_only.sum() > 0 else None
    return {
        "excluding_pregnant": excluding,
        "without_pregnancy_feature": without,
        "prior_cost_plus_pregnancy_rule": rule_block,
        "by_childbearing_stratum": strata,
        "gain_share_pregnant": share,
        "gain_share_women_18_49_not_pregnant": share_women,
        "test_pos_share_pregnancy_in_outcome_year": (
            float((pos & preg_year).sum() / pos.sum()) if pos.any() else None
        ),
        "gain_share_pregnancy_in_outcome_year": (
            float((model_only & preg_year).sum() / n_only) if n_only > 0 else None
        ),
        "gain_share_pregnancy_starting_in_outcome_year": (
            float((model_only & preg_starts).sum() / n_only) if n_only > 0 else None
        ),
        "gain_decomposition": decomposition,
    }


def single_feature_scan(sc: Scored) -> list[dict]:
    """AUROC of each feature alone against the label (missing values set to the median).

    Reported as max(auc, 1 - auc), so a feature that predicts in either direction
    shows up. Anything near 1.0 would mean the label leaked into a feature.
    """
    out = []
    train = sc.frame[sc.frame["split"] == "train"]
    test = sc.test
    for name in sc.features:
        row = {"feature": name}
        for label, rows in (("train", train), ("test", test)):
            x = rows[name].to_numpy(dtype=float, na_value=np.nan)
            y = rows[config.LABEL].to_numpy()
            if np.isnan(x).all() or len(np.unique(y)) < 2:
                row[f"auroc_{label}"] = None
                continue
            x = np.where(np.isnan(x), np.nanmedian(x), x)
            a = roc_auc_score(y, x)
            row[f"auroc_{label}"] = float(max(a, 1 - a))
        out.append(row)
    return sorted(out, key=lambda r: -(r["auroc_test"] or 0))


def split_checks(sc: Scored) -> dict:
    """Counts that must be zero if the split is what it claims to be."""
    f = sc.frame
    fit = f[f["role"] == "fit"]
    plan = sc.plan
    return {
        "holdout_member_rows_in_fit": int(fit["holdout"].sum()),
        "members_in_both_fit_and_holdout": len(
            set(fit["member_id"]) & set(f.loc[f["holdout"], "member_id"])
        ),
        "test_year_rows_in_fit": int(fit["index_year"].isin(plan.test_years).sum()),
        "years_in_more_than_one_split": int(
            len(set(plan.train_years) & {plan.tune_year, plan.calib_year, *plan.test_years})
            + (plan.tune_year == plan.calib_year)
            + len({plan.tune_year, plan.calib_year} & set(plan.test_years))
        ),
        "forbidden_columns_in_features": sorted(
            set(sc.features) & set(config.FORBIDDEN_AS_FEATURES)
        ),
        "holdout_share_of_members": float(f.groupby("member_id")["holdout"].first().mean()),
        "rows_by_role": {k: int(v) for k, v in f["role"].value_counts().items()},
    }


# --------------------------------------------------------------------------- stage


def run() -> dict:
    paths = config.paths()
    s = config.settings()
    sc = load_scored(paths)
    test = sc.test.reset_index(drop=True)
    y = test[config.LABEL].to_numpy(dtype=float)
    params = dict(sc.scorer.metadata["params"])

    _, member_pos = np.unique(test["member_id"].to_numpy(), return_inverse=True)
    counts = M.member_resamples(int(member_pos.max()) + 1, s.n_bootstrap, config.SEED)
    W = counts[:, member_pos]  # (resamples, test rows)

    # Ranking metrics for each approach, on the same resamples.
    point, boot = {}, {}
    for key in MODELS:
        point[key], boot[key] = ranked(test, test[RANK_COLUMN[key]].to_numpy(), W)

    cal_raw, raw_boot = probability_block(test["p_raw"].to_numpy(), test, W)
    cal_cal, cal_boot = probability_block(test["p"].to_numpy(), test, W)
    cal_logreg, _ = probability_block(test["logreg"].to_numpy(), test, W)
    # Calibrated minus raw on the same resamples. Negative means calibration helped.
    calibration_change = {
        name: M.interval(cal_cal[name]["v"] - cal_raw[name]["v"], cal_boot[name] - raw_boot[name])
        for name in ("brier", "ece")
    }

    models = {}
    for key in MODELS:
        iv = intervals(point[key], boot[key])
        prob = {"lgbm": cal_cal, "logreg": cal_logreg}.get(key)
        fpr, tpr, _ = roc_curve(y, test[RANK_COLUMN[key]].to_numpy())
        prec, rec, _ = precision_recall_curve(y, test[RANK_COLUMN[key]].to_numpy())
        models[key] = {
            "label": LABELS[key],
            "auroc": iv["auroc"],
            "auprc": iv["auprc"],
            "brier": prob["brier"] if prob else None,
            "ece": prob["ece"] if prob else None,
            "calibration_slope": prob["calibration_slope"] if prob else None,
            "calibration_intercept": prob["calibration_intercept"] if prob else None,
            "at": at_k_block(iv),
            "roc": thin_curve(fpr, tpr),
            "pr": thin_curve(rec[::-1], prec[::-1]),
            "calibration": prob["bins"] if prob else None,
        }
    prevalence = float(y.mean())

    deltas = {
        "lgbm_minus_prior_cost": delta(
            point["lgbm"], boot["lgbm"], point["prior_cost"], boot["prior_cost"]
        ),
        "logreg_minus_prior_cost": delta(
            point["logreg"], boot["logreg"], point["prior_cost"], boot["prior_cost"]
        ),
        "lgbm_minus_logreg": delta(point["lgbm"], boot["lgbm"], point["logreg"], boot["logreg"]),
    }

    sensitivity = pregnancy_sensitivity(
        sc, test, W, params, point["lgbm"], boot["lgbm"], point["prior_cost"], boot["prior_cost"]
    )

    # Sanity checks that need another fit.
    shuffled_score, _ = ablation(sc, [], params, shuffle=True)
    shuffled_auroc = float(roc_auc_score(y, shuffled_score))
    prior_group = [
        str(n) for n in sc.dictionary.loc[sc.dictionary["group"] == "prior cost", "name"]
    ]
    npc_score, npc_trees = ablation(sc, prior_group, params)
    npc_pt, npc_bt = ranked(test, npc_score, W)
    scan = single_feature_scan(sc)
    top_single = scan[0]
    checks = split_checks(sc)
    sanity = {
        "shuffled_labels": {
            "what": "Same pipeline with the training and tune labels shuffled. A value far "
            "from 0.5 would mean information reaches the test score some other way.",
            "test_auroc": shuffled_auroc,
            "pass": bool(0.45 <= shuffled_auroc <= 0.55),
        },
        "no_prior_cost_features": {
            "what": "Model refit without the prior cost group, to show what the other "
            "features carry on their own.",
            "dropped": prior_group,
            "n_trees": npc_trees,
            **short(npc_pt),
            "recall_at_10_ci": M.interval(npc_pt[f"recall@{HK}"], npc_bt[f"recall@{HK}"]),
        },
        "single_feature_auroc": {
            "what": "AUROC of each feature alone. Above 0.98 would be a leak signal.",
            "max": {"feature": top_single["feature"], "auroc": top_single["auroc_test"]},
            "pass": bool((top_single["auroc_test"] or 0) < 0.98),
            "all": scan,
        },
        "split": {
            **checks,
            "pass": bool(
                checks["holdout_member_rows_in_fit"] == 0
                and checks["members_in_both_fit_and_holdout"] == 0
                and checks["test_year_rows_in_fit"] == 0
                and checks["years_in_more_than_one_split"] == 0
                and not checks["forbidden_columns_in_features"]
            ),
        },
    }
    sanity["all_pass"] = bool(
        sanity["shuffled_labels"]["pass"]
        and sanity["single_feature_auroc"]["pass"]
        and sanity["split"]["pass"]
    )

    metrics = {
        "model_version": config.MODEL_VERSION,
        "n_bootstrap": s.n_bootstrap,
        "bootstrap_note": "Members resampled with replacement, 2.5 and 97.5 percentiles. The "
        "same resamples are used for every approach, so differences are paired. The model is "
        "not refit inside the bootstrap.",
        "ranking_note": "Top-k metrics rank by the raw score, ties broken by member id. Both "
        "test years are ranked together.",
        "k": config.HEADLINE_K,
        "test": {
            "years": list(sc.plan.test_years),
            "n": len(test),
            "n_members": int(test["member_id"].nunique()),
            "n_pos": int(y.sum()),
            "prevalence": prevalence,
        },
        "models": models,
        "random": {
            "label": LABELS["random"],
            "auroc": 0.5,
            "auprc": prevalence,
            "at": {
                f"{k:.2f}": {"precision": prevalence, "recall": k, "lift": 1.0, "dollars": k}
                for k in config.K_GRID
            },
        },
        "deltas": deltas,
        "calibration_before_after": {
            "note": "Isotonic regression fit on the calibration year only, evaluated here on "
            "the test years. Slope and intercept clip probabilities to [0.0001, 0.9999].",
            "raw": cal_raw,
            "calibrated": cal_cal,
            "change": calibration_change,
        },
        "by_year": by_year(sc),
        "member_holdout": seen_unseen(test, W),
        "within_year_ranking_recall_at_10": within_year_ranking(test),
        "sensitivity": sensitivity,
    }
    dataset.write_json(paths.reports_dir / "metrics.json", metrics)
    dataset.write_json(paths.reports_dir / "sanity.json", sanity)

    with tracking.run("evaluation", {"n_bootstrap": s.n_bootstrap}) as log:
        flat = {}
        for key in MODELS:
            flat[f"test_auroc_{key}"] = models[key]["auroc"]["v"]
            flat[f"test_auprc_{key}"] = models[key]["auprc"]["v"]
            flat[f"test_recall_at_10_{key}"] = models[key]["at"][HK]["recall"]["v"]
            flat[f"test_dollars_at_10_{key}"] = models[key]["at"][HK]["dollars"]["v"]
        d = deltas["lgbm_minus_prior_cost"]
        flat["delta_recall_at_10_vs_prior"] = d[f"recall@{HK}"]["v"]
        flat["delta_recall_at_10_vs_prior_lo"] = d[f"recall@{HK}"]["lo"]
        flat["delta_recall_at_10_vs_prior_hi"] = d[f"recall@{HK}"]["hi"]
        flat["test_brier_calibrated"] = cal_cal["brier"]["v"]
        flat["test_ece_calibrated"] = cal_cal["ece"]["v"]
        flat["shuffled_label_auroc"] = shuffled_auroc
        log.metrics(flat)
        log.artifact(paths.reports_dir / "metrics.json")
        log.artifact(paths.reports_dir / "sanity.json")

    r = {k: models[k]["at"][HK]["recall"] for k in MODELS}
    dr = deltas["lgbm_minus_prior_cost"][f"recall@{HK}"]
    print(
        "evaluate: recall at 10% "
        + ", ".join(f"{k} {r[k]['v']:.3f}" for k in MODELS)
        + f". lgbm minus prior cost {dr['v']:+.3f} ({dr['lo']:+.3f} to {dr['hi']:+.3f})"
    )
    return metrics
