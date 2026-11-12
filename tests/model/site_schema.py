"""The shape of every file in docs/data/, as the page reads it."""

from __future__ import annotations

import json
from pathlib import Path

FILES = (
    "summary.json",
    "performance.json",
    "explain.json",
    "fairness.json",
    "drift.json",
    "model_browser.json",
    "parity.json",
)
MODEL_KEYS = ("lgbm", "logreg", "prior_cost")


def load(folder: Path, name: str) -> dict:
    return json.loads((folder / name).read_text())


def need(obj: dict, keys, where: str) -> None:
    missing = [k for k in keys if k not in obj]
    assert not missing, f"{where}: missing {missing}"


def is_interval(x) -> bool:
    return isinstance(x, dict) and {"v", "lo", "hi"} <= set(x)


def is_fraction(x) -> bool:
    return isinstance(x, (int, float)) and -1e-9 <= x <= 1 + 1e-9


def check_summary(d: dict) -> None:
    need(
        d,
        [
            "fixture",
            "generated_at",
            "git_sha",
            "model_version",
            "data",
            "headline",
            "members_field",
        ],
        "summary",
    )
    need(
        d["data"],
        [
            "n_patients",
            "n_member_years",
            "n_features",
            "index_years",
            "train_years",
            "tune_year",
            "calib_year",
            "test_years",
            "n_test",
            "n_test_members",
            "n_test_pos",
            "prevalence_test",
            "member_holdout_share",
            "high_cost_threshold_usd_by_year",
            "n_bootstrap",
        ],
        "summary.data",
    )
    h = d["headline"]
    need(
        h,
        [
            "k",
            "recall_at_k",
            "dollars_at_k",
            "auroc",
            "auprc",
            "brier",
            "ece",
            "delta_recall_at_k_vs_prior",
            "delta_auprc_vs_prior",
        ],
        "summary.headline",
    )
    for block in ("recall_at_k", "dollars_at_k"):
        for key in MODEL_KEYS:
            assert is_interval(h[block][key]), f"{block}.{key}"
            assert h[block][key]["lo"] <= h[block][key]["v"] <= h[block][key]["hi"]
        assert h[block]["random"] == h["k"]
    for block in ("auroc", "auprc"):
        assert all(is_interval(h[block][key]) for key in MODEL_KEYS)
    assert is_interval(h["delta_recall_at_k_vs_prior"]) and is_interval(h["delta_auprc_vs_prior"])
    assert isinstance(h["brier"], float) and isinstance(h["ece"], float)
    field = d["members_field"]
    n = len(field)
    for key in ("rank_model", "rank_prior", "rank_random"):
        assert sorted(m[key] for m in field) == list(range(n)), f"{key} is not a 0-based ranking"
    for m in field[:50]:
        need(m, ["id", "y", "cost_next", "p", "rank_model", "rank_prior", "rank_random"], "field")
        assert m["y"] in (0, 1) and is_fraction(m["p"])


def check_performance(d: dict) -> None:
    need(
        d,
        [
            "fixture",
            "models",
            "calibration_before_after",
            "capacity",
            "by_year",
            "member_holdout",
            "sanity",
            "sensitivity",
        ],
        "performance",
    )
    assert [m["key"] for m in d["models"]] == ["lgbm", "logreg", "prior_cost", "random"]
    for m in d["models"]:
        need(m, ["key", "label", "metrics", "roc", "pr", "calibration"], f"models.{m['key']}")
        need(m["metrics"], ["auroc", "auprc", "brier", "ece", "at"], f"metrics.{m['key']}")
        assert set(m["metrics"]["at"]) == {"0.01", "0.05", "0.10", "0.20"}
        for row in m["metrics"]["at"].values():
            assert all(is_interval(row[x]) for x in ("precision", "recall", "lift", "dollars"))
        assert 2 <= len(m["roc"]) <= 200 and 2 <= len(m["pr"]) <= 200
        if m["key"] in ("lgbm", "logreg"):
            assert is_interval(m["metrics"]["brier"]) and is_interval(m["metrics"]["ece"])
            assert 1 <= len(m["calibration"]) <= 10
            need(m["calibration"][0], ["mean_pred", "obs_rate", "n"], "calibration bin")
        else:
            assert m["metrics"]["brier"] is None and m["calibration"] is None
    for side in ("raw", "calibrated"):
        need(d["calibration_before_after"][side], ["brier", "ece", "bins"], f"calibration.{side}")
    cap = d["capacity"]
    assert [round(r["pct"] * 100) for r in cap] == list(range(1, 101))
    for r in (cap[0], cap[9], cap[-1]):
        need(r, ["pct", "n_contacted", "lgbm", "prior_cost", "logreg", "random"], "capacity")
        for key in ("lgbm", "prior_cost", "logreg", "random"):
            need(r[key], ["found", "dollars", "precision"], f"capacity.{key}")
            assert is_fraction(r[key]["found"]) and is_fraction(r[key]["dollars"])
    for key in MODEL_KEYS:
        assert abs(cap[-1][key]["found"] - 1) < 1e-4 and abs(cap[-1][key]["dollars"] - 1) < 1e-4
    for row in d["by_year"]:
        need(row, ["year", "split", "n", "prevalence", "lgbm", "prior_cost"], "by_year")
        assert row["split"] in ("train", "tune", "calibrate", "test")
        need(row["lgbm"], ["auroc", "auprc", "recall_at_10"], "by_year.lgbm")
    for side in ("seen", "unseen"):
        need(d["member_holdout"][side], ["n", "auroc", "auprc", "recall_at_10"], side)
    need(
        d["sanity"],
        ["shuffled_label_auroc", "no_prior_cost_features", "max_single_feature_auroc"],
        "sanity",
    )
    need(d["sanity"]["no_prior_cost_features"], ["auroc", "auprc", "recall_at_10"], "sanity")
    need(d["sanity"]["max_single_feature_auroc"], ["feature", "auroc"], "sanity")
    s = d["sensitivity"]
    need(
        s,
        [
            "excluding_pregnant",
            "without_pregnancy_feature",
            "gain_share_pregnant",
            "test_pos_share_pregnancy_in_outcome_year",
            "gain_share_pregnancy_in_outcome_year",
            "gain_share_pregnancy_starting_in_outcome_year",
        ],
        "sens",
    )
    need(s["excluding_pregnant"], ["n", "n_pos", "lgbm", "prior_cost"], "excluding_pregnant")
    for block in (
        s["excluding_pregnant"]["lgbm"],
        s["excluding_pregnant"]["prior_cost"],
        s["without_pregnancy_feature"]["lgbm"],
    ):
        need(block, ["auroc", "auprc", "recall_at_10"], "sensitivity block")
        assert is_interval(block["recall_at_10"]) and isinstance(block["auroc"], float)


def check_explain(d: dict, n_features: int) -> None:
    need(d, ["fixture", "base_value", "features", "dependence", "exemplars"], "explain")
    assert len(d["features"]) == n_features
    assert [f["rank"] for f in d["features"]] == list(range(1, n_features + 1))
    for f in d["features"][:3]:
        need(f, ["name", "label", "group", "description", "unit", "importance", "rank"], "feature")
    assert len(d["dependence"]) == min(8, n_features)
    for dep in d["dependence"]:
        need(dep, ["feature", "label", "unit", "points"], "dependence")
        for p in dep["points"]:
            need(p, ["x", "mean_contrib", "n"], "dependence point")
    for ex in d["exemplars"]:
        need(
            ex,
            [
                "id",
                "kind",
                "title",
                "facts",
                "p",
                "percentile",
                "tier",
                "y",
                "cost_next",
                "rank_prior_pct",
                "contribs",
                "other_contrib",
                "row",
            ],
            "exemplar",
        )
        assert ex["kind"] in ("flagged_right", "flagged_wrong", "missed", "routine")
        assert ex["tier"] in ("high", "rising", "routine")
        assert 0 <= ex["percentile"] <= 100 and is_fraction(ex["rank_prior_pct"])
        assert len(ex["row"]) == n_features and len(ex["contribs"]) <= 8
        for c in ex["contribs"]:
            need(c, ["feature", "label", "value_display", "contrib"], "contrib")


def check_fairness(d: dict) -> None:
    need(
        d,
        ["fixture", "excluded_from_model", "small_group_rule", "groups", "proxy_audit"],
        "fairness",
    )
    assert d["excluded_from_model"] == ["race", "ethnicity"]
    assert {g["attribute"] for g in d["groups"]} == {"sex", "age band", "race", "ethnicity"}
    for g in d["groups"]:
        need(
            g,
            [
                "attribute",
                "level",
                "n",
                "n_pos",
                "prevalence",
                "auroc",
                "obs_over_expected",
                "selection_rate_at_10",
                "recall_at_10",
                "small",
            ],
            "group",
        )
        assert g["small"] == (g["n"] < 300 or g["n_pos"] < 30)
    audit = d["proxy_audit"]
    need(audit, ["attribute", "rows", "finding"], "proxy_audit")
    assert audit["attribute"] == "race" and len(audit["finding"]) > 40
    for r in audit["rows"]:
        need(r, ["score_decile", "level", "mean_chronic_conditions", "n"], "proxy row")
        assert 1 <= r["score_decile"] <= 10


def check_drift(d: dict) -> None:
    need(
        d,
        [
            "fixture",
            "reference_years",
            "thresholds",
            "label_lag_note",
            "by_year",
            "feature_psi_latest",
            "simulation",
        ],
        "drift",
    )
    assert d["thresholds"] == {"watch": 0.1, "alert": 0.25}
    for row in d["by_year"]:
        need(row, ["year", "n", "score_psi", "score_ks", "labels_mature", "top_features"], "year")
        for f in row["top_features"]:
            need(f, ["feature", "label", "psi"], "top feature")
    assert d["by_year"][-1]["labels_mature"] is False
    for f in d["feature_psi_latest"]:
        need(f, ["feature", "label", "psi", "status"], "feature_psi_latest")
        assert f["status"] in ("ok", "watch", "alert")
    sim = d["simulation"]
    if sim is not None:
        need(
            sim,
            [
                "title",
                "description",
                "changed_feature",
                "feature_psi_before",
                "feature_psi_after",
                "score_psi_before",
                "score_psi_after",
                "alerts_fired",
                "score_hist_before",
                "score_hist_after",
                "bin_edges",
            ],
            "simulation",
        )
        assert len(sim["bin_edges"]) == len(sim["score_hist_before"]) + 1
        assert "imulat" in sim["title"]  # the word "simulated" has to be in the title


def check_pipeline(d: dict) -> None:
    need(d, ["fixture", "source", "nodes", "edges", "tests", "leakage_checks"], "pipeline")
    need(d["tests"], ["total", "passed", "failed", "by_type"], "pipeline.tests")
    ids = {n["id"] for n in d["nodes"]}
    for n in d["nodes"]:
        need(n, ["id", "layer", "label", "rows", "description"], "node")
    assert all(a in ids and b in ids for a, b in d["edges"])
    for c in d["leakage_checks"]:
        need(c, ["name", "result", "detail"], "leakage check")


def check_browser_model(d: dict, parity: dict) -> None:
    need(
        d,
        [
            "fixture",
            "model_version",
            "n_trees",
            "init_score",
            "features",
            "trees",
            "calibrator",
            "calibrator_input",
            "tiers",
            "score_percentiles",
        ],
        "model_browser",
    )
    assert d["calibrator_input"] in ("probability", "raw")
    assert d["n_trees"] == len(d["trees"]) and len(d["score_percentiles"]) == 101
    assert len(d["calibrator"]["x"]) == len(d["calibrator"]["y"])
    for f in d["features"]:
        need(
            f,
            [
                "name",
                "label",
                "group",
                "unit",
                "type",
                "min",
                "max",
                "step",
                "default",
                "interactive",
                "nullable",
            ],
            "browser feature",
        )
        assert f["type"] in ("numeric", "binary")
        if f["interactive"] and f["type"] == "numeric":
            assert f["max"] > f["min"] and f["step"] > 0
            assert f["default"] is None or f["min"] <= f["default"] <= f["max"]

    def walk(node, depth=0):
        if "v" in node:
            assert set(node) == {"v"}
            return
        assert set(node) == {"f", "t", "d", "l", "r"} and isinstance(node["d"], bool)
        assert 0 <= node["f"] < len(d["features"])
        walk(node["l"], depth + 1)
        walk(node["r"], depth + 1)

    for tree in d["trees"]:
        walk(tree)
    need(parity, ["fixture", "rows", "raw", "p"], "parity")
    assert len(parity["rows"]) == len(parity["raw"]) == len(parity["p"])
    assert all(len(r) == len(d["features"]) for r in parity["rows"])


def check_all(folder: Path, expect_pipeline: bool = True) -> dict:
    """Run every check. Returns the loaded files."""
    files = {name: load(folder, name) for name in FILES}
    for name, d in files.items():
        assert d["fixture"] is False, f"{name} is still fixture data"
    check_summary(files["summary.json"])
    check_performance(files["performance.json"])
    check_explain(files["explain.json"], files["summary.json"]["data"]["n_features"])
    check_fairness(files["fairness.json"])
    check_drift(files["drift.json"])
    check_browser_model(files["model_browser.json"], files["parity.json"])
    if expect_pipeline:
        files["pipeline.json"] = load(folder, "pipeline.json")
        assert files["pipeline.json"]["fixture"] is False
        check_pipeline(files["pipeline.json"])
    return files


if __name__ == "__main__":
    # python -m tests.model.site_schema build/sample/site
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/data")
    checked = check_all(target)
    print(f"{target}: {len(checked)} files match the site data contract")
