"""The published site data: shape, size, and agreement with the reports it came from."""

from __future__ import annotations

import pytest

from riskmodel import config, dataset
from riskmodel.export_site import rounded

from . import site_schema


@pytest.fixture(scope="module")
def site():
    return site_schema.check_all(config.paths().site_data_dir)


@pytest.fixture(scope="module")
def reports():
    r = config.paths().reports_dir
    return {
        name: dataset.read_json(r / f"{name}.json")
        for name in ("metrics", "sanity", "capacity", "fairness", "drift", "explain")
    }


def test_every_file_has_the_contract_keys_and_is_not_fixture_data(site):
    assert set(site) == {*site_schema.FILES, "pipeline.json"}


def test_sizes_are_inside_the_budget():
    folder = config.paths().site_data_dir
    model = (folder / "model_browser.json").stat().st_size
    rest = sum(p.stat().st_size for p in folder.glob("*.json") if p.name != "model_browser.json")
    assert model < config.MAX_BROWSER_MODEL_BYTES
    assert rest < 600_000


def test_headline_matches_metrics_report(site, reports):
    h, m = site["summary.json"]["headline"], reports["metrics"]
    for key in site_schema.MODEL_KEYS:
        assert h["recall_at_k"][key] == rounded(m["models"][key]["at"]["0.10"]["recall"])
        assert h["dollars_at_k"][key] == rounded(m["models"][key]["at"]["0.10"]["dollars"])
        assert h["auroc"][key] == rounded(m["models"][key]["auroc"])
    d = m["deltas"]["lgbm_minus_prior_cost"]
    assert h["delta_recall_at_k_vs_prior"] == rounded(d["recall@0.10"])
    assert h["delta_auprc_vs_prior"] == rounded(d["auprc"])
    data = site["summary.json"]["data"]
    assert data["n_test"] == m["test"]["n"] and data["n_test_pos"] == m["test"]["n_pos"]
    assert data["n_bootstrap"] == m["n_bootstrap"] == 1000


def test_the_paired_difference_is_the_difference_of_the_point_estimates(reports):
    m = reports["metrics"]
    lgbm = m["models"]["lgbm"]["at"]["0.10"]["recall"]["v"]
    prior = m["models"]["prior_cost"]["at"]["0.10"]["recall"]["v"]
    d = m["deltas"]["lgbm_minus_prior_cost"]["recall@0.10"]
    assert d["v"] == pytest.approx(lgbm - prior, abs=1e-12)
    assert d["lo"] <= d["v"] <= d["hi"]


def test_capacity_table_agrees_with_the_headline(site, reports):
    at10 = site["performance.json"]["capacity"][9]
    h = site["summary.json"]["headline"]
    assert at10["lgbm"]["found"] == pytest.approx(h["recall_at_k"]["lgbm"]["v"], abs=1e-4)
    assert at10["prior_cost"]["found"] == pytest.approx(
        h["recall_at_k"]["prior_cost"]["v"], abs=1e-4
    )
    assert at10["lgbm"]["dollars"] == pytest.approx(h["dollars_at_k"]["lgbm"]["v"], abs=1e-4)
    assert at10["n_contacted"] == round(0.10 * site["summary.json"]["data"]["n_test"])


def test_members_field_counts_are_what_the_note_says(site):
    s = site["summary.json"]
    field, note = s["members_field"], s["members_field_note"]
    assert len(field) == 2000 == note["n"]
    band = note["band"]
    assert sum(m["y"] for m in field if m["rank_model"] < band) == note["in_band_model"]
    assert sum(m["y"] for m in field if m["rank_prior"] < band) == note["in_band_prior"]


def test_exemplars_follow_the_selection_rule(site):
    ex = site["explain.json"]["exemplars"]
    kinds = [e["kind"] for e in ex]
    assert len(ex) == 12 and len({e["id"] for e in ex}) == 12
    assert {k: kinds.count(k) for k in set(kinds)} == {
        "flagged_right": 4,
        "flagged_wrong": 2,
        "missed": 2,
        "routine": 4,
    }
    right = [e for e in ex if e["kind"] == "flagged_right"]
    assert sum(e["rank_prior_pct"] >= 0.10 for e in right) >= 2  # the prior-cost rule misses these
    for e in ex:
        flagged = e["percentile"] > 90
        assert flagged == (e["kind"] in ("flagged_right", "flagged_wrong")), e["title"]
        assert (e["y"] == 1) == (e["kind"] in ("flagged_right", "missed")), e["title"]


def test_sanity_checks_passed(reports):
    s = reports["sanity"]
    assert 0.45 <= s["shuffled_labels"]["test_auroc"] <= 0.55
    assert s["single_feature_auroc"]["max"]["auroc"] < 0.98
    assert len(s["single_feature_auroc"]["all"]) == 59
    assert s["split"]["holdout_member_rows_in_fit"] == 0
    assert s["split"]["test_year_rows_in_fit"] == 0
    assert s["split"]["forbidden_columns_in_features"] == []
    assert s["all_pass"] is True


def test_sensitivity_block_is_complete(site, reports):
    s = site["performance.json"]["sensitivity"]
    n_test = site["summary.json"]["data"]["n_test"]
    ex = s["excluding_pregnant"]
    assert ex["n"] + ex["n_excluded"] == n_test
    g = s["gain_decomposition"]
    assert g["net_gain"] == g["found_by_model_only"] - g["found_by_prior_only"]
    assert s["gain_share_pregnant"] == pytest.approx(
        g["found_by_model_only_pregnant"] / g["found_by_model_only"], abs=1e-4
    )
    n_only = g["found_by_model_only"]
    assert s["gain_share_pregnancy_in_outcome_year"] == pytest.approx(
        g["found_by_model_only_pregnancy_in_outcome_year"] / n_only, abs=1e-4
    )
    assert s["gain_share_pregnancy_starting_in_outcome_year"] == pytest.approx(
        g["found_by_model_only_pregnancy_starting_in_outcome_year"] / n_only, abs=1e-4
    )
    # a pregnancy under way on January 1 is also a pregnancy in the outcome year
    assert s["gain_share_pregnancy_starting_in_outcome_year"] + s[
        "gain_share_pregnant"
    ] == pytest.approx(s["gain_share_pregnancy_in_outcome_year"], abs=1e-4)
    assert 0 < s["test_pos_share_pregnancy_in_outcome_year"] < 1
    strata = s["by_childbearing_stratum"]
    assert strata["women_18_49"]["n"] + strata["everyone_else"]["n"] == n_test


def test_fairness_small_groups_are_marked(site):
    groups = site["fairness.json"]["groups"]
    small = [g for g in groups if g["small"]]
    assert small and all(g["attribute"] == "race" for g in small)
    assert (
        sum(g["n"] for g in groups if g["attribute"] == "sex")
        == site["summary.json"]["data"]["n_test"]
    )


def test_drift_report_leads_with_the_real_shifts(site):
    d = site["drift.json"]
    top = d["real_shifts"][0]
    assert top["status"] == "alert" and top["feature"].startswith("outpatient_visits")
    assert top["likely_cause"] and "2021" in top["likely_cause"]
    assert d["simulation"]["alerts_fired"]
    assert [y["year"] for y in d["by_year"]] == site["summary.json"]["data"]["index_years"]


def test_interactive_inputs_are_about_ten(site):
    feats = site["model_browser.json"]["features"]
    chosen = [f["name"] for f in feats if f["interactive"]]
    assert 8 <= len(chosen) <= 12
    assert {"cost_prior_12m", "age", "pregnant_at_index"} <= set(chosen)
    derived = {d["feature"] for d in site["model_browser.json"]["derived"]}
    assert not (derived & set(chosen))
