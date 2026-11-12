"""The scoring API: happy path, validation, batch limit, parity with batch scoring, audit."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from api import audit, schemas
from riskmodel import config, dataset

ROOT = Path(__file__).resolve().parents[2]


def audit_rows(path) -> list[dict]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("select * from score_audit order by rowid")]
    con.close()
    return rows


# ----------------------------------------------------------------------------- happy path


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": config.MODEL_VERSION, "n_features": 59}


def test_score_one_member(client, sample):
    _, _, payloads = sample
    r = client.post("/v1/score", json=payloads[0])
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "probability",
        "percentile",
        "tier",
        "top_factors",
        "n_missing",
        "warnings",
        "model_version",
        "request_id",
    }
    assert 0 <= body["probability"] <= 1
    assert 0 <= body["percentile"] <= 100
    assert body["tier"] in ("high", "rising", "routine")
    assert body["model_version"] == config.MODEL_VERSION
    assert len(body["request_id"]) == 32
    factors = body["top_factors"]
    assert len(factors) == 5
    sizes = [abs(f["contribution"]) for f in factors]
    assert sizes == sorted(sizes, reverse=True)
    for f in factors:
        assert f["direction"] == ("raises" if f["contribution"] > 0 else "lowers")
        assert f["label"] and isinstance(f["value"], str)


def test_model_endpoint_reports_what_was_shipped(client):
    body = client.get("/v1/model").json()
    model_dir = config.paths().model_dir
    assert body["metadata"] == dataset.read_json(model_dir / "metadata.json")
    assert body["features"] == dataset.read_json(model_dir / "feature_spec.json")["features"]
    metrics = dataset.read_json(config.paths().reports_dir / "metrics.json")
    shown = body["test_metrics"]
    assert shown["recall_at_10"]["model"] == metrics["models"]["lgbm"]["at"]["0.10"]["recall"]
    assert (
        shown["recall_at_10"]["prior_cost_rule"]
        == metrics["models"]["prior_cost"]["at"]["0.10"]["recall"]
    )
    assert shown["test"]["n"] == metrics["test"]["n"]


def test_openapi_lists_every_feature(client):
    props = client.get("/openapi.json").json()["components"]["schemas"]["Features"]["properties"]
    assert len(props) == 59
    assert not (set(props) & set(config.FORBIDDEN_AS_FEATURES))


# ------------------------------------------------------------------ parity with batch scoring


def test_batch_endpoint_matches_batch_scoring_exactly(client, scored, sample):
    """The API and the offline pipeline must give the same number, not a close one."""
    rows, X, payloads = sample
    r = client.post("/v1/score/batch", json={"members": payloads})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == len(payloads)
    assert [item["index"] for item in body["results"]] == list(range(len(payloads)))
    got = np.array([item["probability"] for item in body["results"]])
    # the score columns the evaluation, the site export and the drift report read
    assert np.array_equal(got, rows["p"].to_numpy())
    # and the scorer called directly on the same matrix
    assert np.array_equal(got, scored.scorer.probability(X))
    for item, p in zip(body["results"], got, strict=True):
        assert item["tier"] == scored.scorer.tier(p)
        assert item["percentile"] == scored.scorer.percentile(p)


def test_single_endpoint_matches_batch_scoring_exactly(client, sample):
    rows, _, payloads = sample
    expected = rows["p"].to_numpy()
    for i in range(25):
        body = client.post("/v1/score", json=payloads[i]).json()
        assert body["probability"] == expected[i]


def test_top_factors_match_the_explanation_report(client, scored):
    """Same members, same contributions as the exemplars in explain.json."""
    report = dataset.read_json(config.paths().reports_dir / "explain.json")
    for ex in report["exemplars"]:
        payload = {n: v for n, v in zip(scored.features, ex["row"], strict=True) if v is not None}
        body = client.post("/v1/score", json=payload).json()
        # the report was written on another machine, so allow for the last bit of exp()
        assert body["probability"] == pytest.approx(ex["p"], abs=1e-12)
        assert body["tier"] == ex["tier"]
        for got, want in zip(body["top_factors"], ex["contribs"][:5], strict=True):
            assert got["feature"] == want["feature"]
            assert got["contribution"] == pytest.approx(want["contrib"], abs=1e-12)
            assert got["value"] == want["value_display"]


def test_api_agrees_with_the_browser_parity_file(client, scored):
    """docs/data/parity.json is what the in-browser scorer is tested against."""
    parity = json.loads((ROOT / "docs" / "data" / "parity.json").read_text())
    members = [
        {n: v for n, v in zip(scored.features, row, strict=True) if v is not None}
        for row in parity["rows"]
    ]
    body = client.post("/v1/score/batch", json={"members": members}).json()
    got = [item["probability"] for item in body["results"]]
    assert np.max(np.abs(np.array(got) - np.array(parity["p"]))) <= 1e-9


def test_missing_null_and_omitted_are_the_same(client, sample):
    _, _, payloads = sample
    full = next(p for p in payloads if "bmi" in p and "hba1c" in p)
    omitted = {k: v for k, v in full.items() if k not in ("bmi", "hba1c")}
    as_null = {**omitted, "bmi": None, "hba1c": None}
    a = client.post("/v1/score", json=omitted).json()
    b = client.post("/v1/score", json=as_null).json()
    c = client.post("/v1/score", json=full).json()
    assert a["probability"] == b["probability"]
    assert a["n_missing"] == b["n_missing"] == c["n_missing"] + 2


def test_empty_object_is_scored_as_all_missing(client):
    body = client.post("/v1/score", json={}).json()
    assert body["n_missing"] == 59 and 0 <= body["probability"] <= 1


# ------------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("payload", "field", "message"),
    [
        ({"agee": 40}, "agee", "Extra inputs"),
        ({"race": 1}, "race", "Extra inputs"),
        ({"label_high_cost": 1}, "label_high_cost", "Extra inputs"),
        ({"age": "46"}, "age", "valid number"),
        ({"age": True}, "age", "valid number"),
        ({"age": 46.5}, None, "whole number"),
        ({"age": -1}, "age", "greater than or equal to 0"),
        ({"age": 400}, "age", "less than or equal to"),
        ({"sex_female": 2}, "sex_female", "less than or equal to 1"),
        ({"sex_female": 0.5}, None, "whole number"),
        ({"cost_prior_12m": -5}, "cost_prior_12m", "greater than or equal to 0"),
        ({"cost_prior_12m": 1e12}, "cost_prior_12m", "less than or equal to"),
    ],
)
def test_invalid_input_is_rejected(client, audit_path, payload, field, message):
    r = client.post("/v1/score", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any(message in d["msg"] for d in detail)
    if field:
        assert any(field in d["loc"] for d in detail)
    assert audit_rows(audit_path) == []


@pytest.mark.parametrize("text", ['{"age": NaN}', '{"age": Infinity}', "[1, 2]", "not json"])
def test_malformed_bodies_are_rejected_not_crashed(client, text):
    r = client.post("/v1/score", content=text, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_validation_errors_do_not_echo_the_input(client):
    r = client.post("/v1/score", json={"age": 46, "cost_prior_12m": -98765.25})
    assert r.status_code == 422
    assert "98765" not in r.text
    assert all(set(d) == {"loc", "msg", "type"} for d in r.json()["detail"])


def test_value_outside_training_range_is_scored_with_a_warning(client, scored):
    spec = {f["name"]: f for f in client.get("/v1/model").json()["features"]}
    age = spec["age"]["max"] + 2
    lower, upper = schemas.hard_limits(spec["age"])
    assert spec["age"]["max"] < age <= upper and lower == 0
    body = client.post("/v1/score", json={"age": age}).json()
    assert len(body["warnings"]) == 1 and "age" in body["warnings"][0]
    assert client.post("/v1/score", json={"age": 50}).json()["warnings"] == []


# ------------------------------------------------------------------------------ batch limit


def test_batch_limit(client, audit_path):
    one = {"age": 50, "sex_female": 1}
    ok = client.post("/v1/score/batch", json={"members": [one] * schemas.MAX_BATCH})
    assert ok.status_code == 200 and ok.json()["n"] == 1000
    over = client.post("/v1/score/batch", json={"members": [one] * (schemas.MAX_BATCH + 1)})
    assert over.status_code == 422
    assert client.post("/v1/score/batch", json={"members": []}).status_code == 422
    assert client.post("/v1/score/batch", json=[one]).status_code == 422
    assert len(audit_rows(audit_path)) == 1000


def test_one_bad_member_rejects_the_whole_batch(client, audit_path):
    r = client.post("/v1/score/batch", json={"members": [{"age": 50}, {"age": 50, "x": 1}]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][:3] == ["body", "members", 1]
    assert audit_rows(audit_path) == []


# ------------------------------------------------------------------------------------ audit


def test_audit_row_written_for_each_scored_member(client, audit_path, scored, sample):
    _, X, payloads = sample
    single = client.post("/v1/score", json=payloads[0]).json()
    batch = client.post("/v1/score/batch", json={"members": payloads[1:4]}).json()
    rows = audit_rows(audit_path)
    assert len(rows) == 4
    assert set(rows[0]) == {
        "request_id",
        "item_index",
        "scored_at_utc",
        "model_version",
        "input_sha256",
        "probability",
        "tier",
    }
    first = rows[0]
    assert first["request_id"] == single["request_id"] and first["item_index"] == 0
    assert first["probability"] == single["probability"] and first["tier"] == single["tier"]
    assert first["model_version"] == config.MODEL_VERSION
    assert first["scored_at_utc"].endswith("Z") and "T" in first["scored_at_utc"]
    assert [r["request_id"] for r in rows[1:]] == [batch["request_id"]] * 3
    assert [r["item_index"] for r in rows[1:]] == [0, 1, 2]
    for row, x in zip(rows, X[:4], strict=True):
        values = [None if np.isnan(v) else float(v) for v in x]
        assert row["input_sha256"] == audit.input_sha256(values)


def test_audit_stores_no_inputs(client, audit_path, sample):
    _, _, payloads = sample
    payload = {**payloads[0], "cost_prior_12m": 123456.75}
    client.post("/v1/score", json=payload)
    client.app.state.audit.close()  # flush the write-ahead log into the file
    stored = b"".join(p.read_bytes() for p in audit_path.parent.iterdir())
    assert b"123456.75" not in stored
    assert b"cost_prior_12m" not in stored


def test_input_hash_ignores_key_order_and_number_spelling(client, audit_path):
    client.post("/v1/score", json={"age": 46, "sex_female": 1})
    client.post("/v1/score", json={"sex_female": 1.0, "age": 46.0, "bmi": None})
    client.post("/v1/score", json={"age": 47, "sex_female": 1})
    a, b, c = (r["input_sha256"] for r in audit_rows(audit_path))
    assert a == b != c


def test_audit_path_comes_from_the_environment(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere" / "log.db"
    monkeypatch.setenv(audit.ENV, str(target))
    log = audit.AuditLog()
    log.append("r1", "v", [("0" * 64, 0.5, "rising")])
    assert log.count() == 1 and target.exists()
    log.close()
