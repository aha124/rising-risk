"""Write artifacts/reports/pipeline.json from the dbt artifacts and the warehouse.

Run from the repo root after `dbt build`:

    .venv/bin/python warehouse/export_pipeline_json.py

Nodes and edges come from warehouse/target/manifest.json, test results from
warehouse/target/run_results.json, row counts from the DuckDB file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]

LAYERS = {"staging": "staging", "intermediate": "intermediate", "marts": "mart"}

# Singular tests shown on the site as leakage and integrity checks, in display order.
CHECKS = [
    (
        "assert_no_feature_event_on_or_after_index",
        "Features only use records dated before the index date",
    ),
    (
        "assert_outcome_events_inside_outcome_year",
        "Outcome cost only uses records inside the outcome year",
    ),
    (
        "assert_prior_cost_equals_previous_outcome",
        "Last year's cost feature equals the previous year's outcome",
    ),
    (
        "assert_cost_events_reconcile_to_source",
        "Cost events reconcile to the raw files",
    ),
    (
        "assert_cost_events_match_patient_ledger",
        "Cost events match the generator's own patient ledger",
    ),
    (
        "assert_label_prevalence_9_to_11_pct",
        "Label prevalence is between 9% and 11% in every year",
    ),
    (
        "assert_mart_rows_are_eligible",
        "Every member-year in the mart is eligible",
    ),
]


def read_generation_info(raw_path: Path) -> dict:
    info = {}
    path = raw_path / "_generation.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("Records:"):
                key, value = line.split("=", 1)
                info[key.strip()] = value.strip()
    return info


def fmt_date(value: str | None) -> str | None:
    if value and len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def days(n) -> str:
    n = int(n)
    return f"{n} day" if n == 1 else f"{n} days"


def check_details(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    def one(sql: str):
        return con.execute(sql).fetchone()

    details = {}

    n_rows = one("select count(*) from mart_member_year")[0]
    feature_models = [
        "int_cost_windows",
        "int_utilization_windows",
        "int_procedure_windows",
        "int_condition_flags",
        "int_medication_windows",
        "int_observation_latest",
        "int_payer_at_index",
    ]
    union = " union all ".join(
        f"select index_date, max_event_date from {m}" for m in feature_models
    )
    bad, closest = one(
        f"""
        select
            count(*) filter (where max_event_date >= index_date),
            min(index_date - max_event_date)
        from ({union})
        """
    )
    details["assert_no_feature_event_on_or_after_index"] = (
        f"{len(feature_models)} feature models report the latest record they read for each of "
        f"{n_rows:,} member-years. {bad} are on or after the index date. "
        f"The closest is {days(closest)} before it."
    )

    bad_outcome, first, last = one(
        """
        select
            count(*) filter (where y.min_event_date < i.index_date or y.max_event_date >= i.outcome_end_date),
            min(y.min_event_date - i.index_date),
            min(i.outcome_end_date - y.max_event_date)
        from int_outcome as y
        inner join int_member_index as i on y.member_id = i.member_id and y.index_date = i.index_date
        """
    )
    details["assert_outcome_events_inside_outcome_year"] = (
        f"{bad_outcome} of {n_rows:,} member-years have an outcome record outside "
        f"[index date, next January 1). Earliest is {days(first)} after the index date, "
        f"latest is {days(last)} before the next January 1."
    )

    pairs, mismatched = one(
        """
        select count(*), count(*) filter (where abs(cur.cost_prior_12m - prev.cost_next_12m) > 0.01)
        from mart_member_year as cur
        inner join mart_member_year as prev
            on cur.member_id = prev.member_id and cur.index_year = prev.index_year + 1
        """
    )
    details["assert_prior_cost_equals_previous_outcome"] = (
        f"{pairs:,} consecutive member-year pairs compared, {mismatched} differ by more than 1 cent. "
        "A gap or overlap at January 1 would show up here."
    )

    raw_total = one(
        """
        select
            (select sum(total_claim_cost) from stg_encounters)
            + (select sum(base_cost) from stg_medications)
        """
    )[0]
    events_total = float(one("select sum(amount) from int_cost_events")[0])
    diff = abs(events_total - raw_total) / raw_total
    details["assert_cost_events_reconcile_to_source"] = (
        f"Cost events total ${events_total:,.0f}. Raw encounter claim totals plus one fill per "
        f"medication order total ${raw_total:,.0f}. Difference {diff:.4%} (limit 0.5%). "
        "Procedures and immunizations are billed inside the encounter totals and are not added again."
    )

    n_full, ledger, events = one(
        """
        with w as (
            select min(o.observation_date) as d
            from stg_observations as o
            inner join stg_patients as p on o.patient_id = p.patient_id
            where p.death_date is null
        ),
        f as (
            select p.patient_id, p.healthcare_expenses + p.healthcare_coverage as ledger
            from stg_patients as p, w
            where p.death_date is null and p.birth_date > w.d
        ),
        e as (select member_id, sum(amount) as amount from int_cost_events group by member_id)
        select count(*), sum(f.ledger), sum(coalesce(e.amount, 0))
        from f left join e on f.patient_id = e.member_id
        """
    )
    if n_full:
        ledger_diff = abs(float(events) - ledger) / ledger
        details["assert_cost_events_match_patient_ledger"] = (
            f"For {n_full:,} living patients whose whole history is in the export, cost events "
            f"total ${float(events):,.0f} and Synthea's lifetime ledger totals ${ledger:,.0f}. "
            f"Difference {ledger_diff:.4%} (limit 0.5%)."
        )
    else:
        details["assert_cost_events_match_patient_ledger"] = (
            "No living patients with their whole history inside the export window."
        )

    low, high, years = one(
        """
        select min(p), max(p), count(*)
        from (select avg(label_high_cost) as p from mart_member_year group by index_year)
        """
    )
    details["assert_label_prevalence_9_to_11_pct"] = (
        f"Prevalence ranges from {low:.2%} to {high:.2%} across {years} index years."
    )

    details["assert_mart_rows_are_eligible"] = (
        f"All {n_rows:,} member-years are alive on the index date, 18 or older, and have at "
        "least 12 months of history. The rule is restated in the test from the patient table."
    )
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default=str(ROOT / "warehouse" / "rising_risk.duckdb"))
    parser.add_argument("--target-dir", default=str(ROOT / "warehouse" / "target"))
    parser.add_argument("--raw-path", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "reports" / "pipeline.json"))
    args = parser.parse_args()

    target = Path(args.target_dir)
    manifest = json.loads((target / "manifest.json").read_text())
    run_results = json.loads((target / "run_results.json").read_text())
    con = duckdb.connect(args.database, read_only=True)

    def count(relation: str) -> int:
        return int(con.execute(f"select count(*) from {relation}").fetchone()[0])

    ids: dict[str, str] = {}
    nodes = []

    for unique_id, src in sorted(manifest["sources"].items()):
        node_id = f"raw_{src['name']}"
        ids[unique_id] = node_id
        location = src["meta"]["external_location"].format(name=src["name"])
        nodes.append(
            {
                "id": node_id,
                "layer": "source",
                "label": f"{src['name']}.csv",
                "rows": count(location),
                "description": " ".join((src.get("description") or "").split()),
            }
        )

    for unique_id, node in sorted(manifest["nodes"].items()):
        if node["resource_type"] == "seed":
            ids[unique_id] = node["name"]
            nodes.append(
                {
                    "id": node["name"],
                    "layer": "source",
                    "label": f"{node['name']} (seed)",
                    "rows": count(node["name"]),
                    "description": " ".join((node.get("description") or "").split()),
                }
            )

    order = {"staging": 0, "intermediate": 1, "marts": 2}
    models = [n for n in manifest["nodes"].values() if n["resource_type"] == "model"]
    models.sort(key=lambda n: (order[n["fqn"][1]], n["name"]))
    for node in models:
        ids[node["unique_id"]] = node["name"]
        nodes.append(
            {
                "id": node["name"],
                "layer": LAYERS[node["fqn"][1]],
                "label": node["name"],
                "rows": count(node["name"]),
                "description": " ".join((node.get("description") or "").split()),
            }
        )

    edges = []
    for node in models:
        for parent in node["depends_on"]["nodes"]:
            if parent in ids:
                edges.append([ids[parent], node["name"]])
    edges.sort()

    status = {}
    by_type = {"generic": 0, "singular": 0}
    passed = failed = warned = 0
    for result in run_results["results"]:
        node = manifest["nodes"].get(result["unique_id"])
        if not node or node["resource_type"] != "test":
            continue
        kind = "generic" if node.get("test_metadata") else "singular"
        by_type[kind] += 1
        status[node["name"]] = result["status"]
        if result["status"] == "pass":
            passed += 1
        elif result["status"] == "warn":
            warned += 1
        else:
            failed += 1

    details = check_details(con)
    leakage_checks = []
    for name, title in CHECKS:
        leakage_checks.append(
            {
                "name": title,
                "result": "pass" if status.get(name) == "pass" else "fail",
                "detail": details[name],
                "test": name,
            }
        )

    info = read_generation_info(Path(args.raw_path))
    n_patients = count("stg_patients")
    first_event, last_event = con.execute(
        """
        select
            (
                select min(o.observation_date)
                from stg_observations as o
                inner join stg_patients as p on o.patient_id = p.patient_id
                where p.death_date is null
            ),
            (select max(start_date) from stg_encounters)
        """
    ).fetchone()

    pipeline = {
        "fixture": False,
        "source": {
            "generator": "Synthea",
            "version": " ".join(
                x
                for x in [info.get("synthea_version"), info.get("synthea_build_timestamp", "")[:10]]
                if x
            ),
            "seed": int(info["seed"]) if info.get("seed") else None,
            "n_patients": n_patients,
            "state": info.get("state"),
            "date_start": str(first_event),
            "date_end": fmt_date(info.get("end_date")) or str(last_event),
        },
        "nodes": nodes,
        "edges": edges,
        "tests": {
            "total": passed + failed + warned,
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "by_type": by_type,
        },
        "leakage_checks": leakage_checks,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pipeline, indent=2) + "\n")
    print(
        f"{out}: {len(nodes)} nodes, {len(edges)} edges, {passed}/{passed + failed + warned} tests passed"
    )


if __name__ == "__main__":
    main()
