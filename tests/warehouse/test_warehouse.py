"""Tests for the dbt warehouse, run against the committed sample in data/sample.

The first test builds the project and checks that every dbt test passes and that the mart
has the promised shape. The rest are mutation tests: each one copies the project, breaks one
rule on purpose, and checks that the dbt test guarding that rule fails. A leakage test that
cannot fail proves nothing, so these keep the guards honest.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "warehouse"
SAMPLE = ROOT / "data" / "sample"
DBT = Path(sys.executable).parent / "dbt"

pytestmark = pytest.mark.skipif(
    not DBT.exists() or not any(SAMPLE.glob("patients.csv*")),
    reason="needs dbt in the active environment and the sample data",
)

NON_FEATURES = {
    "member_id",
    "index_date",
    "index_year",
    "race",
    "ethnicity",
    "cost_next_12m",
    "label_high_cost",
    "died_in_outcome_year",
    "pregnancy_in_outcome_year",
}


def copy_project(dest: Path) -> Path:
    project = dest / "warehouse"
    shutil.copytree(
        WAREHOUSE,
        project,
        ignore=shutil.ignore_patterns("target", "logs", "dbt_packages", "*.duckdb", "*.wal"),
    )
    return project


def run_dbt(project: Path, database: Path, select: str | None = None) -> dict[str, str]:
    """Run dbt build on the sample and return {node name: status}."""
    cmd = [
        str(DBT),
        "build",
        "--project-dir",
        str(project),
        "--profiles-dir",
        str(project),
        "--vars",
        json.dumps({"raw_path": str(SAMPLE)}),
    ]
    if select:
        cmd += ["--select", select]
    env = {
        "RR_DUCKDB_PATH": str(database),
        "PATH": f"{DBT.parent}:/usr/bin:/bin",
        "HOME": str(project),
    }
    subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    results = json.loads((project / "target" / "run_results.json").read_text())
    return {r["unique_id"].split(".")[2]: r["status"] for r in results["results"]}


def mutate(project: Path, relative_path: str, old: str, new: str) -> None:
    path = project / relative_path
    text = path.read_text()
    assert text.count(old) == 1, f"expected exactly one {old!r} in {relative_path}"
    path.write_text(text.replace(old, new))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("warehouse_build")
    project = copy_project(tmp)
    database = tmp / "sample.duckdb"
    statuses = run_dbt(project, database)
    return statuses, database


def test_build_passes_on_sample(built):
    statuses, _ = built
    bad = {name: status for name, status in statuses.items() if status not in ("pass", "success")}
    assert not bad, f"dbt nodes that did not pass: {bad}"
    assert len(statuses) > 150


def test_mart_shape(built):
    _, database = built
    con = duckdb.connect(str(database), read_only=True)
    columns = con.execute("describe mart_member_year").fetchall()
    names = [c[0] for c in columns]
    types = {c[0]: c[1] for c in columns}

    assert names[:3] == ["member_id", "index_date", "index_year"]
    assert names[-6:] == [
        "race",
        "ethnicity",
        "cost_next_12m",
        "label_high_cost",
        "died_in_outcome_year",
        "pregnancy_in_outcome_year",
    ]

    features = [n for n in names if n not in NON_FEATURES]
    assert 50 <= len(features) <= 70
    for name in features:
        assert types[name] in ("INTEGER", "BIGINT", "DOUBLE"), f"{name} is {types[name]}"

    dictionary = [r[0] for r in con.execute("select name from mart_feature_dictionary").fetchall()]
    assert sorted(dictionary) == sorted(features)

    dupes = con.execute(
        "select count(*) from (select member_id, index_date from mart_member_year group by 1, 2 having count(*) > 1)"
    ).fetchone()[0]
    assert dupes == 0


def test_label_is_top_decile_each_year(built):
    _, database = built
    con = duckdb.connect(str(database), read_only=True)
    rows = con.execute(
        """
        select
            index_year,
            count(*) as n,
            sum(label_high_cost) as n_high,
            min(case when label_high_cost = 1 then cost_next_12m end) as lowest_high,
            max(case when label_high_cost = 0 then cost_next_12m end) as highest_other
        from mart_member_year
        group by index_year
        """
    ).fetchall()
    assert rows
    for _year, n, n_high, lowest_high, highest_other in rows:
        assert n_high == round(n * 0.10)
        assert lowest_high >= highest_other


MUTATIONS = [
    pytest.param(
        "models/intermediate/int_cost_windows.sql",
        "and e.event_date < i.index_date",
        "and e.event_date <= i.index_date",
        "assert_no_feature_event_on_or_after_index",
        id="prior cost reads the index date",
    ),
    pytest.param(
        "models/intermediate/int_utilization_windows.sql",
        "and e.start_date < i.index_date",
        "and e.start_date <= i.index_date",
        "assert_no_feature_event_on_or_after_index",
        id="utilization reads the index date",
    ),
    pytest.param(
        "models/intermediate/int_observation_latest.sql",
        "and o.observation_date < i.index_date",
        "and o.observation_date <= i.index_date",
        "assert_no_feature_event_on_or_after_index",
        id="labs read the index date",
    ),
    pytest.param(
        "models/intermediate/int_outcome.sql",
        "and e.event_date >= i.index_date",
        "and e.event_date > i.index_date",
        "assert_prior_cost_equals_previous_outcome",
        id="outcome skips the index date",
    ),
    pytest.param(
        "models/intermediate/int_outcome.sql",
        "and e.event_date < i.outcome_end_date",
        "and e.event_date <= i.outcome_end_date",
        "assert_outcome_events_inside_outcome_year",
        id="outcome runs into the next year",
    ),
    pytest.param(
        "models/intermediate/int_cost_events.sql",
        "round(cast(base_cost as numeric), 2) as amount",
        "round(cast(total_cost as numeric), 2) as amount",
        "assert_cost_events_match_patient_ledger",
        id="medication refills extrapolated",
    ),
    pytest.param(
        "models/intermediate/int_member_index.sql",
        "and age_at_index >= 18",
        "and age_at_index >= 17",
        "assert_mart_rows_are_eligible",
        id="minors let in",
    ),
]


@pytest.mark.parametrize(("relative_path", "old", "new", "guard"), MUTATIONS)
def test_mutation_is_caught(tmp_path, relative_path, old, new, guard):
    project = copy_project(tmp_path)
    mutate(project, relative_path, old, new)
    statuses = run_dbt(project, tmp_path / "mutated.duckdb", select=f"+{guard}")
    assert statuses.get(guard) == "fail", (
        f"{guard} did not fail after the change: {statuses.get(guard)}"
    )


def test_double_counting_procedures_is_caught(tmp_path):
    project = copy_project(tmp_path)
    path = project / "models" / "intermediate" / "int_cost_events.sql"
    path.write_text(
        path.read_text()
        + """
union all

select
    'proc-' || cast(row_number() over (order by patient_id, start_ts, code) as varchar) as cost_event_id,
    patient_id as member_id,
    start_date as event_date,
    'encounter' as cost_source,
    'procedure' as cost_category,
    encounter_id,
    round(cast(base_cost as numeric), 2) as amount
from {{ ref('stg_procedures') }}
"""
    )
    guards = ["assert_cost_events_reconcile_to_source", "assert_cost_events_match_patient_ledger"]
    statuses = run_dbt(
        project, tmp_path / "mutated.duckdb", select=" ".join(f"+{g}" for g in guards)
    )
    for guard in guards:
        assert statuses.get(guard) == "fail", f"{guard} did not fail: {statuses.get(guard)}"
