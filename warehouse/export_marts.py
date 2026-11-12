"""Export the marts to parquet and write the data profile.

Run from the repo root after `dbt build`:

    .venv/bin/python warehouse/export_marts.py

Writes data/marts/member_year.parquet, data/marts/feature_dictionary.parquet and
artifacts/reports/data_profile.json. Use --marts-dir and --reports-dir to send a sample
build somewhere else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
NON_FEATURES = [
    "member_id",
    "index_date",
    "index_year",
    "race",
    "ethnicity",
    "cost_next_12m",
    "label_high_cost",
    "died_in_outcome_year",
    "pregnancy_in_outcome_year",
]
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def read_generation_info(raw_path: Path) -> dict:
    info = {}
    path = raw_path / "_generation.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("Records:"):
                key, value = line.split("=", 1)
                info[key.strip()] = value.strip()
            elif line.startswith("Records:"):
                info["records"] = line.strip()
    return info


def rounded(value, digits=2):
    return None if value is None else round(float(value), digits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default=str(ROOT / "warehouse" / "rising_risk.duckdb"))
    parser.add_argument("--raw-path", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--marts-dir", default=str(ROOT / "data" / "marts"))
    parser.add_argument("--reports-dir", default=str(ROOT / "artifacts" / "reports"))
    args = parser.parse_args()

    marts_dir = Path(args.marts_dir)
    reports_dir = Path(args.reports_dir)
    marts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.database, read_only=True)

    member_year = marts_dir / "member_year.parquet"
    dictionary = marts_dir / "feature_dictionary.parquet"
    con.execute(
        f"""
        copy (select * from mart_member_year order by index_date, member_id)
        to '{member_year}' (format parquet, compression zstd)
        """
    )
    con.execute(
        f"""
        copy (
            select d.*
            from mart_feature_dictionary as d
            inner join feature_dictionary as s on d.name = s.name
            order by s.position
        ) to '{dictionary}' (format parquet, compression zstd)
        """
    )

    columns = [r[0] for r in con.execute("describe mart_member_year").fetchall()]
    features = [c for c in columns if c not in NON_FEATURES]

    n_rows = con.execute("select count(*) from mart_member_year").fetchone()[0]
    n_members = con.execute("select count(distinct member_id) from mart_member_year").fetchone()[0]
    n_patients, n_living = con.execute(
        "select count(*), count(*) filter (where death_date is null) from stg_patients"
    ).fetchone()

    quantile_sql = ", ".join(
        f"quantile_cont(cost_next_12m, {q}) as q{int(q * 100):02d}" for q in QUANTILES
    )
    by_year_rows = con.execute(
        f"""
        select
            index_year,
            count(*) as n,
            sum(label_high_cost) as n_high_cost,
            avg(label_high_cost) as prevalence,
            min(case when label_high_cost = 1 then cost_next_12m end) as threshold,
            avg(cost_next_12m) as mean_cost,
            max(cost_next_12m) as max_cost,
            avg(case when cost_next_12m = 0 then 1.0 else 0.0 end) as share_zero_cost,
            sum(case when label_high_cost = 1 then cost_next_12m else 0 end) / sum(cost_next_12m)
                as top_decile_share_of_dollars,
            sum(died_in_outcome_year) as n_died,
            {quantile_sql}
        from mart_member_year
        group by index_year
        order by index_year
        """
    ).fetchall()
    names = [d[0] for d in con.description]

    member_years_by_year = {}
    prevalence_by_year = {}
    threshold_by_year = {}
    cost_quantiles_by_year = {}
    cost_summary_by_year = {}
    deaths_by_year = {}
    for row in by_year_rows:
        r = dict(zip(names, row, strict=True))
        year = str(r["index_year"])
        member_years_by_year[year] = int(r["n"])
        prevalence_by_year[year] = round(float(r["prevalence"]), 6)
        threshold_by_year[year] = rounded(r["threshold"])
        cost_quantiles_by_year[year] = {
            f"p{int(q * 100):02d}": rounded(r[f"q{int(q * 100):02d}"]) for q in QUANTILES
        }
        cost_summary_by_year[year] = {
            "mean": rounded(r["mean_cost"]),
            "max": rounded(r["max_cost"]),
            "share_zero_cost": round(float(r["share_zero_cost"]), 6),
            "top_decile_share_of_dollars": round(float(r["top_decile_share_of_dollars"]), 6),
            "n_high_cost": int(r["n_high_cost"]),
        }
        deaths_by_year[year] = int(r["n_died"])

    missing_sql = ", ".join(
        f"avg(case when {c} is null then 1.0 else 0.0 end) as {c}" for c in features
    )
    missing = con.execute(f"select {missing_sql} from mart_member_year").fetchone()
    missing_rate = {c: round(float(v), 6) for c, v in zip(features, missing, strict=True)}

    summary_sql = ", ".join(
        f"avg({c}) as {c}__mean, min({c}) as {c}__min, max({c}) as {c}__max, "
        f"avg(case when {c} = 0 then 1.0 else 0.0 end) as {c}__zero"
        for c in features
    )
    summary_row = con.execute(f"select {summary_sql} from mart_member_year").fetchone()
    feature_summary = {}
    for i, c in enumerate(features):
        mean, low, high, zero = summary_row[i * 4 : i * 4 + 4]
        feature_summary[c] = {
            "mean": rounded(mean, 4),
            "min": rounded(low, 4),
            "max": rounded(high, 4),
            "share_zero": rounded(zero, 6),
            "missing_rate": missing_rate[c],
        }

    mean_sql = ", ".join(f"avg({c}) as {c}" for c in features)
    year_rows = con.execute(
        f"select index_year, {mean_sql} from mart_member_year group by index_year order by index_year"
    ).fetchall()
    feature_mean_by_year = {
        c: {str(row[0]): rounded(row[i + 1], 4) for row in year_rows}
        for i, c in enumerate(features)
    }

    repeat = con.execute(
        """
        select
            avg(case when prev.label_high_cost = 1 then cur.label_high_cost end)
                as p_high_given_high_last_year,
            avg(case when cur.label_high_cost = 1 then prev.label_high_cost end)
                as share_of_high_who_were_high_last_year
        from mart_member_year as cur
        inner join mart_member_year as prev
            on cur.member_id = prev.member_id and cur.index_year = prev.index_year + 1
        """
    ).fetchone()

    export_window_start = con.execute(
        """
        select min(o.observation_date)
        from stg_observations as o
        inner join stg_patients as p on o.patient_id = p.patient_id
        where p.death_date is null
        """
    ).fetchone()[0]

    profile = {
        "source": read_generation_info(Path(args.raw_path)),
        "n_patients": int(n_patients),
        "n_patients_living_at_end": int(n_living),
        "n_members": int(n_members),
        "n_member_years": int(n_rows),
        "n_features": len(features),
        "features": features,
        "index_years": sorted(int(y) for y in member_years_by_year),
        "export_window_start": str(export_window_start),
        "member_years_by_year": member_years_by_year,
        "label_prevalence_by_year": prevalence_by_year,
        "high_cost_threshold_usd_by_year": threshold_by_year,
        "cost_quantiles_usd_by_year": cost_quantiles_by_year,
        "cost_summary_by_year": cost_summary_by_year,
        "deaths_in_outcome_year_by_year": deaths_by_year,
        "label_persistence": {
            "p_high_cost_given_high_cost_last_year": rounded(repeat[0], 4),
            "share_of_high_cost_who_were_high_cost_last_year": rounded(repeat[1], 4),
        },
        "missing_rate": missing_rate,
        "feature_summary": feature_summary,
        "feature_mean_by_year": feature_mean_by_year,
    }
    out = reports_dir / "data_profile.json"
    out.write_text(json.dumps(profile, indent=2) + "\n")

    print(f"{member_year}: {n_rows} rows, {len(columns)} columns, {len(features)} features")
    print(f"{dictionary}: {len(features)} rows")
    print(f"{out}")


if __name__ == "__main__":
    main()
