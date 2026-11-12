# Warehouse

A dbt project that turns Synthea CSV files into one modeling table: one row per adult member per January 1, with features from before that date and next-year cost after it.

## Run it

All commands run from the repo root.

```bash
# 1. Generate the data (about 16 minutes for 12,000 living patients on 2 cores, about 4 GB on disk)
scripts/generate_data.sh

# 2. Classify the condition codes found in the data (rewrites warehouse/seeds/condition_groups.csv)
.venv/bin/python warehouse/build_condition_groups.py

# 3. Build and test (about 35 seconds)
.venv/bin/dbt build --project-dir warehouse --profiles-dir warehouse

# 4. Export the marts and the reports
.venv/bin/python warehouse/export_marts.py
.venv/bin/python warehouse/export_pipeline_json.py
```

Step 1 writes `data/raw/` (not committed) and `data/sample/` (150 living patients, seed 7, committed, 3.9 MB). Step 2 only needs rerunning if the data changes. The committed seed already covers every code in both the full data and the sample.

To build on the sample, change one variable. CI does this.

```bash
RR_DUCKDB_PATH=/tmp/sample.duckdb .venv/bin/dbt build --project-dir warehouse --profiles-dir warehouse \
  --vars '{raw_path: data/sample}'
```

`RR_DUCKDB_PATH` is optional. Without it the build writes `warehouse/rising_risk.duckdb`, the same file a full build uses. The export scripts take `--database`, `--raw-path`, `--marts-dir` and `--reports-dir` so a sample build can be exported somewhere other than `data/marts/`.

Outputs:

| File | What it is |
|---|---|
| `data/marts/member_year.parquet` | The modeling table, 97,570 rows |
| `data/marts/feature_dictionary.parquet` | One row per feature: name, label, group, description, unit, dtype |
| `artifacts/reports/pipeline.json` | Lineage nodes with row counts, edges, test counts, leakage check results |
| `artifacts/reports/data_profile.json` | Counts, prevalence, dollar thresholds and cost quantiles per year, missing rates, feature means per year |

## What was built

13,837 synthetic patients (12,000 alive at the end of 2025, 1,837 deceased), Pennsylvania, Synthea build `bd8a52b`, seed 42, records from January 2011 through December 2025.

The mart has 97,570 member-years from 9,950 members, index years 2015 through 2025, 59 features. About 8,400 to 9,400 members per year. 10.0% of each year is labelled high cost. The cut is between $34,796 and $41,386 depending on the year.

## Layers

`staging/` has one view per source file: rename, cast, parse dates. Nothing else.

`intermediate/` holds the logic:

- `int_member_index` lists who is eligible on each January 1.
- `int_cost_events` has one row per cost event under the cost rule below.
- `int_cost_windows`, `int_utilization_windows`, `int_procedure_windows`, `int_condition_flags`, `int_medication_windows`, `int_observation_latest`, `int_payer_at_index` each compute features for a member-year from records before the index date. Each one also reports the latest record date it read (`max_event_date`), which is what the leakage test checks.
- `int_outcome` computes next-year cost and the label.

`marts/` has `mart_member_year` (keys, then features, then `race` and `ethnicity` for the audit only, then outcomes) and `mart_feature_dictionary`.

`seeds/` has `condition_groups.csv` and `feature_dictionary.csv`.

## Time rules

The index date is January 1 of the index year.

Features use records dated strictly before the index date. Windows are months 1 to 12 before the index date (the previous calendar year) and months 13 to 24 (the year before that). Labs and vitals take the latest value in the 24 months before the index date.

The outcome uses records dated on or after the index date and before the next January 1.

A few choices inside those rules:

- An encounter belongs to the day it starts. A hospital stay that starts in December and ends in January is counted, cost included, in the December year. 104 of 7,888 inpatient stays since 2013 cross a January 1, and so do a few hundred encounters of other classes, mostly skilled nursing and hospice stays, which can be large. The cost of such an encounter includes days after the index date, so for those members the prior-cost features carry a little information from January. Splitting a claim total by day would be a guess, so I left it and am noting it here.
- Inpatient days are different: they are clipped at the index date, so a stay in progress on January 1 contributes only the days already spent.
- A condition or medication counts as active on the index date if it started before that date and had not ended before it.
- Payer is the most recent coverage period that started before the index date.

## Eligibility

A member-year is in the mart if the member is

1. alive on the index date (members who died earlier are out),
2. 18 or older on the index date, and
3. observable for at least 12 months before the index date. Observable history starts at the later of the birth date and the start of the export window (2011-01-02, read from the data).

Rule 3 never removes anyone in this build, because every adult was born more than 12 months before 2015 and the export window opens four years before the first index date. It stays in as a guard for anyone who changes the year range.

Members who die during the outcome year stay in, with their cost up to death and `died_in_outcome_year = 1` (43 to 80 per year). Synthea dates the death certification encounter in the days after the death date, at about $200. It is counted if it falls in the same year.

## Cost rule

**Total cost of care = encounter `TOTAL_CLAIM_COST` + one fill (`BASE_COST`) per medication order.**

Procedure and immunization costs are not added. Medication `TOTALCOST` is not used.

How I got there, on a 500-patient test run that included Synthea's claims files:

1. Encounter totals already contain procedures and immunizations. For every encounter, the non-medication charges on its claim add up to `TOTAL_CLAIM_COST` (24,312 of 31,330 encounters to the cent). Almost all of the rest differ by multiples of $77.60 or $7.50, amounts that sit inside the encounter total and appear in no other file. I take them to be lab charges. So adding `procedures.BASE_COST` or `immunizations.BASE_COST` would count them twice. It would also overcount, because the procedures file lists a price on rows the claim charged at $0: summed procedure prices came to more than the encounter totals they are supposed to be part of.
2. Medications are billed on separate claims and are not inside the encounter total.
3. Synthea bills one fill per medication order. For all 89 living patients whose whole life falls inside the export, `HEALTHCARE_EXPENSES + HEALTHCARE_COVERAGE` in the patients file equals the sum of their encounter totals plus the sum of medication `BASE_COST`, to the cent. That is the generator's own ledger, so the rule matches it exactly.
4. `medications.TOTALCOST` is `BASE_COST x DISPENSES`, where dispenses are extrapolated at export time from the start date to the stop date or the end of the simulation. The generator never bills those refills (medication orders with 6 or more dispenses average 1.007 charges). The extrapolation also produces nonsense: two alteplase orders, a clot-busting drug given once in an emergency, were never stopped and accounted for $30.8M of $50.1M in total medication cost in the test run.

The cost of this choice: ongoing refill spending for chronic medications is understated. Medication orders are 5% of dollars under this rule. A real claims feed would show more.

Three dbt tests guard the rule:

- `assert_cost_events_reconcile_to_source` compares the total in `int_cost_events` with encounter totals plus medication base costs read straight from the raw files (tolerance 0.5%, actual difference 0).
- `assert_cost_events_match_patient_ledger` compares lifetime cost events with Synthea's ledger for the 2,082 living patients born inside the export window (tolerance 0.5%, actual difference 0.007%). This is the one that fails if something is counted twice.
- `assert_prior_cost_equals_previous_outcome` checks that last year's cost for index year Y equals the outcome cost of index year Y-1 for the same member, across 87,620 pairs.

## Condition groups

`seeds/condition_groups.csv` has one row for each of the 308 condition codes in the data, with a category, and a chronic group for the 70 codes that belong to one of 18 groups. `build_condition_groups.py` writes it. The group assignments in that script were made by reading the list of descriptions from the generated data, most frequent first.

Synthea's conditions file is not a clean problem list. Of the ten most common "conditions", five are social findings (stress, full-time employment, part-time employment, limited social contact, social isolation), one is "Medication review due", and four are dental problems or colds. Those rows get the category `social`, `administrative`, `dental`, `symptom` and so on, no chronic group, and never reach a condition flag or count. Some rows worded as findings are diseases for this purpose and are mapped: "Body mass index 30+ - obesity (finding)" and "Prediabetes (finding)".

A few conditions are recorded as short episodes that end (stroke, myocardial infarction, overdose, a single depressive episode, acute myeloid leukemia). They are marked `episodic` and their flag stays on for 36 months after the episode ends.

`pregnant_at_index` is a separate flag, not a chronic group. Prenatal and delivery care is a large share of cost in this data, and a pregnancy in progress on January 1 is known on January 1.

`pregnancy_in_outcome_year` is its counterpart on the outcome side: 1 if a pregnancy is in progress at any point in the outcome year, including one that starts after January 1 (6,393 of 97,570 member-years, against 2,060 with `pregnant_at_index`). It uses the same condition codes read over the outcome window. It is an outcome column, so it is never a feature and is not in the feature dictionary. It exists so the evaluation can say how much of what the model finds is maternity.

## What `years_of_history` does

The setting is `exporter.years_of_history = 15` with end date 2025-12-31. I checked its behavior by generating the same 150 patients twice, once with full history and once with 15 years, and comparing row by row.

- For living patients the window opens in the first days of January 2011. Procedures, observations and immunizations are cut there with no exceptions.
- Conditions and medications are kept if they were active at any time inside the window, with their original start date. That includes ones that started decades earlier and are still active (444 of 444 conditions, 55 of 55 medications) and ones that started earlier and ended inside the window (254 of 254 and 49 of 49). Ones that ended before the window are dropped (0 of 6,203 conditions kept). So "active on the index date" is complete for every index year here.
- Encounters before the window are kept only when a kept condition or medication hangs off them (638 of 11,674). Their claim totals are unchanged. They are a biased sliver of old history, so no feature here looks back past the window: the longest lookback is 36 months, and the first index date is January 2015.
- For deceased patients the window is measured back from the date of death, so someone who died in 2019 has records from about 2004.
- The same seed gives the same patients and the same rows in both runs. Row order differs between runs, and the summary columns in `payers.csv` can differ slightly. The warehouse uses neither.

## Index years, and why 2014 is out

I first set the index years to 2014 through 2025, then dropped 2014. The reason is the generator, not the history window.

Synthea's chronic pain and opioid modules switch on in calendar 2014. In that one year 5,739 chronic pain diagnoses are recorded (90 to 160 a year after that), a set of analgesic and opioid medications appears that does not exist in 2013, and ambulatory encounters jump from 26,926 to 43,513 before falling back to 28,172.

For index year 2014 that means features come from a world without those modules (8% with chronic pain against 33% to 34% in every later year, 1.9 active medications against 2.4) while the outcome year contains the one-off wave. That year would teach the model a relationship that holds nowhere else, so it is out. Change `index_year_start` in `dbt_project.yml` to bring it back.

The wave still shows in two kept features, and I left those years in because a one-off shock in the lookback is something a model in production has to live with:

- Index 2015: `outpatient_visits_12m` averages 5.6 (3.4 to 3.7 in most other years) and `new_chronic_groups_12m` averages 0.37 (about 0.10 otherwise).
- Index 2016: `outpatient_visits_13_24m` averages 5.5.

Calendar 2021 has a second shock: 15,063 vaccination encounters. Index 2022 shows it in `outpatient_visits_12m` (5.3), index 2023 in `outpatient_visits_13_24m` (5.2), and only 2% of members have zero cost in outcome year 2021 against 8% to 10% in other years.

`feature_mean_by_year` in `data_profile.json` has every feature's mean by year.

## Tests

`dbt build` runs 156 tests on the full data and on the sample: 142 generic (unique, not_null, accepted_values, relationships, and a composite-key test) and 14 singular.

The singular tests that matter most:

| Test | What it catches |
|---|---|
| `assert_no_feature_event_on_or_after_index` | A feature model reading a record dated on or after the index date |
| `assert_outcome_events_inside_outcome_year` | Outcome cost reading outside its year |
| `assert_prior_cost_equals_previous_outcome` | A gap or overlap at January 1 |
| `assert_cost_events_reconcile_to_source` | Cost events drifting from the raw files |
| `assert_cost_events_match_patient_ledger` | Double counting |
| `assert_label_prevalence_9_to_11_pct` | A broken label |
| `assert_mart_rows_are_eligible` | An ineligible member-year, with the rule restated from the patient table |
| `assert_features_are_numeric` | A string column in the feature set |
| `assert_feature_dictionary_matches_mart` | The dictionary and the mart falling out of step |
| `assert_pregnancy_in_outcome_year_is_consistent` | The audit-only pregnancy outcome disagreeing with the condition records, or turning up in the feature dictionary |

A leakage test that cannot fail proves nothing, so `tests/warehouse/test_warehouse.py` breaks the rules on purpose. It copies the project, changes one `<` to `<=` (or adds procedure costs, or lets 17-year-olds in), rebuilds on the sample and checks that the right dbt test fails. I also ran the `<` to `<=` change by hand against the full data for all seven feature models and both ends of the outcome window. Each one was caught.

```bash
.venv/bin/python -m pytest tests/warehouse -q    # 11 tests, about a minute
```

## Postgres

`profiles.yml` defines a `postgres` target driven by `RR_PG_HOST`, `RR_PG_PORT`, `RR_PG_USER`, `RR_PG_PASSWORD`, `RR_PG_DBNAME` and `RR_PG_SCHEMA`. I have not run it. The models avoid DuckDB-only functions and use dbt's `datediff` and `dateadd` macros, but two things need doing first: load each CSV into a table in a `raw` schema with every column as text and lowercase column names, and expect to fix whatever a first run turns up. The export scripts read the DuckDB file and would need a Postgres connection instead.
