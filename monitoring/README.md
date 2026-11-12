# Monitoring

`run_drift.py` compares a current population against a reference and exits non-zero when something has moved too far. It is a thin command line over `src/riskmodel/drift.py`, which is unit tested against values worked out by hand.

```bash
.venv/bin/python monitoring/run_drift.py \
    --reference data/marts/member_year.parquet --reference-years 2015-2021 \
    --current data/marts/member_year.parquet --current-years 2025 \
    --output build/drift/2025.json
```

`make drift` runs exactly that. For each of the 59 inputs it reports the population stability index (PSI) and a Kolmogorov-Smirnov statistic against the reference. It also scores both populations with the saved model and compares the score distributions.

| PSI | Status | Exit status |
|---|---|---|
| below 0.10 | ok | 0 |
| 0.10 to 0.25 | watch | 0 (1 with `--fail-on watch`) |
| 0.25 and above | alert | 1 |

Exit status 2 means the inputs could not be read. Missing values get their own bin, so a field that stops arriving shows up as drift in that field.

Two details of the method. Bins come from the reference: deciles for continuous inputs, one bin per value for flags and small counts. And score drift is measured against members held out of training when the reference file has enough of them, because scores on rows the model trained on run high.

## What it finds on this data

Run against index year 2023, it exits 1 and names `outpatient_visits_13_24m` (PSI 0.85). Index year 2022 does the same for `outpatient_visits_12m`. Both look back on calendar 2021, when the generator recorded about 15,000 vaccination encounters. The score PSI stays under 0.10 in both years. Run against 2024 or 2025, it exits 0.

That is the realistic case: a real upstream change, a loud input alert, and a score distribution that hardly moves. Accuracy is less clear. The model stayed ahead of the cost rule in both years, but its recall on held-out members in 2022 is the lowest of any year, on a sample too small to tie that to the shift or to rule it out. The alert is worth having either way. Someone should look, find the vaccination wave, and write down that it is understood. The [model card](../MODEL_CARD.md) has the table for every year.

## The label lag

The features for index year Y exist on January 1 of year Y. Whether a member ended up high cost is known 12 months later, and in a real claims feed a few months after that, once claims have run out.

So on the day the model is used, the only things a monitor can check are inputs and scores. Accuracy for this year's list cannot be measured until next year. Input drift is an early warning that arrives a year before the evidence that would confirm it, and it has to be read with that in mind: an alert is a reason to find the cause, not proof the model is wrong.

## How I would schedule it

The model scores once a year, so the checks follow that calendar.

| When | Check | Action on failure |
|---|---|---|
| Each scoring run, before the list goes out | `run_drift.py` on the new population against the training years. Also the share of members in each tier and the missing rate per input | Exit 1 holds the list until someone has found the cause |
| Monthly, if the mart is refreshed between runs | The same command on the refreshed inputs | Notify, no hold |
| Once the outcome year has closed and claims have run out | Recall at 10% against the cost rule on the same members, calibration, the subgroup table | Retrain if the paired gain over the rule is no longer distinguishable from zero |

Any scheduler that reads an exit status works. A cron entry is enough:

```
0 6 2 1 *  cd /srv/rising-risk && .venv/bin/python monitoring/run_drift.py --reference ref.parquet --current current.parquet --output reports/drift-$(date +\%Y).json || mail -s "rising-risk drift alert" oncall@example.org < /dev/null
```

The JSON report keeps the full per-input table, so the alert can link to it.

The API writes an audit row for every score (`api/audit.py`). In a deployment where scores are served on request instead of in a yearly batch, the probabilities in that table are the input for a rolling score-drift check. That check is not built here.
