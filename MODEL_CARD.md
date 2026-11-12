# Model card: rising-risk model, version 2026.11.0

Every table in this file is written by `scripts/update_readme_metrics.py` from `artifacts/reports/*.json`. CI fails if a table is out of date.

## What it is

A gradient boosted tree model (LightGBM, 137 trees) that scores adult members on January 1 for the chance that their cost of care over the next 12 months lands in the top 10% of members. It was trained on synthetic patients from Synthea. It exists to answer one question: at a fixed outreach capacity, does a model find more of next year's high-cost members than ranking by last year's cost?

## Intended use

Ranking members for a yearly care navigation outreach list, in the setting it was built for: one score per member per January 1, a team that can call a fixed share of members, and a comparison against the cost rule at the same capacity.

On this data it is a worked example of how I would build, test, ship and watch such a model. It is not a model to deploy. A real version would be retrained and re-evaluated on real claims before anyone was called.

## Out of scope

- Any decision about a real person. The model has never seen a real patient.
- Clinical decisions, diagnosis, or triage.
- Coverage, pricing, underwriting, eligibility or benefit decisions. A cost prediction used that way would penalize people for being sick.
- Deciding who would benefit from outreach. The model ranks by expected cost. Whether a call changes anything for a member is a causal question, and answering it needs an experiment.
- Finding likely pregnancies. The model does this as a side effect (see below), and that is a reason to handle maternity separately, not a feature to use.
- Children. Members under 18 on the index date are not in the data it was trained on.

## Data

Synthea synthetic patients, Pennsylvania, seed 42, records from January 2011 through December 2025: 13,837 patients, of whom 9,950 are adults eligible in at least one year. The modeling table has 97,570 member-years, one per eligible member per January 1, for index years 2015 through 2025. A dbt project builds it, and [warehouse/README.md](warehouse/README.md) documents the cost rule, the eligibility rule, the time rules and what the generator does.

A member-year is eligible if the member is alive and 18 or older on the index date. Features use records dated strictly before the index date. The outcome uses records from the index date up to the next January 1. dbt tests enforce both, and a set of mutation tests breaks each rule on purpose to confirm the guarding test fails.

Two things about this data matter for reading the results. First, Synthea's disease modules are rule-driven, which makes next-year cost easier to predict than it is in real claims. Second, total cost is encounter claim totals plus one fill per medication order, so ongoing refill spending is understated.

## Label

`label_high_cost` is 1 if the member's cost of care in the outcome year is in the top 10% of eligible members for that index year. Ties at the cut are broken by member id. Prevalence is 10.0% in every year by construction. The dollar cut moves between about $34,800 and $41,400 depending on the year. Members who die during the outcome year stay in, with their cost up to death.

The label is defined by capacity, not by a dollar amount, because the decision is "which tenth do we call". That means the cut for an index year is set from that year's own outcomes, test years included. It defines the target and is never an input: the model ranks members, and the outreach list is the top tenth by score, so using the model needs no dollar cut and nothing from the outcome year.

## Features

59 numeric inputs, all known before the index date: demographics (age, sex, payer category), prior cost (last 12 months, the 12 before that, the change, the largest single encounter, inpatient share, medication cost), utilization counts for both windows, 18 chronic condition flags and their counts, a flag for a pregnancy in progress on the index date, medication counts, and the latest vitals and labs in the prior 24 months (missing where none was recorded).

Race and ethnicity are never inputs. They are used only in the subgroup audit below. Next-year cost, the label, death in the outcome year and pregnancy during the outcome year are outcome columns. The code refuses to build a feature matrix that names any of them, and a test scrambles them and checks that the fitted model does not change.

## Training

| Step | Rows | Members |
|---|---|---|
| Train | Index years 2015 to 2021 | The 80% not held out |
| Choose hyperparameters | Index year 2022 | The 80% not held out |
| Fit the calibrator and tier cut points | Index year 2023 | The 80% not held out |
| Test | Index years 2024 and 2025 | Everyone |

20% of members, chosen by a hash of the member id, are kept out of every fitting step in every year. They appear only in evaluation.

Three approaches are compared. The cost rule ranks by cost in the last 12 months and involves no fitting. The logistic regression uses median imputation, log scaling of dollar and count features, standardization and an L2 penalty chosen on the tuning year. The tree model is LightGBM with a binary objective, at most 31 leaves and 400 trees, early stopping on tuning-year AUPRC, and 30 Optuna trials with a fixed seed. Isotonic regression fit on the calibration year maps its score to a probability.

Members are ranked by the raw (uncalibrated) score, with ties broken by member id. Isotonic calibration produces ties, so the calibrated probability is for reading and for tiers, and the raw score is for ranking. Every top-k number in this card uses that ranking.

Tiers come from the calibrated probability, with cut points frozen from the calibration year and aimed at the top 5% ("high") and the next 15% ("rising"). Everyone else is "routine". Ties in the calibrated probability make both tiers a little larger than their targets.

## Evaluation

The primary metric was fixed before any model was fit: recall at 10%. Of the members who ended up high cost, what share were in the tenth the team would have called.

<!-- evaluation:start -->
| Approach | Recall at 10% | Recall at 5% | Recall at 20% | Dollars at 10% | AUPRC | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient boosted trees | 56.3% (54.2% to 58.1%) | 41.2% (39.5% to 42.8%) | 73.2% (71.4% to 75.2%) | 39.6% (37.7% to 41.4%) | 0.635 (0.610 to 0.658) | 0.889 (0.882 to 0.897) |
| Logistic regression | 50.7% (48.5% to 52.8%) | 38.4% (36.8% to 40.0%) | 65.0% (62.9% to 67.1%) | 34.4% (32.5% to 36.6%) | 0.537 (0.511 to 0.563) | 0.843 (0.832 to 0.853) |
| Last year's cost | 48.7% (46.6% to 50.7%) | 26.7% (25.1% to 28.5%) | 60.5% (58.5% to 62.5%) | 34.9% (32.8% to 37.0%) | 0.363 (0.341 to 0.390) | 0.770 (0.757 to 0.783) |

| Paired difference | Recall at 10%, points | Dollars at 10%, points | AUPRC | AUROC |
| --- | ---: | ---: | ---: | ---: |
| Trees minus last year's cost | +7.6 (+6.0 to +9.0) | +4.8 (+3.7 to +5.7) | +0.272 (+0.252 to +0.289) | +0.120 (+0.108 to +0.131) |
| Logistic regression minus last year's cost | +2.0 (+0.6 to +3.5) | -0.4 (-1.7 to +0.9) | +0.173 (+0.153 to +0.192) | +0.073 (+0.063 to +0.083) |
| Trees minus logistic regression | +5.5 (+3.9 to +6.9) | +5.2 (+3.6 to +6.5) | +0.098 (+0.084 to +0.112) | +0.047 (+0.041 to +0.053) |

18,613 member-years, 1,861 high cost, index years 2024 and 2025. 95% intervals from 1,000 bootstrap resamples of members. The same resamples are used for every approach, so the differences are paired.
<!-- evaluation:end -->

The tree model beats the cost rule on the primary metric, and the interval on the paired difference excludes zero. It also beats the logistic regression. The logistic regression's gain over the cost rule is small on recall and absent on dollars.

The share of next-year dollars reached at 10% rises as well, by fewer points than the share of members.

Results on members the model never saw in training, against members it did:

<!-- holdout:start -->
| Test rows | Member-years | High cost | Model, recall at 10% | Last year's cost | Difference, points | Model AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Members with rows in training | 14,640 | 1,435 | 57.0% (54.6% to 59.2%) | 50.2% | +6.8 (+5.2 to +8.4) | 0.893 |
| Members never seen in training | 3,973 | 426 | 52.8% (49.5% to 57.3%) | 43.7% | +9.2 (+6.3 to +13.2) | 0.876 |
<!-- holdout:end -->

The gain over the cost rule holds for unseen members. Their recall is a few points lower, with a wide interval on a smaller sample.

## Where the gain comes from

This section changes how the headline should be read, so it comes before the rest.

<!-- decomposition:start -->
| High-cost member-years in the test set, at 10% outreach | Count |
| --- | ---: |
| Reached by both | 839 |
| Reached only by the cost rule | 67 |
| Reached only by the model | 208 |
| &nbsp;&nbsp;of those: pregnancy during the outcome year | 126 (61%) |
| &nbsp;&nbsp;of those: pregnancy that began after January 1 | 120 (58%) |
| &nbsp;&nbsp;of those: pregnant on January 1 | 6 (3%) |
| &nbsp;&nbsp;of those: not a woman 18 to 49 | 76 (37%) |
| &nbsp;&nbsp;&nbsp;&nbsp;of those: chronic kidney disease | 69 |
| &nbsp;&nbsp;&nbsp;&nbsp;of those: diabetes | 71 |

886 of the 1,861 high-cost member-years (48%) include a pregnancy during the outcome year. The members only the model reaches are not near misses for the cost rule: their median position in the prior-cost ranking is the 24th percentile from the top, and 35% are in the bottom half.
<!-- decomposition:end -->

Most of what the model adds over the cost rule is maternity. In this synthetic population, close to half of all high-cost member-years include a pregnancy. A pregnancy that starts during the year is invisible on January 1, but a model with age and sex can rank women of childbearing age higher, and that is what it learned to do. Sex is its most important feature and age its second.

The pregnancy flag is not the explanation. It marks a pregnancy already under way on January 1, and those members already had high cost last year, so the cost rule calls nearly all of them.

The fair question is whether the model does anything beyond that. Ranking each group on its own:

<!-- strata:start -->
| Stratum | Member-years | High cost | Share of all high cost | Model, recall at 10% | Last year's cost, recall at 10% | Difference, points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Women 18 to 49 | 4,863 | 945 | 51% | 36.2% (34.1% to 38.4%) | 20.2% (17.9% to 22.4%) | +16.0 (+13.5 to +18.5) |
| Everyone else | 13,750 | 916 | 49% | 68.1% (65.1% to 70.8%) | 59.8% (56.5% to 63.0%) | +8.3 (+6.0 to +10.3) |

Each stratum is ranked on its own, and the top 10% of that stratum is called.
<!-- strata:end -->

It beats the rule in both. Outside women 18 to 49, nearly all of the extra members it reaches have chronic kidney disease and diabetes. They are people whose cost last year was not in the top tenth and whose conditions made a high-cost year likely.

The gain also holds up under each of these checks:

<!-- sensitivity:start -->
| Check | Model, recall at 10% | Rule, recall at 10% | Difference, points |
| --- | ---: | ---: | ---: |
| Full model, full test set | 56.3% (54.2% to 58.1%) | 48.7% (46.6% to 50.7%) | +7.6 (+6.0 to +9.0) |
| Test set without members pregnant on January 1 (383 rows dropped) | 52.6% (50.2% to 54.8%) | 42.5% (40.2% to 45.1%) | +10.0 (+8.1 to +11.6) |
| Model refit without the pregnancy feature | 55.3% (53.3% to 57.2%) | 48.7% (46.6% to 50.7%) | +6.6 (+5.2 to +8.1) |
| Model refit without the 6 prior-cost features | 56.2% (54.1% to 58.0%) | 48.7% (46.6% to 50.7%) |  |
| Full model against a stronger rule: pregnant members first, then by cost | 56.3% (54.2% to 58.1%) | 48.8% (46.8% to 50.7%) | +7.5 (+6.0 to +8.8) |
<!-- sensitivity:end -->

Dropping all six prior-cost features costs almost nothing. The model gets nearly the same ranking from utilization, conditions and labs. That says the cost rule's information is mostly redundant with the clinical record in this data. It may not be in real claims.

My position: a care team would not use a list of women likely to become pregnant as a rising-risk outreach list. In production, maternity belongs in its own program with its own outreach, and this model should be judged on everyone else. The cost of that position is that it sets aside the stratum where the model's advantage over the rule is largest, and leaves a smaller gain to justify the work of running a model.

I kept the headline on the whole test set anyway, because that was the metric chosen before the results were known.

What the extra members are not: the analysis does not show the model anticipating new diagnoses, first admissions or worsening lab values. It was not tested for that.

## Calibration

<!-- calibration:start -->
| Isotonic calibration | Brier score | Expected calibration error | Slope | Intercept |
| --- | ---: | ---: | ---: | ---: |
| Before (sigmoid of the raw score) | 0.0554 (0.0526 to 0.0581) | 0.0065 (0.0049 to 0.0101) | 1.07 (1.03 to 1.10) | 0.11 (0.01 to 0.21) |
| After | 0.0555 (0.0526 to 0.0582) | 0.0044 (0.0033 to 0.0082) | 0.94 (0.90 to 0.98) | -0.06 (-0.16 to 0.04) |
| Change | +0.0000 (-0.0002 to +0.0003) | -0.0021 (-0.0045 to +0.0012) |  |  |
<!-- calibration:end -->

Isotonic calibration made no measurable difference. The uncalibrated score was already close to observed rates, and the intervals on the change in Brier score and in calibration error both include zero. The calibrator stays in the pipeline because the tiers and the API read a probability, and because a model retrained on other data may not come out calibrated. It should not be described as having helped here.

## Subgroups

<!-- subgroups:start -->
| Group | Member-years | High cost | AUROC | Observed / expected | Share called at 10% | Model, recall at 10% | Last year's cost | Difference, points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sex: Female | 9,453 | 1,347 | 0.839 | 1.03 | 14.8% | 54.3% (51.7% to 56.9%) | 46.5% | +7.8 |
| Sex: Male | 9,160 | 514 | 0.924 | 1.03 | 5.1% | 61.3% (56.5% to 65.5%) | 54.3% | +7.0 |
| Age band: 18 to 34 | 5,169 | 729 | 0.874 | 1.04 | 18.0% | 59.1% (55.5% to 62.3%) | 46.1% | +13.0 |
| Age band: 35 to 49 | 4,537 | 271 | 0.853 | 1.00 | 3.8% | 39.1% (33.5% to 44.6%) | 43.2% | -4.1 |
| Age band: 50 to 64 | 5,066 | 362 | 0.845 | 1.05 | 5.7% | 48.9% (43.0% to 55.2%) | 40.9% | +8.0 |
| Age band: 65 to 79 | 2,834 | 333 | 0.836 | 1.00 | 10.3% | 58.3% (52.5% to 64.0%) | 54.1% | +4.2 |
| Age band: 80 and over | 1,007 | 166 | 0.957 | 1.02 | 17.6% | 83.7% (77.7% to 89.2%) | 75.3% | +8.4 |
| Race: white | 15,334 | 1,535 | 0.891 | 1.04 | 9.8% | 55.9% (53.3% to 58.3%) | 48.3% | +7.6 |
| Race: black | 2,131 | 229 | 0.882 | 0.99 | 12.2% | 59.0% (52.1% to 65.4%) | 52.4% | +6.6 |
| Race: asian | 668 | 60 | 0.886 | 1.05 | 8.2% | 53.3% (39.2% to 67.3%) | 41.7% | +11.7 |
| Race: other (small) | 212 | 11 | 0.911 | 0.55 | 9.0% | 54.5% (0.0% to 85.7%) | 54.5% | +0.0 |
| Race: hawaiian (small) | 191 | 20 | 0.914 | 1.06 | 11.0% | 65.0% (38.1% to 86.4%) | 55.0% | +10.0 |
| Race: native (small) | 77 | 6 | 0.765 | 0.86 | 6.5% | 50.0% (0.0% to 100.0%) | 50.0% | +0.0 |
| Ethnicity: nonhispanic | 17,342 | 1,738 | 0.892 | 1.03 | 9.9% | 56.2% (53.6% to 58.4%) | 49.0% | +7.2 |
| Ethnicity: hispanic | 1,271 | 123 | 0.860 | 0.97 | 10.9% | 56.9% (47.4% to 65.7%) | 44.7% | +12.2 |

One ranking of the whole test set, top 10% called. Recall is the share of a group's high-cost members who were in that top 10%. Groups with fewer than 300 test rows or fewer than 30 high-cost members are marked small. Their numbers are shown for completeness and are too noisy to read.
<!-- subgroups:end -->

Where it does worse:

- **Ages 35 to 49.** Recall at 10% is lower than the cost rule's for this band. This is the one age band where the model is no better than the rule and may be worse. The model calls a smaller share of this band than the rule does: it has learned to favor younger women and older members with chronic disease, and this band is neither.
- **Women against men.** Recall is lower for women than for men, under the model and under the rule. High cost among women in this data is dominated by maternity, which is harder to see coming than chronic disease.
- **Small groups.** Three race groups have too few test rows to read. Their rows are in the table for completeness.

Observed over expected is close to 1 for every group that is large enough to read, so the probabilities are not systematically off for any of them.

## The cost-proxy caveat

Obermeyer and colleagues (Science, 2019) showed that a widely used risk model, by predicting cost, ranked Black patients lower than equally sick White patients, because less was spent on their care at the same level of illness. Any model with cost as its target can do this.

The check here follows theirs: within each risk score decile, compare the number of active chronic condition groups between race groups. If one group is sicker at the same score, the score is under-representing their need.

<!-- proxy:start -->
| Comparison at the same risk score decile | Member-years | Gap in active chronic condition groups |
| --- | ---: | ---: |
| Black minus white | 2,131 | +0.15 (+0.04 to +0.26) |
| Asian minus white | 668 | +0.03 (-0.14 to +0.22) |
| Other minus white (small) | 212 | +0.22 (-0.09 to +0.51) |
| Hawaiian minus white (small) | 191 | +0.05 (-0.27 to +0.37) |
| Native minus white (small) | 77 | +0.22 (-0.28 to +0.73) |
<!-- proxy:end -->

At the same score, Black members carry slightly more chronic condition groups than White members, the direction the paper reported. The gap is small against a test-set average of about 2.5 groups. The other comparisons show nothing clear or are too small to read.

This says nothing about a real population. Synthea does not simulate the access and spending differences that produce the bias in real claims, so a small gap here is not evidence that a model built this way would be fair on real data. On real data this check would need to be run again, and a target closer to need than cost (for example avoidable admissions or active chronic conditions) should be considered.

## Monitoring plan

`monitoring/run_drift.py` compares a current population against the training years: population stability index and a Kolmogorov-Smirnov statistic for each input and for the model score. It exits non-zero when any PSI reaches 0.25, so a scheduler can alert on it. [monitoring/README.md](monitoring/README.md) has the schedule.

Labels lag inputs by a year. Features for index year Y exist on January 1 of Y. Whether a member ended up high cost is known 12 months later, plus claims runout in a real feed. So input and score drift are the only signals available at scoring time, and accuracy can only be checked a year behind. The plan is:

1. On each scoring run: input PSI, score PSI, share of members in each tier, missing rate per input. Alert on PSI at or above 0.25.
2. Once a year's labels are complete: recall at 10% against the cost rule on the same members, calibration, and the subgroup table. Retrain if the paired gain over the rule is no longer distinguishable from zero.
3. On any alert: find the cause before deciding anything. An input alert is a question, not a verdict.

What the data showed, year by year:

<!-- drift:start -->
| Index year | Split | Inputs at alert | Inputs at watch | Largest input PSI | Score PSI | Model, recall at 10% | Last year's cost | Member-years | Labels mature |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2015 | train | 0 | 2 | `new_chronic_groups_12m` (0.23) | 0.003 | 50.3% | 44.7% | 1,622 | yes |
| 2016 | train | 0 | 1 | `outpatient_visits_13_24m` (0.14) | 0.004 | 53.8% | 45.0% | 1,641 | yes |
| 2017 | train | 0 | 0 | `new_chronic_groups_12m` (0.01) | 0.001 | 54.1% | 51.9% | 1,656 | yes |
| 2018 | train | 0 | 0 | `new_chronic_groups_12m` (0.01) | 0.001 | 54.0% | 47.2% | 1,671 | yes |
| 2019 | train | 0 | 0 | `new_chronic_groups_12m` (0.01) | 0.002 | 58.3% | 48.8% | 1,698 | yes |
| 2020 | train | 0 | 0 | `new_chronic_groups_12m` (0.01) | 0.003 | 53.5% | 48.3% | 1,714 | yes |
| 2021 | train | 0 | 0 | `new_chronic_groups_12m` (0.01) | 0.003 | 53.7% | 43.3% | 1,733 | yes |
| 2022 | tune | 1 | 3 | `outpatient_visits_12m` (0.84) | 0.010 | 46.6% | 40.4% | 1,763 | yes |
| 2023 | calibrate | 1 | 1 | `outpatient_visits_13_24m` (0.85) | 0.060 | 53.6% | 47.0% | 1,788 | yes |
| 2024 | test | 0 | 0 | `age` (0.02) | 0.005 | 56.6% | 49.2% | 9,252 (all members) | yes |
| 2025 | test | 0 | 0 | `age` (0.03) | 0.006 | 55.9% | 48.1% | 9,361 (all members) | no |

Reference: index years 2015 to 2021. Watch at PSI 0.10, alert at 0.25. Recall before the test years is on held-out members only, because the model trained on the others. In the test years it is on everyone.
<!-- drift:end -->

The last point in the plan comes from this table. Index years 2022 and 2023 each have an input at alert level: outpatient visit counts, pushed up by a wave of vaccination encounters in calendar 2021. The score distribution stayed below the watch line in both years, and the model stayed ahead of the cost rule. Recall on held-out members in 2022 is the lowest of any year. With fewer than 200 high-cost held-out members a year, that is about what year-to-year noise looks like in this table, so I cannot tie it to the input shift, and I would not rule it out either. A monitor that retrained on every input alert would have retrained twice on thin evidence. The tuning and calibration years are the two affected years, which is worth knowing when reading their numbers.

The drift report also includes two simulated upstream failures on the latest year (a lab feed that stops, and prior cost arriving 15% too high), to show which checks fire.

## Limits

- **Synthetic data.** One generator, one state, one seed. The disease modules are rules, so the problem is easier than real claims, and none of these numbers should be expected to transfer.
- **Cost as the target.** See the cost-proxy caveat. Cost also rewards predicting expensive events that outreach cannot change, and maternity is the clearest case.
- **No causal claim.** The model says who will be expensive. It does not say who will benefit from a call.
- **Uneven across ages.** Ages 35 to 49 get no benefit from the model over the cost rule.
- **Medication cost is understated**, because the generator bills one fill per order.
- **Stays that cross January 1.** An encounter's whole cost is dated at its start, so a stay that begins in December and ends in January puts some January cost into the prior-cost features. This affects a few hundred encounters (inpatient, skilled nursing, hospice). It favors the prior-cost rule at least as much as the model. See `warehouse/README.md`.
- **No claims lag.** Features assume every December claim is known on January 1. Real claims arrive weeks or months late, so a production version would score from an earlier cutoff.
- **One test period.** Two index years from the same generator run. The intervals reflect sampling of members, not what a different year or a different population would do.
- **The tuning and calibration years carry a known input shift.** The model still beat the rule in both, but those years were not clean.

## What this card does not claim

That the model is clinically valid. That it would save money. That calibration improved it. That it works equally well across ages. That the race comparison says anything about a real population. That these results would hold on real claims.

## Reproducing the numbers

```bash
make setup
make warehouse   # needs data/raw, see README
make train
```

`artifacts/model/metadata.json` records the training time, the hash of the mart, the commit, the hyperparameters and library versions. With the same mart, the model file comes out byte-identical. Runs are also logged to a local MLflow store under `mlruns/`.
