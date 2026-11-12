{#
  The modeling table. One row per eligible member per January 1 index date.
  Column order: keys, features, audit-only columns, outcome columns.
  Features use only records dated before index_date. Outcomes use only records on or after it.
#}

{% set groups = var('chronic_groups') %}

select
    -- keys
    i.member_id,
    i.index_date,
    i.index_year,

    -- features: demographics
    cast(i.age_at_index as integer) as age,
    cast(case when p.gender = 'F' then 1 else 0 end as integer) as sex_female,
    pay.payer_medicare,
    pay.payer_medicaid,
    pay.payer_dual_eligible,
    pay.payer_private,
    pay.payer_uninsured,

    -- features: prior cost
    c.cost_prior_12m,
    c.cost_prior_13_24m,
    c.cost_change_12m,
    c.medication_cost_prior_12m,
    c.max_encounter_cost_12m,
    c.inpatient_cost_share_12m,

    -- features: utilization
    u.inpatient_admits_12m,
    u.inpatient_days_12m,
    u.emergency_visits_12m,
    u.urgent_care_visits_12m,
    u.outpatient_visits_12m,
    u.wellness_visits_12m,
    u.other_visits_12m,
    u.inpatient_admits_13_24m,
    u.inpatient_days_13_24m,
    u.emergency_visits_13_24m,
    u.outpatient_visits_13_24m,
    u.months_since_last_inpatient,
    pr.procedures_12m,

    -- features: conditions
    {% for g in groups %}
    cf.has_{{ g }},
    {% endfor %}
    cf.chronic_group_count,
    cf.new_chronic_groups_12m,
    cf.pregnant_at_index,

    -- features: medications
    m.active_medications,
    m.new_medications_12m,

    -- features: vitals and labs
    o.bmi,
    o.systolic_bp,
    o.diastolic_bp,
    o.hba1c,
    o.egfr,
    o.total_cholesterol,
    o.pain_score,
    o.hba1c_measured,
    o.egfr_measured,
    o.cholesterol_measured,

    -- audit only, never model features
    p.race,
    p.ethnicity,

    -- outcome, never model features
    y.cost_next_12m,
    y.label_high_cost,
    y.died_in_outcome_year,
    y.pregnancy_in_outcome_year

from {{ ref('int_member_index') }} as i
inner join {{ ref('stg_patients') }} as p on i.member_id = p.patient_id
inner join {{ ref('int_payer_at_index') }} as pay
    on i.member_id = pay.member_id and i.index_date = pay.index_date
inner join {{ ref('int_cost_windows') }} as c
    on i.member_id = c.member_id and i.index_date = c.index_date
inner join {{ ref('int_utilization_windows') }} as u
    on i.member_id = u.member_id and i.index_date = u.index_date
inner join {{ ref('int_procedure_windows') }} as pr
    on i.member_id = pr.member_id and i.index_date = pr.index_date
inner join {{ ref('int_condition_flags') }} as cf
    on i.member_id = cf.member_id and i.index_date = cf.index_date
inner join {{ ref('int_medication_windows') }} as m
    on i.member_id = m.member_id and i.index_date = m.index_date
inner join {{ ref('int_observation_latest') }} as o
    on i.member_id = o.member_id and i.index_date = o.index_date
inner join {{ ref('int_outcome') }} as y
    on i.member_id = y.member_id and i.index_date = y.index_date
