-- Double counting check against an independent number.
-- patients.HEALTHCARE_EXPENSES + HEALTHCARE_COVERAGE is Synthea's own lifetime ledger for
-- each patient. For living patients born inside the export window the files hold their
-- whole history, so lifetime cost events must add up to the ledger. Fails if the totals
-- differ by more than 0.5%, which is what happens if procedures, immunizations or
-- extrapolated medication refills are added on top of the encounter totals.

with export_window as (

    select min(o.observation_date) as export_window_start
    from {{ ref('stg_observations') }} as o
    inner join {{ ref('stg_patients') }} as p on o.patient_id = p.patient_id
    where p.death_date is null

),

full_history_patients as (

    select
        p.patient_id,
        p.healthcare_expenses + p.healthcare_coverage as ledger_amount
    from {{ ref('stg_patients') }} as p
    cross join export_window as w
    where p.death_date is null and p.birth_date > w.export_window_start

),

events as (

    select
        e.member_id,
        sum(e.amount) as amount
    from {{ ref('int_cost_events') }} as e
    group by e.member_id

),

totals as (

    select
        count(*) as n_patients,
        sum(f.ledger_amount) as ledger_amount,
        sum(coalesce(e.amount, 0)) as cost_events_amount
    from full_history_patients as f
    left join events as e on f.patient_id = e.member_id

)

select *
from totals
where
    n_patients > 0
    and abs(cost_events_amount - ledger_amount) > 0.005 * ledger_amount
