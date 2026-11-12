-- Reconciliation to the raw files under the documented cost rule:
--   total cost = sum of encounters.TOTAL_CLAIM_COST + sum of medications.BASE_COST
-- The raw side is read straight from the source, not from staging. Fails if the two totals
-- differ by more than 0.5%. Adding procedure or immunization line items, or medication
-- TOTALCOST, to int_cost_events moves the total far past that tolerance.

with raw_total as (

    select
        (select sum(cast(total_claim_cost as double precision)) from {{ source('synthea', 'encounters') }})
        + (select sum(cast(base_cost as double precision)) from {{ source('synthea', 'medications') }})
        as amount

),

events_total as (

    select sum(amount) as amount from {{ ref('int_cost_events') }}

)

select
    r.amount as raw_amount,
    e.amount as cost_events_amount,
    abs(e.amount - r.amount) / r.amount as relative_difference
from raw_total as r
cross join events_total as e
where abs(e.amount - r.amount) / r.amount > 0.005 or e.amount is null or r.amount is null
