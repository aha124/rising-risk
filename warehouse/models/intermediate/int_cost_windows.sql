{# Prior cost features. Uses cost events dated strictly before the index date. #}

with idx as (

    select * from {{ ref('int_member_index') }}

),

events as (

    select * from {{ ref('int_cost_events') }}

),

agg as (

    select
        i.member_id,
        i.index_date,
        coalesce(sum(e.amount) filter (where e.event_date >= i.window_12m_start), 0) as cost_prior_12m,
        coalesce(sum(e.amount) filter (where e.event_date < i.window_12m_start), 0) as cost_prior_13_24m,
        coalesce(sum(e.amount) filter (
            where e.event_date >= i.window_12m_start and e.cost_category = 'inpatient'
        ), 0) as inpatient_cost_12m,
        coalesce(sum(e.amount) filter (
            where e.event_date >= i.window_12m_start and e.cost_source = 'medication'
        ), 0) as medication_cost_prior_12m,
        coalesce(max(e.amount) filter (
            where e.event_date >= i.window_12m_start and e.cost_source = 'encounter'
        ), 0) as max_encounter_cost_12m,
        max(e.event_date) as max_event_date
    from idx as i
    left join events as e
        on
            i.member_id = e.member_id
            and e.event_date >= i.window_24m_start
            and e.event_date < i.index_date
    group by i.member_id, i.index_date

)

select
    member_id,
    index_date,
    cast(cost_prior_12m as double precision) as cost_prior_12m,
    cast(cost_prior_13_24m as double precision) as cost_prior_13_24m,
    cast(cost_prior_12m - cost_prior_13_24m as double precision) as cost_change_12m,
    cast(medication_cost_prior_12m as double precision) as medication_cost_prior_12m,
    cast(max_encounter_cost_12m as double precision) as max_encounter_cost_12m,
    -- Null when there was no cost at all in the last 12 months.
    case
        when cost_prior_12m > 0
            then cast(round(inpatient_cost_12m / cost_prior_12m, 4) as double precision)
    end as inpatient_cost_share_12m,
    max_event_date
from agg
