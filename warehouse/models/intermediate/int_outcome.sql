{#
  Outcome per member per index date: cost of care from the index date up to (not including)
  the next January 1, and the high-cost label.

  The label is the top share (10%) of cost within each index year, over eligible members.
  Ties at the cut are broken by member_id so the result is deterministic. Members who die
  during the outcome year stay in, with their cost up to death.

  pregnancy_in_outcome_year is an audit column, never a feature: 1 if a pregnancy is in
  progress at any point in the outcome year. That covers a pregnancy already under way on the
  index date (the pregnant_at_index feature) and one that starts during the year. It uses the
  same condition codes as pregnant_at_index, read over the outcome window instead.
#}

with idx as (

    select * from {{ ref('int_member_index') }}

),

cost as (

    select
        i.member_id,
        i.index_date,
        i.index_year,
        i.died_in_outcome_year,
        cast(coalesce(sum(e.amount), 0) as double precision) as cost_next_12m,
        min(e.event_date) as min_event_date,
        max(e.event_date) as max_event_date
    from idx as i
    left join {{ ref('int_cost_events') }} as e
        on
            i.member_id = e.member_id
            and e.event_date >= i.index_date
            and e.event_date < i.outcome_end_date
    group by i.member_id, i.index_date, i.index_year, i.died_in_outcome_year

),

pregnancy as (

    -- A pregnancy that began before the next January 1 and had not ended before the index date.
    select distinct
        i.member_id,
        i.index_date
    from idx as i
    inner join {{ ref('stg_conditions') }} as c
        on i.member_id = c.patient_id and c.onset_date < i.outcome_end_date
    inner join {{ ref('condition_groups') }} as g on c.code = g.code
    where
        g.category = 'pregnancy'
        and (c.stop_date is null or c.stop_date >= i.index_date)

),

ranked as (

    select
        cost.*,
        cast(case when pg.member_id is null then 0 else 1 end as integer)
            as pregnancy_in_outcome_year,
        row_number() over (
            partition by cost.index_year
            order by cost.cost_next_12m desc, cost.member_id
        ) as cost_rank,
        count(*) over (partition by cost.index_year) as cohort_size
    from cost
    left join pregnancy as pg
        on cost.member_id = pg.member_id and cost.index_date = pg.index_date

)

select
    member_id,
    index_date,
    index_year,
    cost_next_12m,
    cast(
        case when cost_rank <= round(cohort_size * {{ var('label_top_share') }}) then 1 else 0 end
        as integer
    ) as label_high_cost,
    died_in_outcome_year,
    pregnancy_in_outcome_year,
    cost_rank,
    cohort_size,
    min_event_date,
    max_event_date
from ranked
