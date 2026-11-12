{#
  Medication features per member per index date. Only orders that start strictly before
  the index date are read. Active means not stopped before the index date.
#}

with idx as (

    select * from {{ ref('int_member_index') }}

),

joined as (

    select
        i.member_id,
        i.index_date,
        m.code,
        m.start_date,
        (m.stop_date is null or m.stop_date >= i.index_date) as is_active,
        m.start_date >= i.window_12m_start as started_12m
    from idx as i
    inner join {{ ref('stg_medications') }} as m
        on i.member_id = m.patient_id and m.start_date < i.index_date

),

agg as (

    select
        member_id,
        index_date,
        count(distinct code) filter (where is_active) as active_medications,
        count(distinct code) filter (where started_12m) as new_medications_12m,
        max(start_date) filter (where is_active or started_12m) as max_event_date
    from joined
    group by member_id, index_date

)

select
    i.member_id,
    i.index_date,
    cast(coalesce(a.active_medications, 0) as integer) as active_medications,
    cast(coalesce(a.new_medications_12m, 0) as integer) as new_medications_12m,
    a.max_event_date
from idx as i
left join agg as a
    on i.member_id = a.member_id and i.index_date = a.index_date
