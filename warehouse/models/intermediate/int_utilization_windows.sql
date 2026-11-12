{#
  Encounter counts per member per index date. Uses encounters that start strictly before
  the index date. Inpatient days are clipped at the index date so a stay still in progress
  on January 1 contributes only the days already spent.
#}

with idx as (

    select * from {{ ref('int_member_index') }}

),

encounters as (

    select * from {{ ref('stg_encounters') }}

),

joined as (

    select
        i.member_id,
        i.index_date,
        e.encounter_class,
        e.start_date,
        e.start_date >= i.window_12m_start as in_12m,
        e.start_date >= i.window_24m_start and e.start_date < i.window_12m_start as in_13_24m,
        case
            when e.encounter_class = 'inpatient' then
                case
                    when
                        {{ dbt.datediff(
                            'e.start_date',
                            'case when e.stop_date < i.index_date then e.stop_date else i.index_date end',
                            'day'
                        ) }} < 1 then 1
                    else
                        {{ dbt.datediff(
                            'e.start_date',
                            'case when e.stop_date < i.index_date then e.stop_date else i.index_date end',
                            'day'
                        ) }}
                end
            else 0
        end as inpatient_days
    from idx as i
    inner join encounters as e
        on
            i.member_id = e.patient_id
            and e.start_date >= i.window_36m_start
            and e.start_date < i.index_date

),

agg as (

    select
        member_id,
        index_date,
        count(*) filter (where in_12m and encounter_class = 'inpatient') as inpatient_admits_12m,
        coalesce(sum(inpatient_days) filter (where in_12m), 0) as inpatient_days_12m,
        count(*) filter (where in_12m and encounter_class = 'emergency') as emergency_visits_12m,
        count(*) filter (where in_12m and encounter_class = 'urgentcare') as urgent_care_visits_12m,
        count(*) filter (
            where in_12m and encounter_class in ('outpatient', 'ambulatory')
        ) as outpatient_visits_12m,
        count(*) filter (where in_12m and encounter_class = 'wellness') as wellness_visits_12m,
        count(*) filter (
            where in_12m and encounter_class in ('snf', 'hospice', 'home', 'virtual')
        ) as other_visits_12m,
        count(*) filter (where in_13_24m and encounter_class = 'inpatient') as inpatient_admits_13_24m,
        coalesce(sum(inpatient_days) filter (where in_13_24m), 0) as inpatient_days_13_24m,
        count(*) filter (where in_13_24m and encounter_class = 'emergency') as emergency_visits_13_24m,
        count(*) filter (
            where in_13_24m and encounter_class in ('outpatient', 'ambulatory')
        ) as outpatient_visits_13_24m,
        max(start_date) filter (where encounter_class = 'inpatient') as last_inpatient_date,
        max(start_date) as max_event_date
    from joined
    group by member_id, index_date

)

select
    i.member_id,
    i.index_date,
    cast(coalesce(a.inpatient_admits_12m, 0) as integer) as inpatient_admits_12m,
    cast(coalesce(a.inpatient_days_12m, 0) as integer) as inpatient_days_12m,
    cast(coalesce(a.emergency_visits_12m, 0) as integer) as emergency_visits_12m,
    cast(coalesce(a.urgent_care_visits_12m, 0) as integer) as urgent_care_visits_12m,
    cast(coalesce(a.outpatient_visits_12m, 0) as integer) as outpatient_visits_12m,
    cast(coalesce(a.wellness_visits_12m, 0) as integer) as wellness_visits_12m,
    cast(coalesce(a.other_visits_12m, 0) as integer) as other_visits_12m,
    cast(coalesce(a.inpatient_admits_13_24m, 0) as integer) as inpatient_admits_13_24m,
    cast(coalesce(a.inpatient_days_13_24m, 0) as integer) as inpatient_days_13_24m,
    cast(coalesce(a.emergency_visits_13_24m, 0) as integer) as emergency_visits_13_24m,
    cast(coalesce(a.outpatient_visits_13_24m, 0) as integer) as outpatient_visits_13_24m,
    -- Null when there was no inpatient stay in the 36 months before the index date.
    cast(
        {{ dbt.datediff('a.last_inpatient_date', 'i.index_date', 'month') }} as integer
    ) as months_since_last_inpatient,
    a.max_event_date
from idx as i
left join agg as a
    on i.member_id = a.member_id and i.index_date = a.index_date
