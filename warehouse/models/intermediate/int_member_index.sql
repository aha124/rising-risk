{#
  One row per eligible member per January 1 index date.

  Eligible means: alive on the index date, 18 or older on the index date, and at least
  12 months of observable history before the index date.
#}

with patients as (

    select * from {{ ref('stg_patients') }}

),

export_window as (

    -- Synthea's years_of_history setting trims procedures and observations to the export
    -- window with no exceptions, so the earliest one for a living patient marks the first
    -- date from which records are complete.
    select min(o.observation_date) as export_window_start
    from {{ ref('stg_observations') }} as o
    inner join patients as p on o.patient_id = p.patient_id
    where p.death_date is null

),

index_dates as (

    {% for y in range(var('index_year_start'), var('index_year_end') + 1) %}
    select
        {{ y }} as index_year,
        cast('{{ y }}-01-01' as date) as index_date,
        cast('{{ y - 1 }}-01-01' as date) as window_12m_start,
        cast('{{ y - 2 }}-01-01' as date) as window_24m_start,
        cast('{{ y - 3 }}-01-01' as date) as window_36m_start,
        cast('{{ y - 4 }}-01-01' as date) as window_48m_start,
        cast('{{ y + 1 }}-01-01' as date) as outcome_end_date
    {% if not loop.last %}union all{% endif %}
    {% endfor %}

),

candidates as (

    select
        p.patient_id as member_id,
        i.index_date,
        i.index_year,
        i.window_12m_start,
        i.window_24m_start,
        i.window_36m_start,
        i.window_48m_start,
        i.outcome_end_date,
        p.birth_date,
        p.death_date,
        w.export_window_start,
        case
            when p.birth_date > w.export_window_start then p.birth_date
            else w.export_window_start
        end as history_start_date,
        -- Age in whole years on January 1.
        i.index_year - cast(extract(year from p.birth_date) as integer)
        - case
            when extract(month from p.birth_date) = 1 and extract(day from p.birth_date) = 1 then 0
            else 1
        end as age_at_index
    from patients as p
    cross join index_dates as i
    cross join export_window as w

)

select
    member_id,
    index_date,
    index_year,
    window_12m_start,
    window_24m_start,
    window_36m_start,
    window_48m_start,
    outcome_end_date,
    birth_date,
    death_date,
    history_start_date,
    age_at_index,
    case
        when death_date >= index_date and death_date < outcome_end_date then 1
        else 0
    end as died_in_outcome_year
from candidates
where
    (death_date is null or death_date >= index_date)
    and age_at_index >= 18
    and history_start_date <= window_12m_start
