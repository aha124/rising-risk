-- Every member-year in the mart must be eligible. The rule is restated here from
-- stg_patients so the test does not depend on int_member_index being right:
-- alive on the index date, 18 or older, born at least 12 months before the index date,
-- and the index date at least 12 months after the start of the export window.

with export_window as (

    select min(o.observation_date) as export_window_start
    from {{ ref('stg_observations') }} as o
    inner join {{ ref('stg_patients') }} as p on o.patient_id = p.patient_id
    where p.death_date is null

)

select
    m.member_id,
    m.index_date,
    p.birth_date,
    p.death_date
from {{ ref('mart_member_year') }} as m
inner join {{ ref('stg_patients') }} as p on m.member_id = p.patient_id
cross join export_window as w
where
    p.death_date < m.index_date
    or {{ dbt.dateadd('year', 18, 'p.birth_date') }} > m.index_date
    or {{ dbt.dateadd('month', 12, 'w.export_window_start') }} > m.index_date
    or m.index_year < {{ var('index_year_start') }}
    or m.index_year > {{ var('index_year_end') }}
    or m.age <> {{ dbt.datediff('p.birth_date', 'm.index_date', 'year') }}
        - case when extract(month from p.birth_date) = 1 and extract(day from p.birth_date) = 1 then 0 else 1 end
