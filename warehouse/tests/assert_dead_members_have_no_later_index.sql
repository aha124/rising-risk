-- Members who died before an index date are out. Members who die during the outcome year stay in.

select
    m.member_id,
    m.index_date,
    p.death_date,
    m.died_in_outcome_year
from {{ ref('mart_member_year') }} as m
inner join {{ ref('stg_patients') }} as p on m.member_id = p.patient_id
where
    p.death_date < m.index_date
    or (m.died_in_outcome_year = 1 and (p.death_date is null or extract(year from p.death_date) <> m.index_year))
    or (m.died_in_outcome_year = 0 and extract(year from p.death_date) = m.index_year)
