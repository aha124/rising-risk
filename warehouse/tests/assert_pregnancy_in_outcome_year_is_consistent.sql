-- pregnancy_in_outcome_year is an audit-only outcome column. Three things must hold:
-- a member pregnant on the index date is pregnant in the outcome year, the flag is never set
-- for a member with no pregnancy record overlapping the outcome year, and it is not listed
-- in the feature dictionary.

select
    m.member_id,
    m.index_date,
    'pregnant at index but not flagged for the outcome year' as problem
from {{ ref('mart_member_year') }} as m
where m.pregnant_at_index = 1 and m.pregnancy_in_outcome_year = 0

union all

select
    m.member_id,
    m.index_date,
    'flagged without a pregnancy record overlapping the outcome year' as problem
from {{ ref('mart_member_year') }} as m
inner join {{ ref('int_member_index') }} as i
    on m.member_id = i.member_id and m.index_date = i.index_date
where
    m.pregnancy_in_outcome_year = 1
    and not exists (
        select 1
        from {{ ref('stg_conditions') }} as c
        inner join {{ ref('condition_groups') }} as g on c.code = g.code
        where
            c.patient_id = m.member_id
            and g.category = 'pregnancy'
            and c.onset_date < i.outcome_end_date
            and (c.stop_date is null or c.stop_date >= i.index_date)
    )

union all

select
    cast(null as varchar) as member_id,
    cast(null as date) as index_date,
    'outcome column listed in the feature dictionary' as problem
from {{ ref('mart_feature_dictionary') }}
where name = 'pregnancy_in_outcome_year'
