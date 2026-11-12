-- Leakage check for the outcome.
-- Outcome cost may only use events on or after the index date and before the next January 1.

select
    y.member_id,
    y.index_date,
    y.min_event_date,
    y.max_event_date
from {{ ref('int_outcome') }} as y
inner join {{ ref('int_member_index') }} as i
    on y.member_id = i.member_id and y.index_date = i.index_date
where
    y.min_event_date < i.index_date
    or y.max_event_date >= i.outcome_end_date
