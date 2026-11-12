-- Window boundary check.
-- For a member present in two consecutive index years, the cost in the 12 months before
-- the later index date must equal the outcome cost of the earlier index year. A gap or an
-- overlap at January 1 (for example "<=" in place of "<") breaks the equality.

select
    cur.member_id,
    cur.index_year,
    cur.cost_prior_12m,
    prev.cost_next_12m as previous_year_outcome_cost
from {{ ref('mart_member_year') }} as cur
inner join {{ ref('mart_member_year') }} as prev
    on cur.member_id = prev.member_id and cur.index_year = prev.index_year + 1
where abs(cur.cost_prior_12m - prev.cost_next_12m) > 0.01
