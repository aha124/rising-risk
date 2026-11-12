-- The mart must not drop anyone: same member-years as int_member_index, no more, no fewer.

select
    coalesce(i.member_id, m.member_id) as member_id,
    coalesce(i.index_date, m.index_date) as index_date
from {{ ref('int_member_index') }} as i
full outer join {{ ref('mart_member_year') }} as m
    on i.member_id = m.member_id and i.index_date = m.index_date
where i.member_id is null or m.member_id is null
