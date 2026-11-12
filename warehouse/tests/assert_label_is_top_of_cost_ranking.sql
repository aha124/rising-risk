-- Within an index year, no unlabelled member may have a higher outcome cost than a labelled one.

with bounds as (

    select
        index_year,
        min(case when label_high_cost = 1 then cost_next_12m end) as lowest_labelled_cost,
        max(case when label_high_cost = 0 then cost_next_12m end) as highest_unlabelled_cost
    from {{ ref('mart_member_year') }}
    group by index_year

)

select *
from bounds
where highest_unlabelled_cost > lowest_labelled_cost
