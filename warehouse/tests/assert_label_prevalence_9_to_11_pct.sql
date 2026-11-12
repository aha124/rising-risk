-- The high-cost label is the top decile within each index year, so prevalence per year
-- must sit between 9% and 11%.

select
    index_year,
    count(*) as n_members,
    avg(label_high_cost) as prevalence
from {{ ref('mart_member_year') }}
group by index_year
having avg(label_high_cost) < 0.09 or avg(label_high_cost) > 0.11
