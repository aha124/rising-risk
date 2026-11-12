{# One row per model feature, in the same order as the feature columns of mart_member_year. #}

select
    name,
    label,
    feature_group as "group",
    description,
    unit,
    dtype
from {{ ref('feature_dictionary') }}
order by position
