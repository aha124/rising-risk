-- The feature dictionary and the feature columns of the mart must list the same names.

{% set skip = non_feature_columns() %}

with mart_columns as (

    select lower(column_name) as name
    from information_schema.columns
    where
        lower(table_name) = 'mart_member_year'
        and lower(table_schema) = lower('{{ ref('mart_member_year').schema }}')
        and lower(column_name) not in (
            {% for c in skip %}'{{ c }}'{% if not loop.last %}, {% endif %}{% endfor %}
        )

),

dictionary as (

    select name from {{ ref('mart_feature_dictionary') }}

)

select
    coalesce(m.name, d.name) as name,
    case when m.name is null then 'in dictionary only' else 'in mart only' end as problem
from mart_columns as m
full outer join dictionary as d on m.name = d.name
where m.name is null or d.name is null
