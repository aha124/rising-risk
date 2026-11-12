-- No strings in the feature set. Every mart column that is not a key, an audit column or an
-- outcome must have a numeric type.

{% set skip = non_feature_columns() %}

select
    column_name,
    data_type
from information_schema.columns
where
    lower(table_name) = 'mart_member_year'
    and lower(table_schema) = lower('{{ ref('mart_member_year').schema }}')
    and lower(column_name) not in (
        {% for c in skip %}'{{ c }}'{% if not loop.last %}, {% endif %}{% endfor %}
    )
    and lower(data_type) not in (
        'integer', 'bigint', 'smallint', 'double', 'double precision', 'real', 'float', 'numeric'
    )
    and lower(data_type) not like 'decimal%'
    and lower(data_type) not like 'numeric%'
