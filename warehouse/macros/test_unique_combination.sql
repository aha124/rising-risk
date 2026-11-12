{% test unique_combination(model, columns) %}
{# Fails for every combination of the given columns that appears more than once. #}
select
    {{ columns | join(', ') }},
    count(*) as n_rows
from {{ model }}
group by {{ columns | join(', ') }}
having count(*) > 1
{% endtest %}
