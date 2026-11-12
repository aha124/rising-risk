{% macro non_feature_columns() %}
    {{ return([
        'member_id', 'index_date', 'index_year',
        'race', 'ethnicity',
        'cost_next_12m', 'label_high_cost', 'died_in_outcome_year',
        'pregnancy_in_outcome_year'
    ]) }}
{% endmacro %}
