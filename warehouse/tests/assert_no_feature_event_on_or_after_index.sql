-- Leakage check for features.
-- Every feature model reports the latest source event date it read for each member-year.
-- That date must be strictly before the index date. Changing any "< index_date" filter
-- to "<=" lets a January 1 event through and this test returns rows.

{% set feature_models = [
    'int_cost_windows',
    'int_utilization_windows',
    'int_procedure_windows',
    'int_condition_flags',
    'int_medication_windows',
    'int_observation_latest',
    'int_payer_at_index'
] %}

{% for m in feature_models %}
select
    '{{ m }}' as feature_model,
    member_id,
    index_date,
    max_event_date
from {{ ref(m) }}
where max_event_date >= index_date
{% if not loop.last %}union all{% endif %}
{% endfor %}
