select
    patient as patient_id,
    encounter as encounter_id,
    cast("date" as timestamp) as observation_ts,
    cast(cast("date" as timestamp) as date) as observation_date,
    category as category,
    code as code,
    description as description,
    "value" as value_text,
    units as units,
    "type" as value_type
from {{ source('synthea', 'observations') }}
