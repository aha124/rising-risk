select
    patient as patient_id,
    encounter as encounter_id,
    cast(cast("date" as timestamp) as date) as immunization_date,
    code as code,
    description as description,
    cast(base_cost as double precision) as base_cost
from {{ source('synthea', 'immunizations') }}
