select
    patient as patient_id,
    encounter as encounter_id,
    cast("start" as timestamp) as start_ts,
    cast(cast("start" as timestamp) as date) as start_date,
    code as code,
    description as description,
    cast(base_cost as double precision) as base_cost
from {{ source('synthea', 'procedures') }}
