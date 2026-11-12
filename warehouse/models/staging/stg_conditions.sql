select
    patient as patient_id,
    encounter as encounter_id,
    cast("start" as date) as onset_date,
    cast("stop" as date) as stop_date,
    code as code,
    description as description
from {{ source('synthea', 'conditions') }}
