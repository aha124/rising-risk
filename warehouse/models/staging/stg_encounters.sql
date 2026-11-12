select
    id as encounter_id,
    patient as patient_id,
    payer as payer_id,
    cast("start" as timestamp) as start_ts,
    cast("stop" as timestamp) as stop_ts,
    cast(cast("start" as timestamp) as date) as start_date,
    cast(cast("stop" as timestamp) as date) as stop_date,
    encounterclass as encounter_class,
    code as code,
    description as description,
    cast(base_encounter_cost as double precision) as base_encounter_cost,
    cast(total_claim_cost as double precision) as total_claim_cost,
    cast(payer_coverage as double precision) as payer_coverage,
    reasoncode as reason_code,
    reasondescription as reason_description
from {{ source('synthea', 'encounters') }}
