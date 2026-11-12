select
    patient as patient_id,
    payer as payer_id,
    cast(start_date as timestamp) as start_ts,
    cast(end_date as timestamp) as end_ts,
    cast(cast(start_date as timestamp) as date) as start_date,
    cast(cast(end_date as timestamp) as date) as end_date
from {{ source('synthea', 'payer_transitions') }}
