with source as (

    select
        patient as patient_id,
        encounter as encounter_id,
        payer as payer_id,
        cast("start" as timestamp) as start_ts,
        cast("stop" as timestamp) as stop_ts,
        code as code,
        description as description,
        cast(base_cost as double precision) as base_cost,
        cast(dispenses as integer) as dispenses,
        cast(totalcost as double precision) as total_cost,
        reasoncode as reason_code
    from {{ source('synthea', 'medications') }}

)

select
    -- The file has no row id. Build one from the natural key, with a counter for exact repeats.
    md5(
        patient_id || '|' || coalesce(encounter_id, '') || '|' || code || '|' || cast(start_ts as varchar)
    ) || '-' || cast(
        row_number() over (
            partition by patient_id, encounter_id, code, start_ts
            order by stop_ts, base_cost, dispenses
        ) as varchar
    ) as medication_id,
    patient_id,
    encounter_id,
    payer_id,
    start_ts,
    stop_ts,
    cast(start_ts as date) as start_date,
    cast(stop_ts as date) as stop_date,
    code,
    description,
    base_cost,
    dispenses,
    total_cost,
    reason_code
from source
