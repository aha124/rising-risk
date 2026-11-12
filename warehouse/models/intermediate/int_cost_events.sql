{#
  One row per cost event.

  Rule: total cost of care = encounter TOTAL_CLAIM_COST + one fill (BASE_COST) per medication order.

  - Encounter TOTAL_CLAIM_COST already contains the visit charge and the procedures,
    immunizations and labs billed on that claim. Adding procedures.BASE_COST or
    immunizations.BASE_COST on top would count them twice.
  - Medications are billed on their own claims and are not inside the encounter total.
    Synthea bills one fill per order. medications.TOTALCOST (BASE_COST x DISPENSES) is an
    export-time extrapolation that the generator never bills, so it is not used.
  - An encounter's whole cost is dated at its start date, including multi-day inpatient stays.

  See warehouse/README.md for the evidence.
#}

select
    'enc-' || encounter_id as cost_event_id,
    patient_id as member_id,
    start_date as event_date,
    'encounter' as cost_source,
    encounter_class as cost_category,
    encounter_id,
    round(cast(total_claim_cost as numeric), 2) as amount
from {{ ref('stg_encounters') }}

union all

select
    'med-' || medication_id as cost_event_id,
    patient_id as member_id,
    start_date as event_date,
    'medication' as cost_source,
    'pharmacy' as cost_category,
    encounter_id,
    round(cast(base_cost as numeric), 2) as amount
from {{ ref('stg_medications') }}
