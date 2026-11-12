select
    id as payer_id,
    name as payer_name,
    ownership as ownership
from {{ source('synthea', 'payers') }}
