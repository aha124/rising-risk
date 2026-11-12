select
    id as patient_id,
    cast(birthdate as date) as birth_date,
    cast(deathdate as date) as death_date,
    gender as gender,
    race as race,
    ethnicity as ethnicity,
    marital as marital_status,
    county as county,
    cast(healthcare_expenses as double precision) as healthcare_expenses,
    cast(healthcare_coverage as double precision) as healthcare_coverage,
    cast(income as double precision) as income
from {{ source('synthea', 'patients') }}
