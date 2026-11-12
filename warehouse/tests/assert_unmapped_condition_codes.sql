-- Every condition code in the data should have a row in seeds/condition_groups.csv, so each
-- one has been looked at and classified. A new Synthea build or seed can add codes, so this
-- warns and does not fail the build. Unmapped codes are left out of the condition flags.

{{ config(severity = 'warn') }}

select
    c.code,
    min(c.description) as description,
    count(distinct c.patient_id) as n_patients
from {{ ref('stg_conditions') }} as c
left join {{ ref('condition_groups') }} as g on c.code = g.code
where g.code is null
group by c.code
