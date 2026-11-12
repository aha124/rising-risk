{# Procedure count in the 12 months before the index date. #}

with idx as (

    select * from {{ ref('int_member_index') }}

)

select
    i.member_id,
    i.index_date,
    cast(count(p.patient_id) as integer) as procedures_12m,
    max(p.start_date) as max_event_date
from idx as i
left join {{ ref('stg_procedures') }} as p
    on
        i.member_id = p.patient_id
        and p.start_date >= i.window_12m_start
        and p.start_date < i.index_date
group by i.member_id, i.index_date
