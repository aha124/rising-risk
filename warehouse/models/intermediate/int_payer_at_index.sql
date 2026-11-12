{#
  Payer category on the index date: the most recent coverage period that started strictly
  before the index date.
#}

with idx as (

    select * from {{ ref('int_member_index') }}

),

ranked as (

    select
        i.member_id,
        i.index_date,
        p.payer_name,
        p.ownership,
        t.start_date,
        row_number() over (
            partition by i.member_id, i.index_date
            order by t.start_ts desc, t.end_ts desc
        ) as rn
    from idx as i
    inner join {{ ref('stg_payer_transitions') }} as t
        on i.member_id = t.patient_id and t.start_date < i.index_date
    inner join {{ ref('stg_payers') }} as p on t.payer_id = p.payer_id

)

select
    i.member_id,
    i.index_date,
    cast(case when r.payer_name = 'Medicare' then 1 else 0 end as integer) as payer_medicare,
    cast(case when r.payer_name = 'Medicaid' then 1 else 0 end as integer) as payer_medicaid,
    cast(case when r.payer_name = 'Dual Eligible' then 1 else 0 end as integer) as payer_dual_eligible,
    cast(case when r.ownership = 'PRIVATE' then 1 else 0 end as integer) as payer_private,
    cast(case when r.payer_name = 'NO_INSURANCE' or r.payer_name is null then 1 else 0 end as integer)
        as payer_uninsured,
    r.start_date as max_event_date
from idx as i
left join ranked as r
    on i.member_id = r.member_id and i.index_date = r.index_date and r.rn = 1
