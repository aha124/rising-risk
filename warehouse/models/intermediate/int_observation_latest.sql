{#
  Latest vital sign and lab value in the 24 months before the index date, null if none.
  Only observations dated strictly before the index date are read.
#}

{% set measures = {
    'bmi': '39156-5',
    'systolic_bp': '8480-6',
    'diastolic_bp': '8462-4',
    'hba1c': '4548-4',
    'egfr': '33914-3',
    'total_cholesterol': '2093-3',
    'pain_score': '72514-3'
} %}

with idx as (

    select * from {{ ref('int_member_index') }}

),

obs as (

    select
        patient_id,
        observation_ts,
        observation_date,
        code,
        cast(value_text as double precision) as value_num
    from {{ ref('stg_observations') }}
    where
        value_type = 'numeric'
        and code in (
            {% for name, code in measures.items() %}'{{ code }}'{% if not loop.last %}, {% endif %}{% endfor %}
        )

),

ranked as (

    select
        i.member_id,
        i.index_date,
        o.code,
        o.value_num,
        o.observation_date,
        row_number() over (
            partition by i.member_id, i.index_date, o.code
            order by o.observation_ts desc, o.value_num desc
        ) as rn
    from idx as i
    inner join obs as o
        on
            i.member_id = o.patient_id
            and o.observation_date >= i.window_24m_start
            and o.observation_date < i.index_date

),

latest as (

    select
        member_id,
        index_date,
        {% for name, code in measures.items() %}
        max(case when code = '{{ code }}' then value_num end) as {{ name }},
        {% endfor %}
        max(observation_date) as max_event_date
    from ranked
    where rn = 1
    group by member_id, index_date

)

select
    i.member_id,
    i.index_date,
    {% for name, code in measures.items() %}
    l.{{ name }},
    {% endfor %}
    cast(case when l.hba1c is not null then 1 else 0 end as integer) as hba1c_measured,
    cast(case when l.egfr is not null then 1 else 0 end as integer) as egfr_measured,
    cast(case when l.total_cholesterol is not null then 1 else 0 end as integer) as cholesterol_measured,
    l.max_event_date
from idx as i
left join latest as l
    on i.member_id = l.member_id and i.index_date = l.index_date
