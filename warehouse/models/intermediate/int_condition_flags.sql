{#
  Chronic condition group flags per member per index date.

  Only condition rows with an onset strictly before the index date are read. A group is
  flagged when one of its conditions is active on the index date. For the few conditions
  Synthea records as short episodes (marked episodic in the seed, for example a stroke),
  the flag stays on for 36 months after the episode ends.

  "New" means flagged on the index date and not flagged 12 months earlier under the same rule.

  pregnant_at_index is separate from the chronic groups: a pregnancy that began before the
  index date and had not ended by it.
#}

{% set groups = var('chronic_groups') %}

with idx as (

    select * from {{ ref('int_member_index') }}

),

mapped as (

    select
        c.patient_id,
        c.onset_date,
        c.stop_date,
        g.chronic_group,
        g.episodic
    from {{ ref('stg_conditions') }} as c
    inner join {{ ref('condition_groups') }} as g on c.code = g.code
    where g.chronic_group is not null

),

hits as (

    select
        i.member_id,
        i.index_date,
        m.chronic_group,
        max(
            case
                when m.stop_date is null then 1
                when m.episodic = 1 and m.stop_date >= i.window_36m_start then 1
                when m.episodic = 0 and m.stop_date >= i.index_date then 1
                else 0
            end
        ) as flag_at_index,
        max(
            case
                when m.onset_date >= i.window_12m_start then 0
                when m.stop_date is null then 1
                when m.episodic = 1 and m.stop_date >= i.window_48m_start then 1
                when m.episodic = 0 and m.stop_date >= i.window_12m_start then 1
                else 0
            end
        ) as flag_12m_earlier,
        max(m.onset_date) as max_event_date
    from idx as i
    inner join mapped as m
        on i.member_id = m.patient_id and m.onset_date < i.index_date
    group by i.member_id, i.index_date, m.chronic_group

),

pregnancy as (

    -- A pregnancy that began before the index date and had not ended by it.
    select
        i.member_id,
        i.index_date,
        1 as pregnant_at_index,
        max(c.onset_date) as max_event_date
    from idx as i
    inner join {{ ref('stg_conditions') }} as c
        on i.member_id = c.patient_id and c.onset_date < i.index_date
    inner join {{ ref('condition_groups') }} as g on c.code = g.code
    where
        g.category = 'pregnancy'
        and (c.stop_date is null or c.stop_date >= i.index_date)
    group by i.member_id, i.index_date

),

wide as (

    select
        member_id,
        index_date,
        {% for g in groups %}
        max(case when chronic_group = '{{ g }}' then flag_at_index else 0 end) as has_{{ g }},
        {% endfor %}
        sum(flag_at_index) as chronic_group_count,
        sum(case when flag_at_index = 1 and flag_12m_earlier = 0 then 1 else 0 end) as new_chronic_groups_12m,
        max(max_event_date) as max_event_date
    from hits
    group by member_id, index_date

)

select
    i.member_id,
    i.index_date,
    {% for g in groups %}
    cast(coalesce(w.has_{{ g }}, 0) as integer) as has_{{ g }},
    {% endfor %}
    cast(coalesce(w.chronic_group_count, 0) as integer) as chronic_group_count,
    cast(coalesce(w.new_chronic_groups_12m, 0) as integer) as new_chronic_groups_12m,
    cast(coalesce(pg.pregnant_at_index, 0) as integer) as pregnant_at_index,
    case
        when pg.max_event_date is null then w.max_event_date
        when w.max_event_date is null or pg.max_event_date > w.max_event_date then pg.max_event_date
        else w.max_event_date
    end as max_event_date
from idx as i
left join wide as w
    on i.member_id = w.member_id and i.index_date = w.index_date
left join pregnancy as pg
    on i.member_id = pg.member_id and i.index_date = pg.index_date
