// "Where the gain comes from": who the model's extra finds are, the comparison
// inside each of the two groups, and the variations the gain has to survive.

import { $, h, fmt, get, val, isNum, onResize, dataTable, setHTML, numSpan, withCI } from "./util.js";
import { intervalRows } from "./chart.js";

const signedPts = (x) => fmt.signed(x * 100, 1);
const range = (I) => (I && isNum(I.lo) ? `${signedPts(I.lo)} to ${signedPts(I.hi)}` : "");

function dots(n, cls) {
  const box = h("div", { class: "units", "aria-hidden": "true" });
  for (let i = 0; i < n; i++) box.append(h("i", { class: cls }));
  return box;
}

function unitChart(el, groups) {
  el.replaceChildren(...groups.map((g) =>
    h("div", { class: "unit-group", style: `flex:${Math.max(g.model + g.rule, 12)} 1 ${g.model + g.rule > 60 ? 220 : 150}px` },
      h("h4", { text: g.name }),
      h("p", { class: "unit-count" }, h("b", { text: fmt.int(g.model) }), " only the model found"),
      dots(g.model, "u-fill"),
      h("p", { class: "unit-count" }, h("b", { text: fmt.int(g.rule) }), " only the rule found"),
      dots(g.rule, "u-ring"),
      h("p", { class: "unit-net", text: `Net ${fmt.signed(g.model - g.rule, 0)}` }))
  ));
}

// Who the model's extra finds are outside women aged 18 to 49, as far as the data file says.
function restSentence(gd) {
  const n = gd.found_by_model_only_everyone_else;
  const ckd = gd.found_by_model_only_everyone_else_with_chronic_kidney_disease;
  const dm = gd.found_by_model_only_everyone_else_with_diabetes;
  if (!isNum(n) || n <= 0 || !isNum(ckd) || !isNum(dm)) return "";
  return ` Of the ${numSpan(fmt.int(n))} members only the model found outside women aged 18 to 49, ${numSpan(fmt.int(ckd))} have chronic kidney disease and ${numSpan(fmt.int(dm))} have diabetes on record before January 1.`;
}

export function initGain(summary, perf) {
  const sens = get(perf, "sensitivity");
  const gd = get(sens, "gain_decomposition");
  if (!sens || !gd) throw new Error("sensitivity block is missing");
  const hd = get(summary, "headline", {});
  const k = fmt.pct0(get(hd, "k", 0.1));

  // ---- who the extra finds are ----
  const only = gd.found_by_model_only, onlyRule = gd.found_by_prior_only;
  const w = gd.found_by_model_only_women_18_49_not_pregnant, wRule = gd.found_by_prior_only_women_18_49_not_pregnant;
  const pg = gd.found_by_model_only_pregnant, pgRule = gd.found_by_prior_only_pregnant || 0;
  const groups = [
    { name: "Women 18 to 49, not pregnant on January 1", model: w, rule: wRule },
    { name: "Pregnant on January 1", model: pg, rule: pgRule },
    { name: "Everyone else", model: only - w - pg, rule: onlyRule - wRule - pgRule },
  ].filter((g) => isNum(g.model) && isNum(g.rule));
  const share = isNum(sens.gain_share_women_18_49_not_pregnant) ? sens.gain_share_women_18_49_not_pregnant : w / only;
  const maternity = share >= 0.5;

  let lead = `${maternity ? "Mostly from one group." : "From more than one place."} At ${numSpan(k)} capacity the two lists share ${numSpan(fmt.int(gd.found_by_both))} high-cost members. ` +
    `The model finds ${numSpan(fmt.int(only))} that the rule misses, and the rule finds ${numSpan(fmt.int(onlyRule))} that the model misses. ` +
    `Of the model's ${numSpan(fmt.int(only))}, ${numSpan(fmt.int(w))} (${numSpan(fmt.pct0(share))}) are women aged 18 to 49 who were not pregnant on January 1.`;
  setHTML("gain-finding", lead);

  // what makes that group expensive: stated only as far as the data file says it
  const inYear = sens.gain_share_pregnancy_in_outcome_year, starting = sens.gain_share_pregnancy_starting_in_outcome_year;
  const allPos = sens.test_pos_share_pregnancy_in_outcome_year;
  const parts = [];
  if (isNum(starting)) {
    parts.push(`${numSpan(fmt.pct0(starting))} of the members only the model found had a pregnancy that began after January 1` +
      (isNum(inYear) ? `, and ${numSpan(fmt.pct0(inYear))} had one in progress at some point in the year` : "") + "." +
      (isNum(allPos) ? ` Among all high-cost members in the test set, ${numSpan(fmt.pct0(allPos))} did, so maternity is a large part of what high cost means in this synthetic population.` : "") +
      " The model has no way to see a pregnancy that has not started. It ranks these members high because of their age and sex, and in this population that is often enough.");
    // The demographic group and the pregnancy outcome are different sets. Say so when their counts coincide.
    const nInYear = gd.found_by_model_only_pregnancy_in_outcome_year, nStart = gd.found_by_model_only_pregnancy_starting_in_outcome_year;
    if (isNum(nInYear) && isNum(nStart) && isNum(w) && fmt.pct0(nInYear / only) === fmt.pct0(share)) {
      parts.push(`Those two shares match by coincidence, and they count different things. ${numSpan(fmt.int(w))} of the ${numSpan(fmt.int(only))} are women aged 18 to 49 who were not pregnant on January 1, whether or not a pregnancy followed. ` +
        `${numSpan(fmt.int(nInYear))} had a pregnancy at some point in the year: ${numSpan(fmt.int(nStart))} that began after January 1 and ${numSpan(fmt.int(nInYear - nStart))} already under way on that date.`);
    }
  } else {
    parts.push("In this synthetic population, high cost among women of that age is mostly maternity care, and the model ranks them high because of their age and sex. It has no way to see a pregnancy that has not started.");
  }
  if (isNum(pg) && isNum(gd.pregnant_in_prior_top) && isNum(gd.pregnant_test_rows)) {
    parts.push(`The pregnancy flag itself explains almost none of the gain. Only ${numSpan(fmt.int(pg))} of the ${numSpan(fmt.int(only))} were pregnant on January 1, because the rule already calls ${numSpan(fmt.int(gd.pregnant_in_prior_top))} of the ${numSpan(fmt.int(gd.pregnant_test_rows))} members who were.`);
  }
  const rk = get(gd, "model_only_prior_cost_rank");
  if (rk && isNum(rk.median)) {
    parts.push(`The members only the model found were not near misses for the rule. Half of them sat below the top ${numSpan(fmt.pct0(rk.median))} by last year's cost` +
      (isNum(rk["share_beyond_0.50"]) ? `, and ${numSpan(fmt.pct0(rk["share_beyond_0.50"]))} were in the cheaper half of all members` : "") + ".");
  }
  $("gain-detail").innerHTML = parts.map((t) => `<p>${t}</p>`).join("");
  setHTML("unit-sub", `High-cost members that one list holds and the other does not, at ${k} capacity. Each dot is one member-year. Net is the model's count minus the rule's.`);
  unitChart($("chart-units"), groups);
  $("table-units").replaceChildren(dataTable(
    [{ label: "Group" }, { label: "Only the model found", num: true }, { label: "Only the rule found", num: true }, { label: "Net", num: true }],
    [...groups.map((g) => [g.name, fmt.int(g.model), fmt.int(g.rule), fmt.signed(g.model - g.rule, 0)]),
      ["All high-cost members", fmt.int(only), fmt.int(onlyRule), fmt.signed(only - onlyRule, 0)]]
  ));

  // ---- the comparison inside each group ----
  const strata = [
    { key: "women_18_49", name: "Women aged 18 to 49", el: $("chart-stratum-a"), head: $("stratum-a-head") },
    { key: "everyone_else", name: "Everyone else", el: $("chart-stratum-b"), head: $("stratum-b-head") },
  ].map((st) => ({ ...st, d: get(sens, `by_childbearing_stratum.${st.key}`) })).filter((st) => st.d);
  const tops = strata.flatMap((st) => ["lgbm", "prior_cost"].map((m) => get(st.d, `${m}.recall_at_10.hi`, val(get(st.d, `${m}.recall_at_10`)))));
  const max = Math.min(1, Math.ceil(Math.max(...tops) * 10 + 0.5) / 10);
  strata.forEach((st) => {
    const d = st.d, dl = d.delta_recall_at_10_vs_prior;
    st.head.replaceChildren(
      h("h4", { class: "mini-title", style: "font-size:1.02rem", text: st.name }),
      h("p", { class: "sub", text: `${fmt.pct0(d.share_of_test_rows)} of members, ${fmt.pct0(d.share_of_high_cost)} of high-cost members. ${fmt.pct(d.prevalence)} of this group ends up high cost.` }));
    onResize(st.el, () => intervalRows(st.el, [
      { label: "The model", strong: true, I: get(d, "lgbm.recall_at_10"), cls: "model" },
      { label: "Last year's cost", I: get(d, "prior_cost.recall_at_10"), cls: "base" },
    ], { max, format: fmt.pct, tick: fmt.pct0 }));
    st.el.after(h("p", { class: "fig-note", html: dl ? `Difference ${numSpan(signedPts(dl.v))} points (95% interval ${numSpan(range(dl))}).` : "" }));
  });
  const [a, b] = strata;
  if (a && b) {
    const da = a.d.delta_recall_at_10_vs_prior, db = b.d.delta_recall_at_10_vs_prior;
    const both = da.lo > 0 && db.lo > 0;
    setHTML("stratum-finding",
      (both ? "The model beats the rule inside both groups, so the result is not only a matter of sorting one group above the other. "
        : "The model does not clearly beat the rule inside both groups. ") +
      `Among women aged 18 to 49 it reaches ${withCI(get(a.d, "lgbm.recall_at_10"))} against ${withCI(get(a.d, "prior_cost.recall_at_10"))}. ` +
      `Among everyone else it reaches ${withCI(get(b.d, "lgbm.recall_at_10"))} against ${withCI(get(b.d, "prior_cost.recall_at_10"))}.` +
      restSentence(gd));

    // ---- the position, with its cost ----
    if (maternity) {
      setHTML("gain-position",
        `A care team would not use a list of women likely to become pregnant as an outreach list. In my view maternity care belongs in its own program, with its own ways of finding members, and a score that mostly says "woman of childbearing age" gives a nurse nothing to act on. ` +
        `So in production I would send maternity to its own program and judge this model on everyone else, where it reaches ${numSpan(fmt.pct(val(get(b.d, "lgbm.recall_at_10"))))} against ${numSpan(fmt.pct(val(get(b.d, "prior_cost.recall_at_10"))))} for the rule, a gain of ${numSpan(fmt.pts(db.v))} points (95% interval ${numSpan(fmt.pts(db.lo))} to ${numSpan(fmt.pts(db.hi))}).`);
      setHTML("gain-cost",
        `That choice has a cost. It sets aside ${numSpan(fmt.pct0(a.d.share_of_test_rows))} of members and ${numSpan(fmt.pct0(a.d.share_of_high_cost))} of the high-cost ones, which is the part of the population where the model's lead over the rule is largest (${numSpan(fmt.pts(da.v))} points). ` +
        `The headline at the top of the page stays the whole-test-set figure, because that was the measure chosen before the results were in. The figure I would carry into a planning conversation is the one for everyone else.`);
    }
  }

  // ---- variations ----
  const m = get(hd, "recall_at_k.lgbm"), p = get(hd, "recall_at_k.prior_cost"), d = get(hd, "delta_recall_at_k_vs_prior");
  const cell = (I) => (val(I) === null ? "" : fmt.pct(val(I)));
  const diff = (I) => (I && isNum(I.v) ? `${signedPts(I.v)}${isNum(I.lo) ? ` (${range(I)})` : ""}` : "");
  const rows = [["Whole test set, as in the headline", cell(m), cell(p), diff(d)]];
  const ex = sens.excluding_pregnant;
  if (ex) rows.push([`Leaving out the ${fmt.int(ex.n_excluded)} member-years pregnant on January 1`, cell(get(ex, "lgbm.recall_at_10")), cell(get(ex, "prior_cost.recall_at_10")), diff(ex.delta_recall_at_10_vs_prior)]);
  const wo = sens.without_pregnancy_feature;
  if (wo) rows.push(["Model retrained without the pregnancy flag", cell(get(wo, "lgbm.recall_at_10")), cell(p), diff(wo.delta_recall_at_10_vs_prior)]);
  const pr = sens.prior_cost_plus_pregnancy_rule;
  if (pr) rows.push(["Rule improved: pregnant members first, then by last year's cost", cell(m), cell(get(pr, "prior_cost_plus_pregnancy.recall_at_10")), diff(pr.lgbm_minus_rule_recall_at_10)]);
  const np = get(perf, "sanity.no_prior_cost_features");
  if (np && isNum(val(p))) rows.push(["Model retrained without any prior-cost feature", fmt.pct(np.recall_at_10), cell(p), `${signedPts(np.recall_at_10 - val(p))} (no interval computed)`]);
  $("table-variations").replaceChildren(dataTable(
    [{ label: "Variation" }, { label: "Model", num: true }, { label: "Last year's cost", num: true }, { label: "Difference in points (95% interval)", num: true }], rows));
  const lows = [ex && ex.delta_recall_at_10_vs_prior, wo && wo.delta_recall_at_10_vs_prior, pr && pr.lgbm_minus_rule_recall_at_10].filter(Boolean);
  setHTML("variations-note", lows.length && lows.every((I) => I.lo > 0)
    ? "The gain holds in every variation. Each row is the share of high-cost members reached at the same capacity."
    : "The gain does not hold in every variation. Each row is the share of high-cost members reached at the same capacity.");
}
