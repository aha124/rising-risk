// Loads the JSON the pipeline wrote and hands it to each section.
// Every number on the page comes from these files. A section whose file is
// missing shows a short message and the rest of the page still works.

import { $, fmt, get, isNum, initTips, fail } from "./util.js";
import { initField } from "./field.js";
import { heroCopy, initResults } from "./results.js";
import { initLineage } from "./lineage.js";
import { initPlanner } from "./planner.js";
import { initExplain } from "./explain.js";
import { initTryIt } from "./tryit.js";
import { initAudit } from "./audit.js";
import { initMonitor } from "./monitor.js";
import { initGain } from "./gain.js";

const FILES = ["summary", "performance", "explain", "fairness", "drift", "pipeline"];

async function load(name) {
  try {
    const res = await fetch(`data/${name}.json`);
    if (!res.ok) throw new Error(`${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`data/${name}.json could not be loaded`, err);
    return null;
  }
}

function showFixtureBanner() {
  $("fixture-banner").hidden = false;
  document.documentElement.dataset.fixture = "true";
}

function section(name, target, fn) {
  try {
    fn();
  } catch (err) {
    console.warn(`${name} could not be drawn`, err);
    if (target) fail(target, name);
  }
}

function docket(summary) {
  const d = get(summary, "data", {});
  if (isNum(d.n_patients)) $("dk-data").textContent = `${fmt.int(d.n_patients)} synthetic patients (Synthea)`;
  $("dk-version").textContent = get(summary, "model_version", "n/a");
  $("dk-date").textContent = String(get(summary, "generated_at", "n/a")).slice(0, 10);
  $("dk-sha").textContent = String(get(summary, "git_sha", "n/a")).slice(0, 7);
  const k = get(summary, "headline.k", 0.1);
  const slots = {
    k_pct: fmt.pct0(k),
    n_called_test: isNum(d.n_test) ? fmt.int(Math.round(d.n_test * k)) : "n/a",
  };
  document.querySelectorAll("[data-s]").forEach((el) => {
    el.textContent = slots[el.dataset.s] ?? "";
    el.classList.add("num");
  });
}

// The limit about the gain is written from the same numbers as the section it points to.
function limitsCopy(summary, perf) {
  const sens = get(perf, "sensitivity", {});
  const share = sens.gain_share_women_18_49_not_pregnant;
  const rest = get(sens, "by_childbearing_stratum.everyone_else");
  const li = $("limit-gain");
  if (!li) return;
  if (!isNum(share) || share < 0.5 || !rest) { li.hidden = true; return; }
  $("limit-gain-text").innerHTML =
    `<span class="num">${fmt.pct0(share)}</span> of the members the model finds and the rule misses are women aged 18 to 49 who were not pregnant on January 1, and I would handle likely pregnancies through a maternity program, not this list. ` +
    `The figure to carry forward is the one for everyone else: <span class="num">${fmt.pct(get(rest, "lgbm.recall_at_10.v"))}</span> against <span class="num">${fmt.pct(get(rest, "prior_cost.recall_at_10.v"))}</span>. <a href="#gain">Details above.</a>`;
}

async function main() {
  initTips();
  const [summary, perf, explain, fair, drift, pipeline] = await Promise.all(FILES.map(load));
  if ([summary, perf, explain, fair, drift, pipeline].some((f) => f && f.fixture === true)) showFixtureBanner();

  // feature name -> display label, for anything that arrives as a raw column name
  const labels = new Map(get(explain, "features", []).map((f) => [f.name, f.label || f.name]));
  const labelOf = (name) => labels.get(name) || String(name).replace(/_/g, " ");

  if (summary) {
    section("The run details", null, () => docket(summary));
    section("The headline result", "lede", () => heroCopy(summary, perf));
    section("The member field", "field-stage", () => initField(summary));
    section("The comparison", "chart-compare", () => initResults(summary, perf, labelOf));
  } else {
    fail("lede", "The summary data");
  }
  if (pipeline) section("The lineage diagram", "lineage", () => initLineage(pipeline, summary));
  else fail("lineage", "The pipeline data");
  if (perf && summary) section("The breakdown of the gain", "chart-units", () => { initGain(summary, perf); limitsCopy(summary, perf); });
  else fail("chart-units", "The breakdown of the gain");
  if (perf) section("The planner", "planner-bars", () => initPlanner(summary, perf));
  else fail("planner-bars", "The capacity data");
  const loadMember = initTryIt(showFixtureBanner);
  if (explain) section("The member explanations", "chart-waterfall", () => initExplain(explain, summary, loadMember));
  else fail("chart-waterfall", "The explanation data");
  if (fair) section("The subgroup audit", "chart-fairness", () => initAudit(fair, summary, explain));
  else fail("chart-fairness", "The audit data");
  if (drift) section("The drift monitor", "psi-panels", () => initMonitor(drift, perf, summary, labelOf));
  else fail("psi-panels", "The drift data");
  document.documentElement.dataset.ready = "true";
}

main();
