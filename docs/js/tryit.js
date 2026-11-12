// "Try the model": score a member with the real trees, in the browser.
// model_browser.json is fetched only when this part of the page gets close,
// or when a member from the explorer above is loaded into it.

import { $, h, fmt, isNum, numSpan, ordinal, tierName, esc, lowerFirst } from "./util.js";
import { score, percentileOf } from "./scorer.js";

function display(ft, v) {
  if (v === null || v === undefined) return "not measured";
  if (ft.type === "binary") return v === 1 ? "yes" : "no";
  if (ft.unit === "USD") return fmt.usd(v);
  const d = isNum(ft.step) && ft.step < 1 ? (ft.step < 0.1 ? 2 : 1) : 0;
  const unit = ft.unit && !["count", "share", "score", "0/1"].includes(ft.unit) ? " " + ft.unit : "";
  return v.toFixed(d) + unit;
}

function build(model) {
  const feats = Array.isArray(model.features) ? model.features : [];
  const index = new Map(feats.map((ft, j) => [ft.name, j]));
  const defaults = feats.map((ft) => (isNum(ft.default) ? ft.default : null));
  let row = defaults.slice();

  // Inputs that are computed from other inputs (a difference, a sum) follow them.
  const derived = (Array.isArray(model.derived) ? model.derived : [])
    .map((d) => ({ j: index.get(d.feature), op: d.op, of: (d.of || []).map((n) => index.get(n)) }))
    .filter((d) => d.j !== undefined && d.of.every((o) => o !== undefined));
  function applyDerived(r) {
    derived.forEach((d) => {
      const vals = d.of.map((o) => r[o]);
      if (vals.some((v) => v === null || v === undefined)) return;
      if (d.op === "difference" && vals.length === 2) r[d.j] = vals[0] - vals[1];
      else if (d.op === "sum") r[d.j] = vals.reduce((a, b) => a + b, 0);
    });
  }
  applyDerived(defaults);
  row = defaults.slice();
  const typical = score(model, defaults);

  const big = h("p", { class: "big" });
  const line = h("p", { class: "line" });
  const marker = h("span", { class: "mk" });
  const pr = percentileOf(model, model.tiers ? model.tiers.rising : NaN);
  const ph = percentileOf(model, model.tiers ? model.tiers.high : NaN);
  const haveTiers = isNum(pr) && isNum(ph);
  const widths = haveTiers ? [pr, ph - pr, 100 - ph] : [100];
  const classes = haveTiers ? ["t-routine", "t-rising", "t-high"] : ["t-routine"];
  const out = h("div", { class: "out", "aria-live": "polite" },
    h("p", { class: "cap", text: "Probability of landing in next year's high-cost group" }),
    big,
    h("div", { class: "scale", "aria-hidden": "true" },
      h("span", { class: "trk" }, widths.map((w, i) => h("i", { class: classes[i], style: `flex:${Math.max(w, 0.5)} 0 0` }))),
      marker),
    haveTiers ? h("div", { class: "scale-labels", "aria-hidden": "true" },
      [["Routine", `below the ${ordinal(pr)} percentile`], ["Rising", `to the ${ordinal(ph)}`], ["High", "above that"]]
        .map((t, i) => h("span", {}, h("i", { class: classes[i] }), `${t[0]}, ${t[1]}`))) : null,
    line);

  let loadedName = null;
  // A loaded member's record is scored exactly as recorded. Derived inputs are recomputed only after a change made here.
  function update(recompute = true) {
    if (recompute) applyDerived(row);
    const r = score(model, row);
    big.textContent = fmt.prob(r.p);
    marker.style.left = (isNum(r.percentile) ? r.percentile : 0) + "%";
    const same = row.every((v, j) => v === defaults[j] || (v === null && defaults[j] === null));
    const versus = same ? "This is the typical member." : `The typical member scores ${numSpan(fmt.prob(typical.p))}.`;
    line.innerHTML = (isNum(r.percentile) ? `Higher than ${numSpan(fmt.num(r.percentile, 0) + "%")} of members in the test years. ` : "") +
      `Tier: <strong>${esc(tierName(r.tier))}</strong>. ${versus}`;
    return r;
  }

  // ---- controls ----
  const syncers = [];
  const grid = h("div", { class: "ctl-grid" });
  feats.forEach((ft, j) => {
    if (!ft.interactive) return;
    if (ft.type === "binary") {
      const input = h("input", { type: "checkbox" });
      input.addEventListener("change", () => { row[j] = input.checked ? 1 : 0; loadedName = null; state(); update(); });
      syncers.push(() => { input.checked = row[j] === 1; });
      grid.append(h("div", { class: "ctl check" }, h("label", {}, input, ft.label || ft.name)));
      return;
    }
    if (!isNum(ft.min) || !isNum(ft.max) || ft.max <= ft.min) return;
    const id = `ctl-${j}`;
    const outv = h("output", { for: id });
    const input = h("input", { type: "range", id, min: ft.min, max: ft.max, step: isNum(ft.step) && ft.step > 0 ? ft.step : "any" });
    const fallback = isNum(defaults[j]) ? defaults[j] : ft.min;
    let na = null;
    const show = () => {
      outv.textContent = display(ft, row[j]);
      input.setAttribute("aria-valuetext", display(ft, row[j]));
    };
    input.addEventListener("input", () => {
      row[j] = Number(input.value);
      if (na) na.checked = false;
      loadedName = null;
      show(); state(); update();
    });
    // from the outside (reset, a loaded member): the slider shows the nearest value it can, the text shows the true one
    syncers.push(() => {
      input.value = row[j] === null ? fallback : Math.min(ft.max, Math.max(ft.min, row[j]));
      if (na) na.checked = row[j] === null;
      show();
    });
    const box = h("div", { class: "ctl" }, h("div", { class: "top" }, h("label", { for: id, text: ft.label || ft.name }), outv), input);
    if (ft.nullable) {
      na = h("input", { type: "checkbox" });
      na.addEventListener("change", () => { row[j] = na.checked ? null : Number(input.value); loadedName = null; show(); state(); update(); });
      box.append(h("label", { class: "na" }, na, "Not measured"));
    }
    grid.append(box);
  });

  const stateEl = $("tryit-state");
  function state() {
    stateEl.innerHTML = loadedName ? `Showing ${esc(loadedName)}, with all ${feats.length} inputs from that member's record. Move any control to change it.` : "";
  }
  function setRow(next, name) {
    row = feats.map((_, j) => (next[j] === undefined ? null : next[j]));
    loadedName = name || null;
    syncers.forEach((fn) => fn());
    state();
    return update(false);
  }

  const reset = $("tryit-reset");
  reset.hidden = false;
  reset.addEventListener("click", () => setRow(defaults, null));
  $("tryit").replaceChildren(h("div", {}, grid), out);

  const shown = feats.filter((ft) => ft.interactive);
  $("tryit-sub").textContent = `Change a member's record and the ${fmt.int(model.trees.length)} trained trees score it in your browser. ` +
    `${shown.length} of the ${feats.length} inputs have a control. The rest stay at their typical value` +
    (derived.length ? `, except ${derived.length === 1 ? "one that is" : `${derived.length} that are`} computed from other inputs and ${derived.length === 1 ? "follows" : "follow"} them.` : ".");
  $("tryit-defaults").innerHTML = `The typical member is the training median of every input, so it starts as: ` +
    shown.map((ft) => `${esc(lowerFirst(ft.label || ft.name))} ${numSpan(display(ft, defaults[index.get(ft.name)]))}`).join(", ") +
    `. That member scores ${numSpan(fmt.prob(typical.p))}.` +
    (model.interactive_note ? ` ${esc(String(model.interactive_note).replace(/^Inputs without a control stay at their training median\.\s*/, "").replace("a visitor", "you"))}` : "");
  setRow(defaults, null);
  return { setRow };
}

export function initTryIt(onFixture) {
  const fig = $("tryit-fig");
  let pending = null;
  const load = () => {
    if (pending) return pending;
    $("tryit-loading").textContent = "Loading the model.";
    pending = (async () => {
      try {
        const res = await fetch("data/model_browser.json");
        if (!res.ok) throw new Error(res.status);
        const model = await res.json();
        if (!Array.isArray(model.trees) || !model.trees.length) throw new Error("no trees");
        if (model.fixture) onFixture();
        return build(model);
      } catch (err) {
        console.warn("model_browser.json could not be loaded", err);
        $("tryit").replaceChildren(h("p", { class: "error", text: "The in-browser model could not be loaded. The members above still show how the model reasons." }));
        return null;
      }
    })();
    return pending;
  };
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { io.disconnect(); load(); }
  }, { rootMargin: "900px 0px" });
  io.observe(fig);

  // Load one member's full record into the scorer. Returns the score, or null.
  return async function loadMember(row, name) {
    const api = await load();
    if (!api || !Array.isArray(row)) return null;
    const r = api.setRow(row, name);
    fig.scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    return r;
  };
}
