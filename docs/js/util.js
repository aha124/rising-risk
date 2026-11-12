// Small shared helpers: DOM builders, number formats, tooltip, resize.

export const $ = (id) => document.getElementById(id);

function build(ns, tag, attrs, kids) {
  const el = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "html") el.innerHTML = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
export const h = (tag, attrs, ...kids) => build(null, tag, attrs, kids);
export const s = (tag, attrs, ...kids) => build("http://www.w3.org/2000/svg", tag, attrs, kids);

export const isNum = (x) => typeof x === "number" && Number.isFinite(x);

// Read a nested value without throwing: get(obj, "a.b.c", fallback)
export function get(obj, path, fallback = undefined) {
  let cur = obj;
  for (const key of path.split(".")) {
    if (cur === null || cur === undefined) return fallback;
    cur = cur[key];
  }
  return cur === undefined || cur === null ? fallback : cur;
}

// An interval can arrive as {v, lo, hi} or as a bare number.
export const val = (I) => (isNum(I) ? I : isNum(I && I.v) ? I.v : null);

const intFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
export const fmt = {
  int: (x) => (isNum(x) ? intFmt.format(Math.round(x)) : "n/a"),
  pct: (x, d = 1) => (isNum(x) ? (x * 100).toFixed(d) + "%" : "n/a"),
  // a probability, never shown as a flat 0% or 100%
  prob: (x) => (!isNum(x) ? "n/a" : x > 0.999 ? "over 99.9%" : x < 0.001 ? "under 0.1%" : (x * 100).toFixed(1) + "%"),
  pct0: (x) => (isNum(x) ? Math.round(x * 100) + "%" : "n/a"),
  pts: (x, d = 1) => (isNum(x) ? (Math.abs(x) * 100).toFixed(d) : "n/a"),
  num: (x, d = 2) => (isNum(x) ? x.toFixed(d) : "n/a"),
  signed: (x, d = 2) => (isNum(x) ? (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d) : "n/a"),
  usd: (x) => (isNum(x) ? (x < 0 ? "−$" : "$") + intFmt.format(Math.abs(Math.round(x))) : "n/a"),
  compact(x) {
    if (!isNum(x)) return "n/a";
    const a = Math.abs(x);
    if (a >= 1e6) return (x / 1e6).toFixed(a >= 1e7 ? 1 : 2) + "M";
    if (a >= 1e4) return Math.round(x / 1e3) + "K";
    return intFmt.format(x);
  },
};

export function ordinal(n) {
  const v = Math.round(n);
  const tens = v % 100;
  const suffix = tens >= 11 && tens <= 13 ? "th" : ["th", "st", "nd", "rd"][v % 10] || "th";
  return v + suffix;
}

// "Outpatient visits" -> "outpatient visits", but "eGFR" stays as it is
export const lowerFirst = (str) => (/^[A-Z][a-z]/.test(String(str)) ? String(str).charAt(0).toLowerCase() + String(str).slice(1) : String(str));
export const cap = (str) => String(str).charAt(0).toUpperCase() + String(str).slice(1);
const TIER_NAMES = { high: "High", rising: "Rising", routine: "Routine" };
export const tierName = (t) => TIER_NAMES[String(t).toLowerCase()] || cap(t);
// Display names for the raw race and ethnicity codes in the data
const LEVEL_NAMES = {
  white: "White", black: "Black", asian: "Asian", native: "Native American", hawaiian: "Native Hawaiian",
  other: "Other", hispanic: "Hispanic", nonhispanic: "Not Hispanic",
};
export const levelName = (x) => LEVEL_NAMES[String(x).toLowerCase()] || cap(x);

export const esc = (str) =>
  String(str).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// A number with its 95% interval on hover, focus or tap.
export function withCI(I, f = fmt.pct) {
  const v = val(I);
  if (v === null) return '<span class="num">n/a</span>';
  if (!I || !isNum(I.lo) || !isNum(I.hi)) return `<span class="num">${f(v)}</span>`;
  const range = `95% interval ${f(I.lo)} to ${f(I.hi)}`;
  return `<button type="button" class="num" data-tip="${esc(range)}" aria-label="${esc(f(v) + ", " + range)}">${f(v)}</button>`;
}
export const numSpan = (text) => `<span class="num">${esc(text)}</span>`;

// ---- tooltip: one element, positioned in the viewport ----
let tipEl = null;
export function showTip(html, x, y) {
  tipEl = tipEl || $("tip");
  if (!tipEl) return;
  tipEl.innerHTML = html;
  tipEl.hidden = false;
  const r = tipEl.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  let left = x + 14;
  if (left + r.width > vw - 8) left = x - r.width - 14;
  left = Math.max(8, Math.min(left, vw - r.width - 8));
  let top = y - r.height - 12;
  if (top < 8) top = y + 18;
  tipEl.style.left = left + "px";
  tipEl.style.top = top + "px";
}
export function hideTip() {
  if (tipEl) tipEl.hidden = true;
}

export function initTips() {
  const show = (el) => {
    const r = el.getBoundingClientRect();
    showTip(esc(el.dataset.tip), r.left + r.width / 2 - 14, r.top);
  };
  const target = (e) => (e.target instanceof Element ? e.target.closest("[data-tip]") : null);
  document.addEventListener("pointerover", (e) => { const el = target(e); if (el) show(el); });
  document.addEventListener("pointerout", (e) => { if (target(e)) hideTip(); });
  document.addEventListener("focusin", (e) => { const el = target(e); if (el) show(el); });
  document.addEventListener("focusout", (e) => { if (target(e)) hideTip(); });
  document.addEventListener("click", (e) => { const el = target(e); if (el) show(el); else hideTip(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });
  window.addEventListener("scroll", hideTip, { passive: true });
}

// Call fn now and again whenever the element's width changes.
export function onResize(el, fn) {
  let last = -1;
  let raf = 0;
  const run = () => {
    raf = 0;
    const w = Math.round(el.clientWidth);
    if (w === last || w === 0) return;
    last = w;
    fn(w);
  };
  new ResizeObserver(() => { if (!raf) raf = requestAnimationFrame(run); }).observe(el);
  run();
  return () => { last = -1; run(); };
}

export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// headers: [{label, num}], rows: arrays of cell text (first cell is a row header)
export function dataTable(headers, rows, caption) {
  return h(
    "table",
    { class: "data" },
    caption ? h("caption", { text: caption }) : null,
    h("thead", {}, h("tr", {}, headers.map((c) => h("th", { scope: "col", class: c.num ? "n" : null, text: c.label })))),
    h(
      "tbody",
      {},
      rows.map((r) =>
        h("tr", {}, r.map((cell, i) => {
          const isNode = cell && cell.nodeType;
          const attrs = { class: headers[i] && headers[i].num ? "n" : null };
          if (i === 0) return h("th", { scope: "row", ...attrs }, isNode ? cell : String(cell));
          return h("td", attrs, isNode ? cell : String(cell));
        }))
      )
    )
  );
}

export function setHTML(id, html) {
  const el = $(id);
  if (el) el.innerHTML = html;
}
export function fail(id, what) {
  const el = $(id);
  if (el) el.replaceChildren(h("p", { class: "error", text: `${what} is not available.` }));
}
