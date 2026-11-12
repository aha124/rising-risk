// One line-chart function used by most figures, plus the shared frame and axes.
// Colors come from CSS classes, so a theme change needs no redraw.

import { scaleLinear, scaleLog, line, curveStepAfter, curveLinear, ticks as niceTicks } from "../vendor/d3-subset.min.js";
import { s, h, showTip, hideTip, isNum, esc } from "./util.js";

export function frame(el, { height = 280, m = {} } = {}) {
  const W = Math.max(200, Math.round(el.clientWidth));
  const margin = { top: 10, right: 12, bottom: 34, left: 44, ...m };
  const w = W - margin.left - margin.right;
  const ht = height - margin.top - margin.bottom;
  const svg = s("svg", { viewBox: `0 0 ${W} ${height}`, height, "aria-hidden": "true", focusable: "false" });
  const g = s("g", { transform: `translate(${margin.left},${margin.top})` });
  svg.append(g);
  el.replaceChildren(svg);
  return { svg, g, w, h: ht, W, margin };
}

export function xAxis(f, x, { values, format, label, grid = false } = {}) {
  const tv = values || x.ticks(Math.max(2, Math.floor(f.w / 80)));
  f.g.append(s("line", { class: "axis", x1: 0, x2: f.w, y1: f.h, y2: f.h }));
  tv.forEach((t) => {
    const px = x(t);
    if (grid) f.g.append(s("line", { class: "grid", x1: px, x2: px, y1: 0, y2: f.h }));
    f.g.append(s("line", { class: "axis", x1: px, x2: px, y1: f.h, y2: f.h + 4 }));
    f.g.append(s("text", { x: px, y: f.h + 17, "text-anchor": "middle", text: format ? format(t) : t }));
  });
  if (label) f.g.append(s("text", { class: "lab muted", x: f.w, y: f.h + 31, "text-anchor": "end", text: label }));
}

export function yAxis(f, y, { values, format, label, count } = {}) {
  const tv = values || y.ticks(count || Math.max(2, Math.floor(f.h / 48)));
  tv.forEach((t) => {
    const py = y(t);
    f.g.append(s("line", { class: "grid", x1: 0, x2: f.w, y1: py, y2: py }));
    f.g.append(s("text", { x: -8, y: py + 4, "text-anchor": "end", text: format ? format(t) : t }));
  });
  if (label) f.g.append(s("text", { class: "lab muted", x: -f.margin.left, y: -f.margin.top + 11, text: label }));
}

/*
 opts = {
   height, margin,
   x: { domain, format, label, values },   y: { domain, format, label, values, count },
   series: [{ name, cls: "model"|"base"|"ink"|"signal"|"dot", thin, dash, step, pts: [[x, y], ...],
              dots: true | (pt, i) => "fill" | "hollow" | null }],
   refsY: [{ y, label }], refsX: [{ x, label }], diagonal: bool,
   tip: { x: (xv) => string }        // header text for the tooltip
 }
*/
export function lineChart(el, opts) {
  const f = frame(el, { height: opts.height || 280, m: opts.margin });
  const xd = opts.x.domain;
  const yd = opts.y.domain;
  const mk = opts.log ? scaleLog : scaleLinear;
  const x = mk().domain(xd).range([0, f.w]).clamp(!!opts.log);
  const y = mk().domain(yd).range([f.h, 0]).clamp(!!opts.log);
  if (!opts.log && opts.y.nice !== false && !opts.y.values) y.nice(opts.y.count || Math.max(2, Math.floor(f.h / 48)));

  yAxis(f, y, opts.y);
  xAxis(f, x, opts.x);

  (opts.refsX || []).forEach((r) => {
    if (!isNum(r.x)) return;
    f.g.append(s("line", { class: "ref", x1: x(r.x), x2: x(r.x), y1: 0, y2: f.h }));
    if (r.label) {
      const right = x(r.x) > f.w - 90;
      f.g.append(s("text", { class: "lab", x: x(r.x) + (right ? -6 : 6), y: 11, "text-anchor": right ? "end" : "start", text: r.label }));
    }
  });
  (opts.refsY || []).forEach((r) => {
    if (!isNum(r.y)) return;
    f.g.append(s("line", { class: "ref", x1: 0, x2: f.w, y1: y(r.y), y2: y(r.y) }));
    if (r.label) f.g.append(s("text", { class: "lab", x: f.w, y: y(r.y) - 5, "text-anchor": "end", text: r.label }));
  });
  if (opts.diagonal) {
    const a = Math.max(xd[0], y.domain()[0]);
    const b = Math.min(xd[1], y.domain()[1]);
    f.g.append(s("line", { class: "ln thin s-ink", x1: x(a), y1: y(a), x2: x(b), y2: y(b) }));
  }

  const series = (opts.series || []).filter((sr) => sr && Array.isArray(sr.pts) && sr.pts.length);
  [...series].sort((a, b) => (a.z || 0) - (b.z || 0)).forEach((sr) => {
    const pts = sr.pts.filter((p) => isNum(p[0]) && isNum(p[1]));
    const path = line().x((p) => x(p[0])).y((p) => y(p[1])).curve(sr.step ? curveStepAfter : curveLinear);
    f.g.append(s("path", {
      class: `ln s-${sr.cls || "ink"}${sr.thin ? " thin" : ""}`,
      d: path(pts),
      "stroke-dasharray": sr.dash ? "4 4" : null,
    }));
    if (sr.dots) {
      pts.forEach((p, i) => {
        const kind = sr.dots === true ? "fill" : sr.dots(p, i);
        if (!kind) return;
        f.g.append(s("circle", {
          cx: x(p[0]), cy: y(p[1]), r: kind === "hollow" ? 3.5 : 4.5,
          class: kind === "hollow" ? `f-paper s-${sr.cls}` : `f-${sr.cls} ring`,
          "stroke-width": 2,
        }));
      });
    }
  });

  // hover: nearest x, one row per series
  if (opts.tip !== false && series.length) {
    const cross = s("line", { class: "ref", y1: 0, y2: f.h, visibility: "hidden" });
    f.g.append(cross);
    const hit = s("rect", { class: "hit", x: 0, y: 0, width: f.w, height: f.h });
    const yf = opts.y.format || ((v) => v);
    const nearest = (pts, xv) => pts.reduce((best, p) => (Math.abs(p[0] - xv) < Math.abs(best[0] - xv) ? p : best), pts[0]);
    const move = (e) => {
      const box = hit.getBoundingClientRect();
      const xv = x.invert(((e.clientX - box.left) / box.width) * f.w);
      const anchor = nearest(series[0].pts, xv);
      cross.setAttribute("x1", x(anchor[0]));
      cross.setAttribute("x2", x(anchor[0]));
      cross.setAttribute("visibility", "visible");
      const head = opts.tip && opts.tip.x ? opts.tip.x(anchor[0]) : (opts.x.format ? opts.x.format(anchor[0]) : anchor[0]);
      const rows = series.filter((sr) => sr.name).map((sr) => {
        const p = nearest(sr.pts, anchor[0]);
        const key = `<span class="key ${sr.dotsOnly ? "dot" : "line"}${sr.thin ? " thin" : ""}${sr.dash ? " dash" : ""} c-${sr.cls || "ink"}"></span>`;
        return `<tr><td>${key}${esc(sr.name)}</td><td>${esc(yf(p[1]))}</td></tr>`;
      }).join("");
      showTip(`<b>${esc(head)}</b><table>${rows}</table>`, e.clientX, e.clientY);
    };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
    f.g.append(hit);
  }
  return { f, x, y };
}

/*
 Dot and 95% interval, one row per item, label on its own line above the mark.
 rows: [{ label, strong, I: {v, lo, hi} | number, cls: "model"|"base"|"ink"|"signal", hollow, tip }]
*/
export function intervalRows(el, rows, { max, format = (v) => v, tick = format, rowH = 50 } = {}) {
  rows = rows.filter((r) => isNum(r.I) || (r.I && isNum(r.I.v)));
  if (!rows.length) return;
  const v = (I) => (isNum(I) ? I : I.v);
  const top = (I) => (isNum(I) ? I : isNum(I.hi) ? I.hi : I.v);
  const hi = isNum(max) ? max : Math.max(...rows.map((r) => top(r.I))) * 1.08;
  const f = frame(el, { height: rows.length * rowH + 34, m: { top: 4, left: 2, right: 64, bottom: 30 } });
  const x = scaleLinear().domain([0, hi]).range([0, f.w]);
  if (!isNum(max)) x.nice(4);
  x.ticks(Math.max(3, Math.floor(f.w / 90))).forEach((tk) => {
    f.g.append(s("line", { class: "grid", x1: x(tk), x2: x(tk), y1: 0, y2: f.h }));
    f.g.append(s("text", { x: x(tk), y: f.h + 17, "text-anchor": "middle", text: tick(tk) }));
  });
  rows.forEach((r, i) => {
    const y0 = i * rowH;
    const cy = y0 + 32;
    const cls = r.cls || "ink";
    f.g.append(s("text", { class: `lab${r.strong ? " strong" : ""}`, x: 0, y: y0 + 16, text: r.label }));
    f.g.append(s("line", { class: "grid", x1: 0, x2: f.w, y1: cy, y2: cy }));
    const has = !isNum(r.I) && isNum(r.I.lo) && isNum(r.I.hi);
    if (has) {
      f.g.append(s("line", { class: `ln s-${cls}`, x1: x(r.I.lo), x2: x(r.I.hi), y1: cy, y2: cy, "stroke-width": 2 }));
      [r.I.lo, r.I.hi].forEach((e) => f.g.append(s("line", { class: `ln s-${cls}`, x1: x(e), x2: x(e), y1: cy - 6, y2: cy + 6, "stroke-width": 2 })));
    }
    f.g.append(s("circle", { cx: x(v(r.I)), cy, r: 5, class: r.hollow ? `f-paper s-${cls}` : `f-${cls} ring`, "stroke-width": 2 }));
    f.g.append(s("text", { class: "val", x: x(top(r.I)) + 12, y: cy + 4, text: format(v(r.I)) }));
    const hit = s("rect", { class: "hit", x: 0, y: y0, width: f.w + 60, height: rowH, style: "cursor:default" });
    const tip = `<b>${esc(r.label)}</b><br>${esc(format(v(r.I)))}` + (has ? `<br>95% interval ${esc(format(r.I.lo))} to ${esc(format(r.I.hi))}` : r.tip ? `<br>${esc(r.tip)}` : "");
    hit.addEventListener("pointermove", (e) => showTip(tip, e.clientX, e.clientY));
    hit.addEventListener("pointerleave", hideTip);
    f.g.append(hit);
  });
}

export { niceTicks, h };
