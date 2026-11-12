// The member field: every dot is one member from the test years. Dots that ended
// up high cost carry the signal color. The first block is the call list (the top
// share the team can reach). Changing the ordering moves each dot to its new place.

import { easeCubicInOut } from "../vendor/d3-subset.min.js";
import { $, h, fmt, get, isNum, showTip, hideTip, reducedMotion, dataTable, numSpan } from "./util.js";

const ORDERS = [
  { key: "random", field: "rank_random", name: "Random order" },
  { key: "prior", field: "rank_prior", name: "Last year's cost" },
  { key: "model", field: "rank_model", name: "The model" },
];
const DURATION = 950;
const GRAY_DELAY = 260;

export function initField(summary) {
  const raw = get(summary, "members_field", []);
  const members = raw.filter((m) => m && ORDERS.every((o) => isNum(m[o.field])));
  const N = members.length;
  if (N < 10) throw new Error("members_field is missing or too small");
  const k = get(summary, "headline.k", 0.1);
  const B = Math.max(1, Math.round(N * k));

  // position of each member under each ordering (re-ranked, so ties or gaps cannot break the layout)
  const pos = {};
  const at = {};
  for (const o of ORDERS) {
    const idx = members.map((_, i) => i).sort((a, b) => members[a][o.field] - members[b][o.field] || a - b);
    pos[o.key] = new Int32Array(N);
    idx.forEach((member, p) => { pos[o.key][member] = p; });
    at[o.key] = idx;
  }
  const isPos = members.map((m) => m.y === 1);
  const totalPos = isPos.filter(Boolean).length;
  const found = {};
  for (const o of ORDERS) found[o.key] = at[o.key].slice(0, B).filter((i) => isPos[i]).length;
  const onPriorList = (i) => pos.prior[i] < B;
  const newFinds = at.model.slice(0, B).filter((i) => isPos[i] && !onPriorList(i)).length;

  // ---- controls ----
  const controls = $("field-controls");
  ORDERS.forEach((o) => {
    controls.append(
      h("label", {},
        h("input", { type: "radio", name: "order", value: o.key }),
        h("span", { class: "opt-name", text: o.name }),
        h("span", { class: "opt-count", text: `${fmt.int(found[o.key])} of ${fmt.int(totalPos)} reached` }),
        h("span", { class: "opt-bar", "aria-hidden": "true" }, h("i", { style: `width:${(100 * found[o.key]) / Math.max(1, totalPos)}%` }))
      )
    );
  });

  $("field-table").replaceChildren(
    dataTable(
      [{ label: "Ordering" }, { label: "On the call list", num: true }, { label: "High cost, on the list", num: true },
        { label: "High cost, not called", num: true }, { label: "Share reached", num: true }],
      ORDERS.map((o) => [o.name, fmt.int(B), fmt.int(found[o.key]), fmt.int(totalPos - found[o.key]), fmt.pct(found[o.key] / Math.max(1, totalPos))]),
      `A random ${fmt.int(N)} of the test member-years, ${fmt.int(totalPos)} of whom ended up high cost.`
    )
  );

  // The picture is a sample, so its counts are not the full-test figures in the headline. Say so.
  const note = get(summary, "members_field_note", {});
  if (isNum(note.in_band_model) && (note.in_band_model !== found.model || note.in_band_prior !== found.prior)) {
    console.warn("members_field_note does not match the counts computed from members_field");
  }
  const nTest = get(summary, "data.n_test");
  const fullM = get(summary, "headline.recall_at_k.lgbm.v"), fullP = get(summary, "headline.recall_at_k.prior_cost.v");
  let sampleNote = `The picture is a random ${numSpan(fmt.int(N))}` + (isNum(nTest) ? ` of the ${numSpan(fmt.int(nTest))} member-years in the test set` : " test member-years");
  if (isNum(fullM) && isNum(fullP) && totalPos > 0) {
    sampleNote += `, so its shares (${numSpan(fmt.pct0(found.model / totalPos))} for the model, ${numSpan(fmt.pct0(found.prior / totalPos))} for last year's cost) sit a little off the full-test figures above (${numSpan(fmt.pct(fullM))} and ${numSpan(fmt.pct(fullP))})`;
  }
  sampleNote += ".";

  // ---- sentence ----
  const sentence = $("field-sentence");
  function describe(key) {
    const f = found[key];
    const list = `the call list of ${numSpan(fmt.int(B))}`;
    const of = `${numSpan(fmt.int(f))} of the ${numSpan(fmt.int(totalPos))} members who ended up high cost`;
    let text;
    if (key === "random") {
      text = `In random order, ${list} holds ${of}. That is what calling without any information reaches.`;
    } else if (key === "prior") {
      text = `Sorted by last year's cost, ${list} holds ${of}. The other ${numSpan(fmt.int(totalPos - f))} are not called.`;
    } else {
      const d = f - found.prior;
      const cmp = d > 0 ? `${numSpan(fmt.int(d))} more than last year's cost finds`
        : d < 0 ? `${numSpan(fmt.int(-d))} fewer than last year's cost finds`
        : "the same number last year's cost finds";
      text = `Sorted by the model's score, ${list} holds ${of}, ${cmp}.`;
      if (newFinds > 0) text += ` ${numSpan(fmt.int(newFinds))} of them were not on the last-year's-cost list.`;
    }
    return `${text} ${sampleNote}`;
  }

  // ---- canvas ----
  const stage = $("field-stage");
  const canvas = $("field-canvas");
  const ctx = canvas.getContext("2d");
  const labBand = $("field-label-band");
  const labRest = $("field-label-rest");
  let L = null; // layout
  let colors = {};
  const cur = { x: new Float32Array(N), y: new Float32Array(N) };
  const from = { x: new Float32Array(N), y: new Float32Array(N) };
  const to = { x: new Float32Array(N), y: new Float32Array(N) };
  // ?order=model (or prior, random) opens on that ordering and skips the opening sequence
  const asked = new URLSearchParams(window.location.search).get("order");
  const fixed = ORDERS.some((o) => o.key === asked);
  let order = fixed ? asked : reducedMotion() ? "model" : "random";
  let animStart = 0;
  let raf = 0;
  let hover = -1;

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name) => cs.getPropertyValue(name).trim();
    colors = { signal: v("--signal"), dot: v("--dot"), ink: v("--ink"), band: v("--paper-2"), paper: v("--paper") };
  }

  function layout(W) {
    const wide = W >= 700;
    const pad = 3;
    const inner = W - pad * 2;
    if (wide) {
      const R = W >= 980 ? 20 : 25;
      const bandCols = Math.ceil(B / R);
      const restCols = Math.ceil((N - B) / R);
      const gap = 1.7;
      const cell = Math.min(15, inner / (bandCols + restCols + gap));
      const top = 46;
      L = {
        wide, cell, top, height: top + R * cell + pad + 2,
        xy(p) {
          if (p < B) return [pad + (Math.floor(p / R) + 0.5) * cell, top + ((p % R) + 0.5) * cell];
          const q = p - B;
          return [pad + (bandCols + gap + Math.floor(q / R) + 0.5) * cell, top + ((q % R) + 0.5) * cell];
        },
        index(px, py) {
          const row = Math.floor((py - top) / cell);
          if (row < 0 || row >= R) return -1;
          const c = (px - pad) / cell;
          if (c >= 0 && c < bandCols) { const p = Math.floor(c) * R + row; return p < B ? p : -1; }
          const c2 = c - bandCols - gap;
          if (c2 < 0) return -1;
          const p = B + Math.floor(c2) * R + row;
          return p < N ? p : -1;
        },
        band: { x: pad - 1, y: top - 1, w: bandCols * cell + 2, h: R * cell + 2 },
        labels: [{ left: 0, top: 0, width: Math.max(120, bandCols * cell + gap * cell - 8) }, { left: pad + (bandCols + gap) * cell, top: 0, width: null }],
      };
    } else {
      const C = W < 430 ? 40 : 50;
      const cell = inner / C;
      const bandRows = Math.ceil(B / C);
      const restRows = Math.ceil((N - B) / C);
      const top = 26;
      const gapH = 40;
      const restTop = top + bandRows * cell + gapH;
      L = {
        wide, cell, top, height: restTop + restRows * cell + pad + 2,
        xy(p) {
          if (p < B) return [pad + ((p % C) + 0.5) * cell, top + (Math.floor(p / C) + 0.5) * cell];
          const q = p - B;
          return [pad + ((q % C) + 0.5) * cell, restTop + (Math.floor(q / C) + 0.5) * cell];
        },
        index(px, py) {
          const col = Math.floor((px - pad) / cell);
          if (col < 0 || col >= C) return -1;
          if (py >= top && py < top + bandRows * cell) { const p = Math.floor((py - top) / cell) * C + col; return p < B ? p : -1; }
          if (py < restTop) return -1;
          const p = B + Math.floor((py - restTop) / cell) * C + col;
          return p < N ? p : -1;
        },
        band: { x: pad - 1, y: top - 1, w: C * cell + 2, h: bandRows * cell + 2 },
        labels: [{ left: 0, top: 0, width: null }, { left: 0, top: restTop - 26, width: null }],
      };
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(L.height * dpr);
    canvas.style.height = L.height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    [labBand, labRest].forEach((el, i) => {
      el.style.left = L.labels[i].left + "px";
      el.style.top = L.labels[i].top + "px";
      el.style.width = L.labels[i].width ? L.labels[i].width + "px" : "auto";
      el.style.whiteSpace = L.labels[i].width ? "normal" : "nowrap";
    });
  }

  function target(key, into) {
    const p = pos[key];
    for (let i = 0; i < N; i++) {
      const [x, y] = L.xy(p[i]);
      into.x[i] = x;
      into.y[i] = y;
    }
  }

  function draw(t) {
    const r = Math.max(1.6, L.cell * 0.36);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = colors.band;
    ctx.fillRect(L.band.x, L.band.y, L.band.w, L.band.h);
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(L.band.x, L.band.y, L.band.w, L.band.h);

    // gray dots recede a little while they travel, so the eye can follow the signal dots
    ctx.globalAlpha = t === null ? 1 : 1 - 0.45 * Math.sin(Math.PI * t);
    ctx.fillStyle = colors.dot;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      if (isPos[i]) continue;
      ctx.moveTo(cur.x[i] + r, cur.y[i]);
      ctx.arc(cur.x[i], cur.y[i], r, 0, 6.2832);
    }
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = colors.signal;
    ctx.beginPath();
    const rs = r * 1.08;
    for (let i = 0; i < N; i++) {
      if (!isPos[i]) continue;
      ctx.moveTo(cur.x[i] + rs, cur.y[i]);
      ctx.arc(cur.x[i], cur.y[i], rs, 0, 6.2832);
    }
    ctx.fill();
    if (hover >= 0) {
      ctx.strokeStyle = colors.ink;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cur.x[hover], cur.y[hover], r + 3, 0, 6.2832);
      ctx.stroke();
    }
  }

  function tick(now) {
    const total = DURATION + GRAY_DELAY;
    const el = now - animStart;
    for (let i = 0; i < N; i++) {
      const local = (el - (isPos[i] ? 0 : GRAY_DELAY)) / DURATION;
      const e = easeCubicInOut(Math.max(0, Math.min(1, local)));
      cur.x[i] = from.x[i] + (to.x[i] - from.x[i]) * e;
      cur.y[i] = from.y[i] + (to.y[i] - from.y[i]) * e;
    }
    const t = Math.min(1, el / total);
    draw(t < 1 ? t : null);
    raf = t < 1 ? requestAnimationFrame(tick) : 0;
  }

  function labels(key) {
    const f = found[key];
    const tail = (n) => (L && L.wide ? `<br>${numSpan(fmt.int(n))} high cost` : `, ${numSpan(fmt.int(n))} of them high cost`);
    labBand.innerHTML = `<b>Called:</b> ${numSpan(fmt.int(B))}${tail(f)}`;
    labRest.innerHTML = `<b>Not called:</b> ${numSpan(fmt.int(N - B))}${tail(totalPos - f)}`;
  }

  function setOrder(key, animate) {
    order = key;
    hover = -1;
    hideTip();
    controls.querySelectorAll("input").forEach((inp) => { inp.checked = inp.value === key; });
    sentence.innerHTML = describe(key);
    labels(key);
    canvas.setAttribute("aria-label", `${fmt.int(N)} members as dots, ordered by: ${ORDERS.find((o) => o.key === key).name}.`);
    if (!L) return;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    if (animate && !reducedMotion()) {
      from.x.set(cur.x);
      from.y.set(cur.y);
      target(key, to);
      animStart = performance.now();
      raf = requestAnimationFrame(tick);
    } else {
      target(key, cur);
      draw(null);
    }
  }

  function resize() {
    const W = Math.round(stage.clientWidth);
    if (!W) return;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    layout(W);
    target(order, cur);
    labels(order);
    draw(null);
  }

  // ---- hover and tap: which member is this ----
  function pick(e) {
    if (raf || !L) return;
    const box = canvas.getBoundingClientRect();
    const p = L.index(e.clientX - box.left, e.clientY - box.top);
    const i = p >= 0 ? at[order][p] : -1;
    if (i === hover) return;
    hover = i;
    draw(null);
    if (i < 0) return hideTip();
    const m = members[i];
    const rows = [
      ["Cost the next year", fmt.usd(m.cost_next)],
      ["Model rank", `${fmt.int(pos.model[i] + 1)} of ${fmt.int(N)}`],
      ["Last-year's-cost rank", `${fmt.int(pos.prior[i] + 1)} of ${fmt.int(N)}`],
    ].map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("");
    showTip(`<b>${isPos[i] ? "Ended up high cost" : "Did not end up high cost"}</b><table>${rows}</table>`, e.clientX, e.clientY);
  }
  canvas.addEventListener("pointermove", pick);
  canvas.addEventListener("pointerdown", pick);
  canvas.addEventListener("pointerleave", () => { if (hover >= 0) { hover = -1; draw(null); } hideTip(); });

  // ---- wiring ----
  let timers = [];
  const cancelAuto = () => { timers.forEach(clearTimeout); timers = []; };
  controls.addEventListener("change", (e) => { cancelAuto(); setOrder(e.target.value, true); });
  controls.addEventListener("pointerdown", cancelAuto);

  readColors();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); if (L) draw(null); });
  new ResizeObserver(() => {
    if (L && Math.round(stage.clientWidth) === L.W) return;
    resize();
    L.W = Math.round(stage.clientWidth);
  }).observe(stage);
  resize();
  L.W = Math.round(stage.clientWidth);
  setOrder(order, false);

  // The one piece of motion on the page: play random, then last year's cost, then the model, once.
  if (!reducedMotion() && !fixed) {
    const io = new IntersectionObserver((entries) => {
      if (!entries.some((en) => en.isIntersecting)) return;
      io.disconnect();
      timers.push(setTimeout(() => setOrder("prior", true), 900));
      timers.push(setTimeout(() => setOrder("model", true), 900 + 2600));
    }, { threshold: 0.35 });
    io.observe(canvas);
  }

  return { setOrder, found, totalPos, B, N };
}
