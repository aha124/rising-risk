// Checks that the browser scorer gives the same answer as Python.
// Run from anywhere: node docs/tests/parity.mjs
// Fails (exit code 1) if any calibrated probability differs by more than 1e-9.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { rawScore, calibrate } from "../js/scorer.js";

const TOL = 1e-9;
const dataDir = join(dirname(fileURLToPath(import.meta.url)), "..", "data");
const load = (name) => JSON.parse(readFileSync(join(dataDir, name), "utf8"));

const model = load("model_browser.json");
const parity = load("parity.json");

if (!Array.isArray(parity.rows) || parity.rows.length === 0) {
  console.error("parity.json has no rows");
  process.exit(1);
}
if (model.trees.length !== model.n_trees) {
  console.error(`n_trees says ${model.n_trees} but the file holds ${model.trees.length} trees`);
  process.exit(1);
}

let worstP = 0;
let worstRaw = 0;
let failures = 0;
let rowsWithNulls = 0;

parity.rows.forEach((row, i) => {
  if (row.length !== model.features.length) {
    console.error(`row ${i}: ${row.length} values, model has ${model.features.length} features`);
    failures++;
    return;
  }
  if (row.some((v) => v === null)) rowsWithNulls++;
  const raw = rawScore(model, row);
  const p = calibrate(model, raw);
  const dRaw = Math.abs(raw - parity.raw[i]);
  const dP = Math.abs(p - parity.p[i]);
  worstRaw = Math.max(worstRaw, dRaw);
  worstP = Math.max(worstP, dP);
  if (!(dP <= TOL)) {
    failures++;
    if (failures <= 5) console.error(`row ${i}: js p=${p} python p=${parity.p[i]} (raw ${raw} vs ${parity.raw[i]})`);
  }
});

const kind = model.fixture || parity.fixture ? " (fixture model)" : "";
console.log(
  `${parity.rows.length} rows${kind}, ${rowsWithNulls} with nulls, ${model.trees.length} trees. ` +
    `Largest difference: probability ${worstP.toExponential(2)}, raw score ${worstRaw.toExponential(2)}.`
);
if (failures > 0) {
  console.error(`FAIL: ${failures} rows differ from Python by more than ${TOL}`);
  process.exit(1);
}
console.log("PASS");
