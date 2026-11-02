// node tests/check_js.mjs  — checks beach.js against the Python medium (tests/ref_js.json).
import { readFileSync } from "node:fs";
import { makeBeach, port, decodeBed, Tank, SIGMA } from "../beach.js";
import { BEDS } from "../beds.js";

const ref = JSON.parse(readFileSync(new URL("./ref_js.json", import.meta.url)));
const B = makeBeach();
let ok = true;

// 1. a ping from west, heard at harbourS, 600 steps, on bed 300
{
  const T = new Tank(B, decodeBed(BEDS.bed300));
  T.addPulse(port(B, "west"), 1.0);
  const hS = port(B, "harbourS");
  let maxd = 0, maxr = 0;
  for (let t = 0; t < 600; t++) {
    T.step();
    let y = 0; for (let k = 0; k < B.n; k++) y += hS[k] * T.u[k];
    maxd = Math.max(maxd, Math.abs(y - ref.ping_west_harbourS_bed300[t]));
    maxr = Math.max(maxr, Math.abs(ref.ping_west_harbourS_bed300[t]));
  }
  console.log(`ping trace: max |JS - Python| = ${maxd.toExponential(2)} (trace max ${maxr.toFixed(2)})`);
  ok &&= maxd < 1e-9 * maxr;
}

// 2. router contrast, time-domain steady state vs frequency-domain Python
for (const name of ["initial202", "gd202"]) {
  const out = [];
  for (const [omega, right, wrong] of [[0.40, "harbourN", "harbourS"], [0.47, "harbourS", "harbourN"]]) {
    const T = new Tank(B, decodeBed(BEDS[name]));
    const drv = [{ w: port(B, "west"), amp: 1, omega }];
    const hr = port(B, right), hw = port(B, wrong);
    const yr = [], yw = [];
    for (let t = 0; t < 8000; t++) {
      T.step(drv);
      if (t >= 6000) {
        let a = 0, b = 0;
        for (let k = 0; k < B.n; k++) { a += hr[k] * T.u[k]; b += hw[k] * T.u[k]; }
        yr.push(a); yw.push(b);
      }
    }
    const pw = v => { const m = v.reduce((s, x) => s + x, 0) / v.length; return v.reduce((s, x) => s + (x - m) ** 2, 0); };
    out.push(10 * Math.log10(pw(yr) / pw(yw)));
  }
  const py = ref.contrast[name];
  console.log(`${name}: JS ${out.map(v => v.toFixed(2)).join(", ")} dB  vs Python ${py.map(v => v.toFixed(2)).join(", ")} dB`);
  ok &&= out.every((v, i) => Math.abs(v - py[i]) < 0.5);
}
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
