// node tests/check_crossbar.mjs — checks crossbar.js against the identities the page claims,
// and prints the numbers quoted in the README (page defaults, fixed seeds).
import * as X from "../crossbar.js";

let ok = true;
const check = (name, cond, msg) => { console.log(`${cond ? "PASS" : "FAIL"}  ${name}: ${msg}`); ok &&= cond; };

// 1. the sampled time average is 1/2 (a + b)^2
{
  let m = 0;
  for (const [a, b] of [[1, 0.1], [1, -0.1], [-1, 0.37], [0.3, 0], [2.2, -1.7]]) m = Math.max(m, Math.abs(X.stressSampled(a, b) - X.stress(a, b)));
  check("time-averaged stress", m < 1e-12, `max |sampled - 1/2 (a+b)^2| = ${m.toExponential(1)}`);
}

// 2. habit sand writes exactly the outer product k v^T / K (phase-coded codes, no clipping)
{
  const R = X.rng(7), k = X.codes(1, X.K, 0, R)[0], v = X.codes(1, X.V, 0, R)[0];
  const b = X.makeBeach("habit");
  X.write(b, k, v);
  let m = 0;
  for (let i = 0; i < X.V; i++) for (let j = 0; j < X.K; j++) m = Math.max(m, Math.abs(b.D[i][j] - k[j] * v[i] / X.K));
  check("habit write = k v^T / K", m < 1e-12, `max error ${m.toExponential(1)}`);
}

// 3. plain scour writes the same cross term plus a uniform dig of r (1/2 A^2 + 1/2 beta^2 - SIGMA_M)
{
  const R = X.rng(8), k = X.codes(1, X.K, 0, R)[0], v = X.codes(1, X.V, 0, R)[0];
  const b = X.makeBeach("scour"), off = b.r * (0.5 + 0.5 * b.beta ** 2 - X.SIGMA_M);
  X.write(b, k, v);
  let m = 0;
  for (let i = 0; i < X.V; i++) for (let j = 0; j < X.K; j++) m = Math.max(m, Math.abs(b.D[i][j] - (k[j] * v[i] / X.K + off)));
  check("scour write = k v^T / K + uniform dig", m < 1e-12, `uniform dig per write ${off.toFixed(4)} vs cross term ${(1 / X.K).toFixed(4)}; error ${m.toExponential(1)}`);
}

// 4. pings never write, for every rule
{
  let moved = 0;
  for (const rule of ["scour", "habit", "delta"]) {
    const R = X.rng(9), keys = X.codes(8, X.K, 0.3, R), vals = X.codes(8, X.V, 0, R);
    const b = X.makeBeach(rule); X.store(b, keys, vals, 1);
    const before = b.D.map(r => Float64Array.from(r));
    for (let t = 0; t < 50; t++) X.read(b, keys[t % 8]);
    for (let i = 0; i < X.V; i++) for (let j = 0; j < X.K; j++) moved = Math.max(moved, Math.abs(b.D[i][j] - before[i][j]));
  }
  check("pings move no sand", moved === 0, `max |change| after 50 pings = ${moved}`);
}

// 5. order: habit writes commute (no clipping), delta writes do not
function orderDiff(rule, P) {
  const R = X.rng(11), keys = X.codes(P, X.K, 0.3, R), vals = X.codes(P, X.V, 0, R);
  const f = X.makeBeach(rule), r = X.makeBeach(rule);
  X.store(f, keys, vals); X.store(r, [...keys].reverse(), [...vals].reverse());
  let m = 0;
  for (let i = 0; i < X.V; i++) for (let j = 0; j < X.K; j++) m = Math.max(m, Math.abs(f.D[i][j] - r.D[i][j]));
  return { m, clipped: f.clipped + r.clipped };
}
{
  const h = orderDiff("habit", 8), d = orderDiff("delta", 8), s = orderDiff("scour", 8);
  check("habit: AB = BA", h.m < 1e-12 && h.clipped === 0, `max |D_fwd - D_rev| = ${h.m.toExponential(1)}`);
  check("delta: AB != BA", d.m > 1e-3, `max |D_fwd - D_rev| = ${d.m.toFixed(4)}`);
  console.log(`      scour: max |D_fwd - D_rev| = ${s.m.toExponential(1)}, basins clipped ${s.clipped}`);
}

// 6. delta rule, weak-teacher limit: orthogonal keys are stored exactly in one pass
{
  const H = [[1]];
  while (H.length < X.K) { const n = H.length; for (let i = 0; i < n; i++) { H.push([...H[i], ...H[i].map(x => -x)]); H[i] = [...H[i], ...H[i]]; } }
  const R = X.rng(12), vals = X.codes(X.K, X.V, 0, R);
  for (const beta of [0.1, 1e-4]) {
    const b = X.makeBeach("delta", beta); X.store(b, H, vals, 1);
    const rc = X.recall(b, H, vals);
    console.log(`      delta, 16 orthogonal keys, beta ${beta}: bits ${rc.bits.toFixed(4)}, cos ${rc.cos.toFixed(6)}`);
    if (beta === 1e-4) check("delta exact in weak-teacher limit", rc.cos > 0.99999, `cos ${rc.cos.toFixed(6)}`);
  }
}

// Numbers for the README (page defaults)
const Ps = [2, 4, 8, 12, 16, 24, 32];
for (const ov of [0, 0.5]) {
  console.log(`\ncapacity, key overlap ${ov} (bit accuracy, 8 sets):  P = ${Ps.join(", ")}`);
  for (const [rule, passes] of [["scour", 1], ["habit", 1], ["delta", 1], ["delta", 3]]) {
    console.log(`  ${(rule + (rule === "delta" ? ` x${passes}` : "")).padEnd(9)} ${X.capacity(rule, Ps, ov, 8, passes).map(x => x.toFixed(3)).join("  ")}`);
  }
}
console.log("\ntwo beaches and a puddle, P = 12, overlap 0.5 (bit accuracy, mean of 8 seeds)");
for (const [rule, passes] of [["scour", 1], ["habit", 1], ["delta", 3]]) {
  const row = Object.keys(X.GATES).map(g => { let s = 0; for (let t = 0; t < 8; t++) s += X.chain(rule, 12, 0.5, g, 100 + t, passes); return `${g} ${(s / 8).toFixed(3)}`; });
  console.log(`  ${(rule + (rule === "delta" ? ` x${passes}` : "")).padEnd(9)} ${row.join("   ")}`);
}
console.log(ok ? "\nall checks pass" : "\nSOME CHECKS FAILED");
process.exit(ok ? 0 : 1);
