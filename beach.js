// The beach, in JavaScript: the same medium as beach/medium.py (checked against it by
// tests/check_js.mjs).  WallMemoryLaw's stadium, leapfrog stepping, damping 0.006, Neumann
// walls; the floor is sand, depth = 1 - sand, every face carries g * depth.

export const N = 96, C0 = 0.65, G = C0 * C0, DT = 0.5, SIGMA = 1.55;
export const CENTER = (N - 1) / 2, SL = 18, SR = 22;
export const SAND_MIN = 0, SAND_MAX = 0.6;
export const PORTS = {
  west: [50, 18], north: [30, 48], south: [66, 40], harbourN: [36, 78], harbourS: [60, 78],
};
// calibration constants from the Python runs (g1_order.py)
export const SIGMA_C = 0.6011319568283339;      // critical stress (median of swell A on seed 0)
export const PROBE_AMP = 0.07514570671630716;    // weak ping: max stress <= 0.125 sigma_c
export const RATE = 0.007272281815208983;        // scour rate at write strength 0.04

export function makeBeach(damp = 0.006) {
  const inside = new Uint8Array(N * N);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const X = c - CENTER, Y = r - CENTER;
    const box = Math.abs(X) <= SL && Math.abs(Y) <= SR;
    const cap = (Math.abs(X) - SL) ** 2 + Y * Y <= SR * SR;
    inside[r * N + c] = box || cap ? 1 : 0;
  }
  const idx = new Int32Array(N * N).fill(-1);
  const cells = [];
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++)
    if (inside[r * N + c]) { idx[r * N + c] = cells.length; cells.push([r, c]); }
  const n = cells.length;
  const fi = [], fj = [];
  for (const [dr, dc] of [[1, 0], [0, 1]]) {
    for (let k = 0; k < n; k++) {
      const r2 = cells[k][0] + dr, c2 = cells[k][1] + dc;
      if (r2 < N && c2 < N && inside[r2 * N + c2]) { fi.push(k); fj.push(idx[r2 * N + c2]); }
    }
  }
  const deg = new Float64Array(n);
  for (let f = 0; f < fi.length; f++) { deg[fi[f]]++; deg[fj[f]]++; }
  return { inside, idx, cells, n, fi: Int32Array.from(fi), fj: Int32Array.from(fj), nf: fi.length, deg, damp };
}

export function port(B, name, sigma = SIGMA) {
  const [r0, c0] = PORTS[name];
  const w = new Float64Array(B.n);
  for (let k = 0; k < B.n; k++) {
    const [r, c] = B.cells[k];
    w[k] = Math.exp(-((r - r0) ** 2 + (c - c0) ** 2) / (2 * sigma * sigma));
  }
  return w;
}

export function decodeBed(b64) {
  const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return Float64Array.from(new Float32Array(bytes.buffer));
}

// A running tank: wave state + sand + time-averaged stress.
export class Tank {
  constructor(B, sand) {
    this.B = B;
    this.sand = Float64Array.from(sand);
    this.u = new Float64Array(B.n);
    this.prev = new Float64Array(B.n);
    this.Lu = new Float64Array(B.n);
    this.inst = new Float64Array(B.n);
    this.stress = new Float64Array(B.n);
    this.w = new Float64Array(B.nf);
    this.t = 0;
    this.refreshFaces();
  }
  refreshFaces() {
    const { fi, fj, nf } = this.B, s = this.sand;
    for (let f = 0; f < nf; f++) this.w[f] = G * (1 - 0.5 * (s[fi[f]] + s[fj[f]]));
  }
  calm() { this.u.fill(0); this.prev.fill(0); this.stress.fill(0); }
  addPulse(weights, amp) { for (let k = 0; k < this.B.n; k++) this.u[k] += amp * weights[k]; }
  // drives: [{w: weights, amp, omega}] ; f[n] = amp * weights * cos(omega n DT)
  // alpha > 0 also updates the time-averaged stress (EMA of (grad u)^2 per cell).
  step(drives = [], alpha = 0) {
    const { n, fi, fj, nf, deg, damp } = this.B;
    const a = 1 + damp * DT / 2, b = 1 - damp * DT / 2, dt2 = DT * DT;
    const u = this.u, prev = this.prev, Lu = this.Lu, w = this.w, inst = this.inst;
    Lu.fill(0);
    if (alpha) inst.fill(0);
    for (let f = 0; f < nf; f++) {
      const i = fi[f], j = fj[f], d = u[i] - u[j];
      Lu[i] += w[f] * d; Lu[j] -= w[f] * d;
      if (alpha) { const g2 = d * d; inst[i] += g2; inst[j] += g2; }
    }
    const phase = drives.map(dv => dv.amp * Math.cos(dv.omega * this.t * DT));
    for (let k = 0; k < n; k++) {
      let fk = 0;
      for (let q = 0; q < drives.length; q++) fk += phase[q] * drives[q].w[k];
      prev[k] = (2 * u[k] - b * prev[k] - dt2 * Lu[k] + dt2 * fk) / a;  // prev now holds u[n+1]
    }
    this.prev = u; this.u = prev; this.t++;
    if (alpha) for (let k = 0; k < n; k++) this.stress[k] += alpha * (inst[k] / deg[k] - this.stress[k]);
  }
  // scour (local) or exner (conservative) write from the time-averaged stress
  write(rate, mode = "scour", sigc = SIGMA_C) {
    const { n, fi, fj, nf } = this.B, s = this.sand, S = this.stress;
    const E = new Float64Array(n);
    for (let k = 0; k < n; k++) E[k] = Math.max(S[k] - sigc, 0);
    let moved = 0;
    if (mode === "scour") {
      for (let k = 0; k < n; k++) {
        const ns = Math.min(SAND_MAX, Math.max(SAND_MIN, s[k] - rate * E[k]));
        moved += Math.abs(ns - s[k]); s[k] = ns;
      }
    } else {
      const lap = new Float64Array(n);
      for (let f = 0; f < nf; f++) { const d = E[fj[f]] - E[fi[f]]; lap[fi[f]] += d; lap[fj[f]] -= d; }
      for (let k = 0; k < n; k++) {
        const ns = Math.min(SAND_MAX, Math.max(SAND_MIN, s[k] + rate * lap[k]));
        moved += Math.abs(ns - s[k]); s[k] = ns;
      }
    }
    if (moved > 0) this.refreshFaces();
    return moved;
  }
}
