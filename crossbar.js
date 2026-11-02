// crossbar.js — an idealised "crossbar beach": the algebra behind attention.html.
//
// K key channels run east from inlets on the west shore. V harbour channels run north from
// harbours on the south shore. Where a key channel crosses a harbour channel there is a small
// basin (a puddle). Its sand depth D[i][j] is the memory.
//
// Model assumptions (stated on the page):
//  1. Both channels pour into the basin's sloshing mode, so its amplitude is a_j + b_i
//     (key wave a_j = A k_j from the inlet, teacher wave b_i = beta t_i from the harbour).
//     Signed amplitudes are phase 0 or pi of one swell frequency.
//  2. Basin sand moves with the time-averaged sloshing energy sigma = <((a+b) cos wt)^2>,
//     = 1/2 (a + b)^2 for one swell frequency, and only if sigma exceeds the mobility
//     (Shields) threshold SIGMA_M. Below it, nothing moves.
//  3. A basin passes the key channel's wave into the harbour channel in proportion to its
//     depth deviation D (signed: which side of the basin's resonance it is tuned to), weakly,
//     in one pass, with no echoes (Born approximation). So harbour i hears sum_j D[i][j] a_j.
//  4. Depth is bounded: |D| <= D_MAX (bedrock below, the water surface above).
// This computes the algebra of that idealised crossbar. It does NOT solve the wave equation;
// the wave-equation beach is index.html and the Python in this repo.

export const K = 16, V = 16;
export const A = 1.0;          // key swell amplitude
export const EPS = 0.3;        // ping amplitude (weak)
export const SIGMA_M = 0.2;    // mobility threshold: 1/2 EPS^2 = 0.045 < SIGMA_M, so pings never write
export const D_MAX = 1.0;
export const NS = 32;          // samples per period for the time average

// small seeded RNG (mulberry32)
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function stressSampled(a, b) {       // <((a + b) cos wt)^2> over one period, sampled
  let s = 0;
  for (let n = 0; n < NS; n++) { const c = Math.cos(2 * Math.PI * n / NS); s += (a + b) * (a + b) * c * c; }
  return s / NS;
}
export const stress = (a, b) => 0.5 * (a + b) * (a + b);   // its exact value; the test checks they agree

export function makeBeach(rule, beta = 0.1) {
  // rule: "scour" (plain sand: fixed reference SIGMA_M), "habit" (sliding threshold: reference
  // theta = the stress the basin is used to), "delta" (habit + listen before writing).
  const D = Array.from({ length: V }, () => new Float64Array(K));
  // Habituation: under random phase-coded traffic (|a| = A, |b| = beta) the cross term
  // averages out, so the habituated stress is 1/2 A^2 + 1/2 beta^2 at every basin.
  const theta = 0.5 * A * A + 0.5 * beta * beta;
  const r = 1 / (K * A * beta);              // so the clean cross term writes D += k v^T / K
  return { rule, beta, D, theta, r, clipped: 0 };
}

export function read(bch, key) {           // weak ping: harbour i hears sum_j D_ij * EPS k_j, rescaled by 1/EPS
  const y = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    let s = 0;
    for (let j = 0; j < K; j++) {
      const a = EPS * key[j];
      if (stress(a, 0) > SIGMA_M) throw new Error("ping above threshold");
      s += bch.D[i][j] * a;
    }
    y[i] = s / EPS;
  }
  return y;
}

export function write(bch, key, teach) {   // one exposure: key swell + teacher swell, sand moves
  const { D, r } = bch;
  const ref = bch.rule === "scour" ? SIGMA_M : bch.theta;
  for (let i = 0; i < V; i++) {
    const b = bch.beta * teach[i];
    for (let j = 0; j < K; j++) {
      const sg = stress(A * key[j], b);
      if (sg <= SIGMA_M) continue;           // too calm to move sand at all
      let d = D[i][j] + r * (sg - ref);
      if (d > D_MAX) { d = D_MAX; bch.clipped++; } else if (d < -D_MAX) { d = -D_MAX; bch.clipped++; }
      D[i][j] = d;
    }
  }
}

export function store(bch, keys, vals, passes = 1) {
  for (let p = 0; p < passes; p++) {
    for (let n = 0; n < keys.length; n++) {
      let teach = vals[n];
      if (bch.rule === "delta") {            // listen first: teach only what the beach gets wrong
        const y = read(bch, keys[n]);
        teach = vals[n].map((v, i) => v - y[i]);
      }
      write(bch, keys[n], teach);
    }
  }
}

export function codes(n, len, overlap, rand) {
  // n random +-1 patterns of length len. overlap in [0,1): each pattern is a shared base
  // pattern with each bit flipped with probability (1 - overlap) / 2, so mean pairwise
  // correlation is overlap^2.
  const base = Array.from({ length: len }, () => (rand() < 0.5 ? -1 : 1));
  return Array.from({ length: n }, () => base.map(b => (rand() < (1 - overlap) / 2 ? -b : b)));
}

export function recall(bch, keys, vals) {   // mean bit accuracy and mean cosine over stored pairs
  let bits = 0, cs = 0;
  for (let n = 0; n < keys.length; n++) {
    const y = read(bch, keys[n]);
    let dot = 0, ny = 0, nv = 0;
    for (let i = 0; i < V; i++) {
      bits += Math.sign(y[i]) === vals[n][i] ? 1 : 0;
      dot += y[i] * vals[n][i]; ny += y[i] * y[i]; nv += 1;
    }
    cs += ny > 0 ? dot / Math.sqrt(ny * nv) : 0;
  }
  return { bits: bits / (keys.length * V), cos: cs / keys.length };
}

// Gates between beaches (puddles), applied to the heard amplitude of each harbour channel.
export const GATES = {
  none: a => a,
  overtop: a => Math.sign(a) * Math.max(Math.abs(a) - 0.5, 0),     // spills only what is over the crest
  breaker: a => Math.max(-1, Math.min(1, 3 * a)),                   // shoaling gain x3, then depth-limited breaking
};

export function chain(rule, P, overlap, gate, seed, passes = 1, beta = 0.1) {
  // Beach A stores x -> y, beach B stores y -> z. Ping x into A, pass the harbour outputs
  // through the gate into B's inlets, read B's harbours. Bit accuracy against z.
  const R = rng(seed);
  const X = codes(P, K, overlap, R), Y = codes(P, V, overlap, R), Z = codes(P, V, 0, R);
  const a = makeBeach(rule, beta), b = makeBeach(rule, beta);
  store(a, X, Y, passes); store(b, Y, Z, passes);
  let bits = 0;
  for (let n = 0; n < P; n++) {
    const y = read(a, X[n]).map(GATES[gate]);
    // B's inlets receive the gated harbour outputs of A as their key swell
    const z = new Float64Array(V);
    for (let i = 0; i < V; i++) { let s = 0; for (let j = 0; j < K; j++) s += b.D[i][j] * y[j]; z[i] = s; }
    for (let i = 0; i < V; i++) bits += Math.sign(z[i]) === Z[n][i] ? 1 : 0;
  }
  return bits / (P * V);
}

export function capacity(rule, Ps, overlap, sets, passes = 1, beta = 0.1, seed0 = 1) {
  return Ps.map(P => {
    let s = 0;
    for (let t = 0; t < sets; t++) {
      const R = rng(seed0 + 1000 * t + P);
      const keys = codes(P, K, overlap, R), vals = codes(P, V, 0, R);
      const bch = makeBeach(rule, beta);
      store(bch, keys, vals, passes);
      s += recall(bch, keys, vals).bits;
    }
    return s / sets;
  });
}
