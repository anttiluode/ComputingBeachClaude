"""POST HOC (not pre-registered), after G1 failed P1a.

At scour s = 0.04 the SAND separates AB from BA (LOO 0.98) but the registered read does
not (0.59).  Is the order invisible to waves, or just to that read?
  read      : the registered read (south probe, 16 freqs, both harbours), after the storms
  diff      : read(after) - read(before), each with its own independent read noise
  diff3     : as diff, but probing from all three ports (west, north, south)
Noise is set per raw read (nu x RMS of raw after-reads), so diff pays noise twice.
Also recomputes the worst probe stress over sigma_c on every written bed (lost in the
G1 crash).
"""
import json
import numpy as np

import g1_order as G
from beach.medium import rough_bed

WRITER, S = "scour", 0.04
smax0 = max(G.bch.stress(G.bch.solve(G.s0, w, G.probe)).max() for w in G.READ_W)
G.PROBE_AMP = np.sqrt(0.5 * G.SIGC / smax0) / 2.0      # identical rule to g1_order.main
ports = [G.bch.port(p) for p in ("west", "north", "south")]


def read(sand, srcs):
    f, smax = [], 0.0
    for w in G.READ_W:
        lu_eta = G.bch.solve(sand, w, np.stack([p * G.PROBE_AMP for p in srcs], 1))
        for k in range(len(srcs)):
            eta = lu_eta[:, k]
            smax = max(smax, G.bch.stress(eta).max())
            f += [G.hN @ eta, G.hS @ eta]
    f = np.array(f)
    return np.r_[f.real, f.imag], smax / G.SIGC


rate = G.rate_for(WRITER, S, G.SIGC)
rows = {"read": [], "diff_b": [], "diff_a": [], "d3_b": [], "d3_a": []}
y, groups, worst = [], [], 0.0
for seed in G.SEEDS:
    b0 = rough_bed(G.bch, seed)
    r0, _ = read(b0, [G.probe])
    r30, _ = read(b0, ports)
    for lab, order in ((0, "AB"), (1, "BA")):
        sd = G.history(seed, order, WRITER, rate, G.SIGC)
        r1, p1 = read(sd, [G.probe])
        r31, p3 = read(sd, ports)
        worst = max(worst, p1, p3)
        rows["read"].append(r1)
        rows["diff_b"].append(r0); rows["diff_a"].append(r1)
        rows["d3_b"].append(r30); rows["d3_a"].append(r31)
        y.append(lab); groups.append(seed)
    print(seed, flush=True)
y, groups = np.array(y), np.array(groups)
R = {k: np.array(v) for k, v in rows.items()}
rng = np.random.default_rng(777)
out = {"worst_probe_stress_over_sigma_c": float(worst), "reads_write_nothing": bool(worst < 1.0)}
for nu in (0.0, 0.01, 0.05):
    rms1 = np.sqrt(np.mean(R["read"] ** 2))
    rms3 = np.sqrt(np.mean(R["d3_a"] ** 2))
    n = lambda X, rms: X + nu * rms * rng.standard_normal(X.shape)
    out[f"nu{nu}"] = dict(
        read=G.lda_loo(n(R["read"], rms1), y, groups, rng),
        diff=G.lda_loo(n(R["diff_a"], rms1) - n(R["diff_b"], rms1), y, groups, rng),
        diff3=G.lda_loo(n(R["d3_a"], rms3) - n(R["d3_b"], rms3), y, groups, rng))
    print(nu, out[f"nu{nu}"], flush=True)
json.dump(out, open("results/posthoc_g1_diffread.json", "w"), indent=1)
print(out)
