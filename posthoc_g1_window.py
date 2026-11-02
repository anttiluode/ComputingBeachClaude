"""POST HOC (after G1 P1a failed, and after Sol's ComputingBeachSol G1 passed).

Question: the sand stores swell order (sand decoder 0.98) but the steady-state swell read
could not see it (0.59).  Sol's repo reads with time-domain echoes in a finite window and
passes.  WallMemoryLaw's tour law says a short listening window bounds the scattering
order, so the echo is a low-degree function of the medium.  Is the failure the
infinite-window read, or the 3112-number writable floor?

Same beds as G1 (same writer, rate, threshold, seeds, histories).  Material frozen during
the read (as in G1 and in Sol's protocol).  Read: Gaussian ping (sigma 1.55) released from
rest at west, north and south in turn; traces at both harbours, every 2nd step.
  cumulative windows [0, t1)      t1 in {100, 150, 200, 300, 450, 600}
  sliding windows    [t1-100, t1) same t1
Decoder and noise exactly as G1 (ridge LDA, leave one bed out, nu = 0.01 of feature RMS).

Predictions, written before this script was run:
  W1  scour s=0.04: some cumulative window with t1 <= 200 reaches >= 0.80.
  W2  accuracy falls as the window grows: the [0,600) read is at least 0.10 below the best
      window with t1 <= 200.
  W3  exner s=0.04 stays below 0.70 at every window (its sand decoder was only 0.66).
If W1 fails, the tour-law explanation is wrong for this medium, and the likely culprit is the
dimension of the writable state.
"""
import json
import sys
import numpy as np
import scipy.sparse as sp

import g1_order as G
from beach.medium import DT, SIGMA, G as GRAV

T = 600
T1S = [100, 150, 200, 300, 450, 600]
b = G.bch
rr, cc = b.cells[:, 0], b.cells[:, 1]
from beach.medium import PORTS
pulses = np.stack([np.exp(-((rr - PORTS[p][0]) ** 2 + (cc - PORTS[p][1]) ** 2) / (2 * SIGMA ** 2))
                   for p in ("west", "north", "south")], 1)
H = np.stack([G.hN, G.hS], 0)


def ping_traces(sand):
    """(3 sources, 2 harbours, T) traces, material frozen, leapfrog from rest."""
    a, bb = 1 + b.damp * DT / 2, 1 - b.damp * DT / 2
    Lm = (b.Gr.T @ sp.diags(GRAV * b.face_depth(sand)) @ b.Gr).tocsr()
    u, prev = pulses.copy(), np.zeros_like(pulses)
    out = np.zeros((3, 2, T))
    for t in range(T):
        nxt = (2 * u - bb * prev - DT ** 2 * (Lm @ u)) / a
        prev, u = u, nxt
        out[:, :, t] = (H @ u).T
    return out


def main(writer):
    rate = G.rate_for(writer, 0.04, G.SIGC)
    traces, y, groups = [], [], []
    for seed in G.SEEDS:
        for lab, order in ((0, "AB"), (1, "BA")):
            sd = G.history(seed, order, writer, rate, G.SIGC)
            traces.append(ping_traces(sd)); y.append(lab); groups.append(seed)
    Tr, y, groups = np.array(traces), np.array(y), np.array(groups)
    rng = np.random.default_rng(4242)
    res = {"cumulative": {}, "sliding": {}}
    for kind in ("cumulative", "sliding"):
        for t1 in T1S:
            t0 = 0 if kind == "cumulative" else t1 - 100
            X = Tr[:, :, :, t0:t1:2].reshape(len(Tr), -1)
            X = X + 0.01 * np.sqrt(np.mean(X ** 2)) * rng.standard_normal(X.shape)
            res[kind][str(t1)] = G.lda_loo(X, y, groups, rng)
            print(writer, kind, t1, res[kind][str(t1)], flush=True)
    return res


if __name__ == "__main__":
    out = {w: main(w) for w in ("scour", "exner")}
    sc = out["scour"]["cumulative"]
    early = max(sc[str(t)] for t in T1S if t <= 200)
    out["W1_pass"] = bool(early >= 0.80)
    out["W2_pass"] = bool(sc["600"] <= early - 0.10)
    out["W3_pass"] = bool(max(max(v.values()) for v in out["exner"].values()) < 0.70)
    json.dump(out, open("results/posthoc_g1_window.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k.startswith("W")})
