"""G4: the distributed sand floor on the shared order benchmark (PREREG_G4G5.md)."""
import json
import numpy as np
import scipy.sparse as sp

import g1_order as G
from beach.medium import rough_bed, DT, SIGMA, PORTS, G as GRAV, WRITERS, apply

SEEDS = list(range(300, 348))
WX = 0.43
MX = [0.5, 1.0, 2.0]
T = 700
WRITER, S = "scour", 0.04
b = G.bch
srcX = b.port("south")
ampX = np.sqrt(G.bch.stress(b.solve(G.s0, G.WA, G.srcA)).max() / b.stress(b.solve(G.s0, WX, srcX)).max())
rate = G.rate_for(WRITER, S, G.SIGC)
rr, cc = b.cells[:, 0], b.cells[:, 1]
pulses = np.stack([np.exp(-((rr - PORTS[p][0]) ** 2 + (cc - PORTS[p][1]) ** 2) / (2 * SIGMA ** 2))
                   for p in ("west", "north", "south")], 1)
HARB = np.stack([G.hN, G.hS], 0)
smax0 = max(b.stress(b.solve(G.s0, w, G.probe)).max() for w in G.READ_W)
G.PROBE_AMP = np.sqrt(0.5 * G.SIGC / smax0) / 2.0


def burst_X(sand, m):
    for _ in range(G.NW):
        st = b.stress(b.solve(sand, WX, srcX * ampX))
        sand = apply(sand, WRITERS[WRITER](b, st, G.SIGC, rate * m))
    return sand


def history(seed, order, m=None):
    sand = rough_bed(b, seed)
    for k in order:
        if k == "A":
            sand = G.burst(sand, G.WA, G.srcA, WRITER, rate, G.SIGC)
        elif k == "B":
            sand = G.burst(sand, G.WB, G.srcB * G.ampB, WRITER, rate, G.SIGC)
        else:
            sand = burst_X(sand, m)
    return sand


def pings(sand):
    a, bb = 1 + b.damp * DT / 2, 1 - b.damp * DT / 2
    Lm = (b.Gr.T @ sp.diags(GRAV * b.face_depth(sand)) @ b.Gr).tocsr()
    u, prev = pulses * G.PROBE_AMP, np.zeros_like(pulses)
    out = np.zeros((3, 2, T))
    smax = 0.0
    for t in range(T):
        nxt = (2 * u - bb * prev - DT ** 2 * (Lm @ u)) / a
        prev, u = u, nxt
        out[:, :, t] = (HARB @ u).T
        if t % 25 == 0:
            gr = b.Gr @ u
            smax = max(smax, float(((b.Fc @ (gr ** 2)) / b.deg[:, None]).max()))
    return out, smax / G.SIGC


# arrivals, measured once on seed 300's initial bed
tr0, _ = pings(rough_bed(b, SEEDS[0]))
arr = []
for k in range(3):
    a = [int(np.argmax(np.abs(tr0[k, h]) > 0.01 * np.abs(tr0[k, h]).max())) for h in range(2)]
    arr.append(min(a))


def windows(tr):
    early = np.concatenate([tr[k, :, arr[k]:arr[k] + 100:2].ravel() for k in range(3)])
    late = np.concatenate([tr[k, :, arr[k] + 300:arr[k] + 400:2].ravel() for k in range(3)])
    return tr[:, :, ::2].ravel(), early, late


def evaluate(pairs, rng):
    """pairs: list of (seed, sandA_label0, sandB_label1). Returns accuracies."""
    feats = {k: [] for k in ("R1", "R2", "R3", "R4", "sand")}
    y, groups, worst = [], [], 0.0
    for seed, s0_, s1_ in pairs:
        for lab, sd in ((0, s0_), (1, s1_)):
            r1, p1 = G.read(sd)
            tr, p2 = pings(sd)
            worst = max(worst, p1, p2)
            r2, r3, r4 = windows(tr)
            for k, v in zip(("R1", "R2", "R3", "R4", "sand"), (r1, r2, r3, r4, sd)):
                feats[k].append(v)
            y.append(lab); groups.append(seed)
    y, groups = np.array(y), np.array(groups)
    out = {}
    for k, v in feats.items():
        X = np.array(v)
        if k != "sand":
            X = X + 0.01 * np.sqrt(np.mean(X ** 2)) * rng.standard_normal(X.shape)
        out[k] = G.lda_loo(X, y, groups, rng)
    out["worst_probe_stress_over_sigma_c"] = worst
    return out


def main():
    rng = np.random.default_rng(31337)
    res = {"arrivals_steps_per_source(west,north,south)": arr, "conditions": {}}
    pairs = [(s, history(s, "AB"), history(s, "BA")) for s in SEEDS]
    res["conditions"]["AB_BA"] = evaluate(pairs, rng)
    print("AB_BA", res["conditions"]["AB_BA"], flush=True)
    for m in MX:
        pairs = [(s, history(s, "ABX", m), history(s, "BAX", m)) for s in SEEDS]
        res["conditions"][f"ABX_BAX_m{m}"] = evaluate(pairs, rng)
        print(m, res["conditions"][f"ABX_BAX_m{m}"], flush=True)
    c = res["conditions"]
    best = lambda d: max(d[k] for k in ("R1", "R2", "R3", "R4"))
    res["reads_write_nothing"] = bool(max(d["worst_probe_stress_over_sigma_c"] for d in c.values()) < 1.0)
    res["P4a_pass"] = bool(c["AB_BA"]["R2"] >= 0.65 and c["AB_BA"]["R3"] > c["AB_BA"]["R4"])
    res["P4b_pass"] = bool(all(d["R3"] >= d["R1"] for d in c.values()))
    res["P4c_pass"] = bool(c["ABX_BAX_m1.0"]["sand"] >= 0.80)
    res["P4d_pass"] = bool(best(c["ABX_BAX_m2.0"]) <= best(c["AB_BA"]) - 0.05)
    json.dump(res, open("results/g4.json", "w"), indent=1)
    print({k: v for k, v in res.items() if k.startswith("P4") or k.startswith("reads") or k.startswith("arr")})


if __name__ == "__main__":
    main()
