"""Pilot (post hoc, not registered): ReadWrite x ComputingBeach.
Can a KNOWN write (a probe storm X) reveal the AB/BA order that passive pings read only at ~0.75?
Fresh beds 1100-1131, G4's instruments and LOO-bed decoder."""
import sys, json, numpy as np
sys.path.insert(0, ".")
import g4_floor as G4
import g1_order as G
from beach.medium import rough_bed, WRITERS, apply

b = G.bch
SEEDS = range(1100, 1132)
srcH = b.port("harbourN")
ampH = np.sqrt(b.stress(b.solve(G.s0, G.WA, G.srcA)).max() / b.stress(b.solve(G.s0, 0.50, srcH)).max())


def burst_H(sand):
    for _ in range(G.NW):
        st = b.stress(b.solve(sand, 0.50, srcH * ampH))
        sand = apply(sand, WRITERS["scour"](b, st, G.SIGC, G4.rate))
    return sand


rows = {k: [] for k in ("pass", "pass_steady", "dX", "dH", "dXH", "pass+dX+dH", "sand", "dsandX", "dsandH")}
y, groups = [], []
for s in SEEDS:
    for lab, order in ((0, "AB"), (1, "BA")):
        sd = G4.history(s, order)
        tr, _ = G4.pings(sd)
        p = tr[:, :, ::2].ravel()
        sx = G4.burst_X(sd, 1.0); trx, _ = G4.pings(sx)
        sh = burst_H(sd); trh, _ = G4.pings(sh)
        dx = trx[:, :, ::2].ravel() - p
        dh = trh[:, :, ::2].ravel() - p
        rows["pass"].append(p); rows["pass_steady"].append(G.read(sd)[0])
        rows["dX"].append(dx); rows["dH"].append(dh); rows["dXH"].append(np.r_[dx, dh])
        rows["pass+dX+dH"].append(np.r_[p, dx, dh])
        rows["sand"].append(sd); rows["dsandX"].append(sx - sd); rows["dsandH"].append(sh - sd)
        y.append(lab); groups.append(s)
    print(s, flush=True)
y, groups = np.array(y), np.array(groups)
np.savez_compressed("results/posthoc_write_to_read.npz",
                    y=y, groups=groups, **{k: np.array(v) for k, v in rows.items()})
rng = np.random.default_rng(7)
G.PROBE_AMP = G4.G.PROBE_AMP
out = {}
for k, v in rows.items():
    X = np.array(v)
    if not k.startswith(("sand", "dsand")):
        X = X + 0.01 * np.sqrt(np.mean(X ** 2)) * rng.standard_normal(X.shape)
    out[k] = G.lda_loo(X, y, groups, rng)
print(json.dumps(out, indent=1))
