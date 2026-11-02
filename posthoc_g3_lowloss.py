"""POST HOC (not pre-registered): G3 rerun at low damping, after G2 showed loss destroys the teacher alignment.
G3: can the beach learn to route two swells to two harbours?  (PREREG.md, P3a-P3d)"""
import json
import sys
import numpy as np

from beach.medium import Beach, rough_bed, scour, exner, apply

FREQS = [(0.40, "harbourN", "harbourS"), (0.47, "harbourS", "harbourN")]
SEEDS = range(200, 205)
EPOCHS = 300
STEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001

import sys
DAMP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0005
bch = Beach(damp=DAMP)
src = bch.port("west")
H = {k: bch.port(k) for k in ("harbourN", "harbourS")}


def contrast(sand):
    c = []
    for w, right, wrong in FREQS:
        eta = bch.solve(sand, w, src)
        c.append(10 * np.log10(abs(H[right] @ eta) ** 2 / abs(H[wrong] @ eta) ** 2))
    return float(np.mean(c))


def norm_step(d):
    m = np.abs(d).max()
    return d * (STEP / m) if m > 0 else d


def direction(arm, sand, P, sigc):
    tot = np.zeros(bch.n)
    for k, (w, right, wrong) in enumerate(FREQS):
        A_lu = None
        eta = bch.solve(sand, w, src)
        yr, yw = H[right] @ eta, H[wrong] @ eta
        # J_k = (|yw|^2 - |yr|^2)/P_k ;  e = 2 dJ/dconj(y)
        er, ew = -2 * yr / P[k], 2 * yw / P[k]
        if arm == "gd":
            g = np.zeros(bch.n)
            for rd, e in ((H[right], er), (H[wrong], ew)):
                gg, _, _ = bch.grad_sand(sand, w, src, rd, lambda y, e=e: e)
                g += gg
            tot += -g
            continue
        if arm == "natural":
            tot += scour(bch, bch.stress(eta), sigc[k], 1.0)
            continue
        # physical teacher swell from both harbours, amplitude e (beta = 1)
        tau = bch.solve(sand, w, H[right] * er + H[wrong] * ew)
        wr = exner if arm == "teacher_exner" else scour
        if arm.endswith("one"):
            tot += wr(bch, bch.stress(eta, tau), sigc[k], 1.0) - wr(bch, bch.stress(eta), sigc[k], 1.0)
        else:
            tot += 0.5 * (wr(bch, bch.stress(eta, tau), sigc[k], 1.0) - wr(bch, bch.stress(eta, -tau), sigc[k], 1.0))
    return tot


ARMS = ["gd", "teacher_scour", "teacher_scour_one", "natural"]
out = {"curves": {}, "final": {}}
for seed in SEEDS:
    s0 = rough_bed(bch, seed)
    P, sigc = [], []
    for w, right, wrong in FREQS:
        eta = bch.solve(s0, w, src)
        P.append(abs(H[right] @ eta) ** 2 + abs(H[wrong] @ eta) ** 2)
        sigc.append(float(np.quantile(bch.stress(eta), 0.5)))
    c0 = contrast(s0)
    out["final"].setdefault("initial", []).append(c0)
    for arm in ARMS:
        sand = s0.copy()
        curve = [c0]
        for ep in range(EPOCHS):
            sand = apply(sand, norm_step(direction(arm, sand, P, sigc)))
            if (ep + 1) % 10 == 0:
                curve.append(contrast(sand))
        out["curves"].setdefault(arm, []).append(curve)
        out["final"].setdefault(arm, []).append(curve[-1])
        np.save(f"results/g3lo_sand_{arm}_{seed}.npy", sand)
        print(seed, arm, f"{c0:+.2f} -> {curve[-1]:+.2f} dB  (best {max(curve):+.2f})", flush=True)

gain = {a: float(np.mean(np.array(out["final"][a]) - np.array(out["final"]["initial"]))) for a in ARMS}
out["mean_gain_dB"] = gain
out["would_gd_ge_10dB"] = gain["gd"] >= 10
out["would_teacher_ge_half_gd"] = gain["teacher_scour"] >= 0.5 * gain["gd"]

out["would_natural_within_3dB"] = abs(gain["natural"]) <= 3
json.dump(out, open(f"results/posthoc_g3_lowloss_d{DAMP}_s{STEP}.json", "w"), indent=1)
print({k: v for k, v in out.items() if k not in ("curves", "final")})
