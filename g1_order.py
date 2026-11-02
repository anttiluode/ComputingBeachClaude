"""G1: does the beach remember the order of two swells?  (PREREG.md, P1a-P1d)

Usage: python g1_order.py [main|slope]
"""
import json
import sys
import numpy as np

from beach.medium import Beach, rough_bed, WRITERS, apply

WA, WB = 0.40, 0.47
NW = 10
STRENGTHS = [0.005, 0.01, 0.02, 0.04]
SEEDS = list(range(48))
READ_W = np.linspace(0.30, 0.55, 16)
NOISES = [0.01, 0.05]

bch = Beach()
srcA, srcB = bch.port("west"), bch.port("north")
probe, hN, hS = bch.port("south"), bch.port("harbourN"), bch.port("harbourS")

# ---- calibration on seed 0 (frozen rules from PREREG)
s0 = rough_bed(bch, 0)
stA = bch.stress(bch.solve(s0, WA, srcA))
stB = bch.stress(bch.solve(s0, WB, srcB))
ampB = np.sqrt(stA.max() / stB.max())
SIGC = float(np.median(stA))


def rate_for(writer, s, sigc):
    d = WRITERS[writer](bch, stA, sigc, 1.0)
    return s / np.abs(d).max()


def burst(sand, w, src, writer, rate, sigc, frozen=None):
    """10 write steps. frozen: if given, fields come from this bed (open loop)."""
    for _ in range(NW):
        bed = sand if frozen is None else frozen
        st = bch.stress(bch.solve(bed, w, src))
        sand = apply(sand, WRITERS[writer](bch, st, sigc, rate))
    return sand


def history(seed, order, writer, rate, sigc, openloop=False):
    sand = rough_bed(bch, seed)
    b0 = sand.copy() if openloop else None
    for k in order:
        if k == "A":
            sand = burst(sand, WA, srcA, writer, rate, sigc, b0)
        else:
            sand = burst(sand, WB, srcB, writer, rate, sigc, b0)
    return sand


PROBE_AMP = None


def read(sand):
    """Weak probe: returns 64 real features, and the max probe stress / sigma_c."""
    feats, smax = [], 0.0
    for w in READ_W:
        eta = bch.solve(sand, w, probe * PROBE_AMP)
        smax = max(smax, bch.stress(eta).max())
        feats += [hN @ eta, hS @ eta]
    f = np.array(feats)
    return np.r_[f.real, f.imag], smax / SIGC


def lda_loo(X, y, groups, rng):
    """Ridge LDA, leave one group (bed) out.  X already noisy per read."""
    acc = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Z = (X - mu) / sd
        m0, m1 = Z[tr & (y == 0)].mean(0), Z[tr & (y == 1)].mean(0)
        R = Z[tr] - np.where((y[tr] == 1)[:, None], m1, m0)
        S = R.T @ R / max(1, tr.sum() - 2)
        S += 0.01 * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        wv = np.linalg.solve(S, m1 - m0)
        th = wv @ (m0 + m1) / 2
        pred = (Z[te] @ wv > th).astype(int)
        acc += list(pred == y[te])
    return float(np.mean(acc))


def main():
    global PROBE_AMP
    # probe amplitude: max probe stress <= 0.5 sigma_c on the calibration bed, with a x4 margin
    smax = max(bch.stress(bch.solve(s0, w, probe)).max() for w in READ_W)
    PROBE_AMP = np.sqrt(0.5 * SIGC / smax) / 2.0
    rng = np.random.default_rng(12345)
    out = dict(sigma_c=SIGC, ampB=float(ampB), probe_amp=float(PROBE_AMP), cells={})
    worst_probe = 0.0
    for writer in ("scour", "exner"):
        for s in STRENGTHS:
            for loop in ("closed", "open"):
                if loop == "open" and s != 0.04:
                    continue
                rate = rate_for(writer, s, SIGC)
                feats, sands, y, groups = [], [], [], []
                for seed in SEEDS:
                    for lab, order in ((0, "AB"), (1, "BA")):
                        sd = history(seed, order, writer, rate, SIGC, openloop=(loop == "open"))
                        f, pm = read(sd)
                        worst_probe = max(worst_probe, pm)
                        feats.append(f); sands.append(sd); y.append(lab); groups.append(seed)
                F, Sd, y, groups = map(np.array, (feats, sands, y, groups))
                rms = np.sqrt(np.mean(F ** 2))
                diff = float(np.mean([np.linalg.norm(Sd[2 * i] - Sd[2 * i + 1]) for i in range(len(SEEDS))]))
                cell = dict(sand_AB_BA_mean_norm=diff,
                            sand_upper_bound=lda_loo(Sd, y, groups, rng))
                for nu in NOISES:
                    Xn = F + nu * rms * rng.standard_normal(F.shape)
                    cell[f"read_loo_nu{nu}"] = lda_loo(Xn, y, groups, rng)
                key = f"{writer}_s{s}_{loop}"
                out["cells"][key] = cell
                print(key, cell, flush=True)
    out["worst_probe_stress_over_sigma_c"] = worst_probe
    out["reads_write_nothing"] = worst_probe < 1.0
    c = out["cells"]
    out["P1a_pass"] = all(c[f"{w}_s0.04_closed"]["read_loo_nu0.01"] >= 0.90 for w in ("scour", "exner"))
    out["P1b_pass"] = all(c[f"{w}_s0.04_open"]["read_loo_nu0.01"] <= 0.65 for w in ("scour", "exner"))
    out["P1c_pass"] = all(c[f"{w}_s0.005_closed"]["read_loo_nu0.05"] <= 0.75 for w in ("scour", "exner"))
    json.dump(out, open("results/g1.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k != "cells"})


def slope():
    out = {}
    for writer in ("scour", "exner"):
        diffs = []
        for s in STRENGTHS:
            rate = rate_for(writer, s, 0.0)
            d = [np.linalg.norm(history(seed, "AB", writer, rate, 0.0) - history(seed, "BA", writer, rate, 0.0))
                 for seed in SEEDS[:12]]
            diffs.append(float(np.mean(d)))
        sl = float(np.polyfit(np.log(STRENGTHS), np.log(diffs), 1)[0])
        out[writer] = dict(diffs=diffs, slope=sl, pass_=abs(sl - 2) <= 0.3)
        print(writer, out[writer], flush=True)
    out["P1d_pass"] = all(out[w]["pass_"] for w in ("scour", "exner"))
    json.dump(out, open("results/g1_slope.json", "w"), indent=1)


if __name__ == "__main__":
    {"main": main, "slope": slope}[sys.argv[1] if len(sys.argv) > 1 else "main"]()
