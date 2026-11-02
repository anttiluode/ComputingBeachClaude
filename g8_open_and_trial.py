"""G8: (a) boundary vs bulk loss at matched Q for the physical teacher, (b) trial-and-keep
learning under this cavity's loss.  PREREG_G8.md."""
import json
import sys
import numpy as np
from scipy.stats import spearmanr

from beach.medium import Beach, rough_bed, DT
from beach.open import OpenBeach, sponge
from beach.sliding import Task, run_arm, cos
from beach.trial import keep_or_revert

SEEDS = range(700, 708)
W, BETA = 0.40, 0.003
TASKS = {"quiet": 1, "loud": -1}


def ringdown_Q(bch, sand, w=0.40):
    g = bch.gamma if isinstance(bch, OpenBeach) else bch.damp
    a, b = 1 + g * DT / 2, 1 - g * DT / 2
    L = bch.laplacian(sand).tocsr()
    src = bch.port("west")
    u, prev, E = np.zeros(bch.n), np.zeros(bch.n), []
    for n in range(1600):
        f = src * np.exp(-((n - 20) / 6) ** 2) * np.cos(w * n * DT)
        prev, u = u, (2 * u - b * prev - DT ** 2 * (L @ u) + DT ** 2 * f) / a
        if n % 25 == 0 and n > 150:
            E.append(0.5 * np.sum(((u - prev) / DT) ** 2) + 0.5 * u @ (L @ u))
    k = np.polyfit(np.arange(len(E)) * 25 * DT, np.log(E), 1)[0]
    return float(-w / k)


def make(cond):
    kind, val = cond
    if kind == "bulk":
        return Beach(damp=val)
    ob = OpenBeach(np.zeros(Beach(0.0).n))
    ob.gamma = sponge(ob, kind, val)
    return ob


CONDS = [("bulk", 0.0006), ("east", 0.05), ("bulk", 0.0016), ("east", 0.3), ("bulk", 0.002),
         ("east", 1.0), ("bulk", 0.006), ("rim", 0.05), ("rim", 0.3)]
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def part_a():
    res = {}
    for cond in CONDS:
        bch = make(cond)
        cs, Qs = [], []
        for seed in SEEDS:
            sand = rough_bed(bch, seed)
            Qs.append(ringdown_Q(bch, sand))
            for sign in TASKS.values():
                t = Task(bch, W, bch.port("west"), bch.port("harbourN"), sign, BETA)
                eta, g, tau, y = t.fields(sand)
                cs.append(cos(-(bch.stress(eta, tau) - bch.stress(eta)), -g))
        key = f"{cond[0]} {cond[1]}"
        res[key] = dict(Q=float(np.median(Qs)), median_cos=float(np.median(cs)),
                        min_cos=float(np.min(cs)), cos=cs, Qs=Qs)
        print("a", key, f"Q {res[key]['Q']:.0f}  cos {res[key]['median_cos']:+.3f}", flush=True)
    keys = [f"{c[0]} {c[1]}" for c in CONDS]
    out = dict(conds=res)
    out["P8a_pass"] = bool(all(res[keys[s]]["median_cos"] <= res[keys[b]]["median_cos"] + 0.05 for b, s in PAIRS))
    out["P8b_pass"] = bool(res["rim 0.3"]["median_cos"] <= 0.3)
    rho = spearmanr([res[k]["Q"] for k in keys], [res[k]["median_cos"] for k in keys]).correlation
    out["spearman_Q_cos"] = float(rho)
    out["P8c_pass"] = bool(rho >= 0.8)
    return out


FAMS = ["white", "smooth", "storm", "exner"]
TRIALS, STEP = 200, 0.005


def part_b():
    bch = Beach(damp=0.006)
    runs = {}
    for tname, sign in TASKS.items():
        beds = []
        for seed in SEEDS:
            sand = rough_bed(bch, seed)
            t = Task(bch, W, bch.port("west"), bch.port("harbourN"), sign, BETA)
            J0 = t.J(sand)
            db = lambda J: float(10 * np.log10(abs(J) / abs(J0)))   # J ∝ |y|^2
            gd = run_arm(t, sand, "gd", TRIALS, STEP)
            gd_best = min(gd["J"])
            row = dict(seed=seed, gd_best_db=db(gd_best), gd_final_db=db(gd["J"][-1]))
            for f in FAMS:
                r = keep_or_revert(t.J, bch, sand, f, TRIALS, STEP, seed)
                row[f + "_db"], row[f + "_kept"] = db(r["J"][-1]), r["kept"]
                row[f + "_curve_db"] = [db(J) for J in r["J"][::10]]
            beds.append(row)
            print("b", tname, seed, " ".join(f"{k}:{v:+.1f}" for k, v in row.items() if k.endswith("_db")), flush=True)
        med = {k: float(np.median([b[k] for b in beds])) for k in beds[0] if k.endswith("_db")}
        runs[tname] = dict(beds=beds, median=med)
        print("b", tname, med, flush=True)
    out = dict(runs=runs)
    q, l = runs["quiet"], runs["loud"]
    # quiet: dB negative is good (J = +|y|^2/2 falls).  loud: J = -|y|^2/2, so J falling means
    # |J| rising: db(J) = 10 log10(|J|/|J0|) > 0 is good.
    good = {"quiet": lambda v: -v, "loud": lambda v: v}
    out["P8d_pass"] = bool(all(sum(good[k](b["storm_db"]) >= 3 for b in runs[k]["beds"]) >= 7 for k in runs))
    out["P8e_pass"] = bool(all(good[k](runs[k]["median"]["storm_db"]) > good[k](runs[k]["median"]["white_db"]) for k in runs))
    out["P8f_pass"] = bool(all(good[k](runs[k]["median"]["exner_db"]) < good[k](runs[k]["median"]["storm_db"]) for k in runs))
    out["P8g_pass"] = bool(l["median"]["storm_db"] < 0.5 * l["median"]["gd_best_db"])
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if "a" in which:
        A = part_a()
        json.dump(A, open("results/g8a.json", "w"), indent=1)
        print({k: v for k, v in A.items() if k != "conds"})
    if "b" in which:
        B = part_b()
        json.dump(B, open("results/g8b.json", "w"), indent=1)
        print({k: v for k, v in B.items() if k != "runs"})
