"""G9a data: storms hit fresh copies of one bed; record what each memory keeps.

Usage: python g9_storms.py <bed seed> <n storms> <tones> <out.npz> [nfreq]
nfreq: storms use only the first nfreq of the frequencies (8, 0, 4, 12, ...) (default 16 = all)
s: write strength, max |d sand| of A's first write step on seed 0 (default 0.04, G1/G4's strongest)
"""
import sys
import time
import numpy as np

from beach.medium import rough_bed
from beach import watch as W

FREQ_ORDER = [8, 0, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]


def main():
    seed, nst, ntones, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    nfreq = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    fset = FREQ_ORDER[:nfreq]
    S = float(sys.argv[6]) if len(sys.argv) > 6 else 0.04
    rate = W.G1.rate_for("scour", S, W.SIGC)
    rng = np.random.default_rng(10_000 + seed * 7 + ntones + 1000 * (nfreq != 16) * nfreq)
    sand0 = rough_bed(W.b, seed)
    r5_0, r2_0, p0 = W.read5(sand0)
    rec = dict(r5=[], r2=[], dsand=[], heard=[], heard_dense=[], loc=[], f=[], m=[], probe=[p0])
    t0 = time.time()
    for i in range(nst):
        st = W.draw_storm(rng, ntones, fset)
        sand, heard = W.run_storm(sand0, st, rate)
        r5, r2, pm = W.read5(sand)
        rec["r5"].append(r5 - r5_0); rec["r2"].append(r2 - r2_0)
        rec["dsand"].append((sand - sand0).astype(np.float32))
        rec["heard"].append(heard); rec["heard_dense"].append(W.heard_dense(sand0, st)); rec["loc"].append((st["r"], st["c"]))
        rec["f"].append(st["f"]); rec["m"].append(st["m"]); rec["probe"].append(pm)
        if i % 50 == 0:
            print(seed, ntones, i, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, r5=np.array(rec["r5"]), r2=np.array(rec["r2"]),
                        dsand=np.array(rec["dsand"]), heard=np.array(rec["heard"]), heard_dense=np.array(rec["heard_dense"]),
                        loc=np.array(rec["loc"]), f=np.array(rec["f"]), m=np.array(rec["m"]),
                        probe=np.array(rec["probe"]), r5_0=r5_0, r2_0=r2_0,
                        seed=seed, ntones=ntones, nfreq=nfreq, s=S)
    print("done", seed, ntones, f"{time.time() - t0:.0f}s", "worst probe", max(rec["probe"]))


if __name__ == "__main__":
    main()
