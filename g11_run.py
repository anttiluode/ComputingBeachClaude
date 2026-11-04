"""G11: dark sand.  Is the AB/BA order outside the beach's index, or inside it but pointing a
different way on every bed?  (PREREG_G11.md)

Usage:
  python g11_run.py check                       instrument checks (Jacobian vs finite differences,
                                                 linear superposition of two pings)
  python g11_run.py across <seed0> <seed1> <out.npz>   AB/BA pairs on beds seed0..seed1-1
  python g11_run.py within <seed> <npairs> <out.npz>   jittered AB/BA pairs on ONE bed
"""
import sys
import time
import numpy as np

from beach import dark as D
from beach.medium import rough_bed

b = D.b


def check():
    rng = np.random.default_rng(5)
    s = rough_bed(b, 0)
    for band in ("low", "high"):
        J = D.jacobian(s, band)
        d = rng.standard_normal(b.n) * 1e-4
        y0, _ = D.read(s, band)
        yp, _ = D.read(s + d, band)
        ym, _ = D.read(s - d, band)
        fd = (yp - ym) / 2
        err = np.linalg.norm(J @ d - fd) / np.linalg.norm(fd)
        print(band, "jacobian vs central difference, rel err", f"{err:.2e}")
    # two pings at once = sum of the pings (the field is linear in the drive)
    w = 0.45
    p, q = b.port("west") * D.PAMP[("low", "west")], b.port("north") * D.PAMP[("low", "north")]
    yp = D.RMAT @ b.solve(s, w, p)
    yq = D.RMAT @ b.solve(s, w, q)
    ypq = D.RMAT @ b.solve(s, w, p + q)
    print("interaction y_pq - y_p - y_q + y_0, relative:", f"{np.linalg.norm(ypq - yp - yq) / np.linalg.norm(ypq):.1e}")


def across(s0, s1, out):
    rec = {k: [] for k in ("seed", "h0", "hAB", "hBA", "dstorm", "storm_loc", "probe_max")}
    for band in ("low", "high"):
        for k in ("y0", "yAB", "yBA", "ystorm", "JD", "Jc", "Jr", "Js", "sv"):
            rec[f"{band}_{k}"] = []
    Jkeep = {"low": [], "high": []}
    Dkeep = []
    t0 = time.time()
    for seed in range(s0, s1):
        rng = np.random.default_rng(90_000 + seed)
        h0 = rough_bed(b, seed)
        hAB, hBA = D.history(h0, "AB"), D.history(h0, "BA")
        ds, st = D.storm_write(h0, rng)
        hmid = 0.5 * (hAB + hBA)
        Dl, c, r = hBA - hAB, hmid - h0, h0 - h0.mean()
        pm = 0.0
        for band in ("low", "high"):
            ys = []
            for sand in (h0, hAB, hBA, h0 + ds):
                y, p = D.read(sand, band)
                pm = max(pm, p)
                ys.append(y)
            for k, y in zip(("y0", "yAB", "yBA", "ystorm"), ys):
                rec[f"{band}_{k}"].append(y)
            J = D.jacobian(hmid, band)
            for k, x in zip(("JD", "Jc", "Jr"), (Dl, c, r)):
                rec[f"{band}_{k}"].append(J @ x)
            rec[f"{band}_Js"].append(D.jacobian(h0, band) @ ds)
            Jr = np.vstack([J.real, J.imag])
            rec[f"{band}_sv"].append(np.linalg.svd(Jr, compute_uv=False))
            Jkeep[band].append(J.astype(np.complex64))
        Dkeep.append(Dl)
        for k, v in zip(("seed", "h0", "hAB", "hBA", "dstorm", "storm_loc", "probe_max"),
                        (seed, h0, hAB, hBA, ds, (st["r"], st["c"]), pm)):
            rec[k].append(v)
        print(seed, f"{time.time() - t0:.0f}s probe {pm:.3f}", flush=True)
    # transplant: bed i's order pattern read through bed j's index.  T[i, j] = J_j D_i
    for band in ("low", "high"):
        rec[f"{band}_T"] = [[Jj @ Di for Jj in Jkeep[band]] for Di in Dkeep]
    np.savez_compressed(out, **{k: np.array(v) for k, v in rec.items()})


def within(seed, npairs, out):
    rng = np.random.default_rng(70_000 + seed)
    h0 = rough_bed(b, seed)
    rec = {"y": [], "lab": [], "pair": [], "sand": [], "probe_max": []}
    t0 = time.time()
    for k in range(npairs):
        jit = (rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015),
               rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1))
        for lab, order in ((0, "AB"), (1, "BA")):
            sd = D.history(h0, order, jit)
            y, p = D.read(sd, "low")
            for kk, v in zip(("y", "lab", "pair", "sand", "probe_max"), (y, lab, k, sd, p)):
                rec[kk].append(v)
        print(seed, k, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, seed=seed, **{k: np.array(v) for k, v in rec.items()})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "across":
        across(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        within(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
