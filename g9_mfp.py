"""G9a, the strongest ordinary-weight competitor: a digital watcher that ALSO has a model of the
(frozen) basin and localizes by Bartlett matched-field processing.  No training storms.

The model is the bed the digital system believes in: exact, the true bed plus smooth error of
size eps (sand units; the bed's own roughness sd is 0.06), a flat bed at the mean depth, or a
different random bed with the same statistics.

Usage: python g9_mfp.py <data.npz> -> JSON on stdout (test storms = last 20 %)
"""
import json
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

from beach.medium import rough_bed, N
from beach import watch as W
import g9_analyse as A

KS = (2, 5, 20, 64)
EPS = (0.01, 0.03, 0.06, 0.1)


def models(seed):
    b = W.b
    s0 = rough_bed(b, seed)
    g = gaussian_filter(np.random.default_rng(7777 + seed).standard_normal((N, N)), 3.0)
    g = g[b.cells[:, 0], b.cells[:, 1]]
    pert = (g - g.mean()) / g.std()
    out = {"exact": s0}
    for e in EPS:
        out[f"eps{e}"] = s0 + e * pert
    out["flat"] = np.full(b.n, 0.3)
    out["other_bed"] = rough_bed(b, seed + 500)
    return out


SHAPE = {"flat_wrongshape": 1.0}     # flat bed AND the storm assumed point-like (sigma 1 vs 2.5)


def main(path):
    d = A.load(path)
    seed = int(d["seed"])
    S = len(d["loc"])
    te = np.arange(S - S // 5, S)
    rng = np.random.default_rng(seed * 31 + 5)
    hd = d["heard_dense"][te]
    rms = np.sqrt(np.mean(np.abs(d["heard_dense"]) ** 2))
    hd = hd + A.NU * rms * (rng.standard_normal(hd.shape) + 1j * rng.standard_normal(hd.shape)) / np.sqrt(2)
    Y = d["loc"].astype(float)[te]
    F = d["f"][te]
    res = {"seed": seed, "ntones": int(d["ntones"]), "n_test": int(len(te)), "mfp": {}}
    ms = {k: (v, W.STORM_SIGMA) for k, v in models(seed).items()}
    for k, sg in SHAPE.items():
        ms[k] = (np.full(W.b.n, 0.3), sg)
    for name, (sm, sg) in ms.items():
        R = W.mfp_replicas(sm, W.DENSE, sg)
        res["mfp"][name] = {f"k{k}": A.err(np.array([W.mfp_locate(R, hd[i], F[i], k)
                                                     for i in range(len(te))]), Y) for k in KS}
        print(name, res["mfp"][name], file=sys.stderr, flush=True)
    return res


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1))
