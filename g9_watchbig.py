"""G9a, the watcher's data budget: how many training storms do ordinary weights need to match
the beach?  Idealised watcher (storm field on the unwritten bed, one solve per storm), single
tone, same storm distribution as g9_storms.py.

Usage:
  python g9_watchbig.py gen <bed seed> <n storms> <out.npz>
  python g9_watchbig.py fit <data.npz>          # learning curve, prints JSON
"""
import json
import sys
import time
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beach.medium import rough_bed
from beach import watch as W
import g9_analyse as A

NTEST = 1000
CURVE = [400, 1600, 6400, 16000]
KRR_CAP = 4000


def gen(seed, n, out):
    rng = np.random.default_rng(50_000 + seed)
    sand0 = rough_bed(W.b, seed)
    H, L, Fq = [], [], []
    t0 = time.time()
    for i in range(n):
        st = W.draw_storm(rng, 1)
        H.append(W.heard_initial(sand0, st)); L.append((st["r"], st["c"])); Fq.append(st["f"])
        if i % 2000 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, heard=np.array(H), loc=np.array(L), f=np.array(Fq), seed=seed)


def fit_curve(path):
    d = dict(np.load(path))
    rng = np.random.default_rng(int(d["seed"]) + 77)
    heard = d["heard"]
    rmsh = np.sqrt(np.mean(np.abs(heard) ** 2))
    heard = heard + A.NU * rmsh * (rng.standard_normal(heard.shape) + 1j * rng.standard_normal(heard.shape)) / np.sqrt(2)
    Y = d["loc"].astype(float)
    S = len(Y)
    te = np.arange(S - NTEST, S)
    res = {"seed": int(d["seed"]), "curve": {}}
    for k in (5, 2):
        Xb = np.array([W.watch_features(W.watch_memory(h, f, k)) for h, f in zip(heard, d["f"])])
        Xp = np.array([W.watch_features_pooled(W.watch_memory(h, f, k)) for h, f in zip(heard, d["f"])])
        for n in [c for c in CURVE if c <= S - NTEST]:
            tr = np.arange(n)
            row = {"CHANCE": A.err(np.tile(Y[tr].mean(0), (NTEST, 1)), Y[te])}
            for tag, X in (("binned", Xb), ("pooled", Xp)):
                sub = tr if n <= KRR_CAP else tr       # grid search below handles the cap
                name, _, est, _ = A.fit(X[sub], Y[sub], krr=(n <= KRR_CAP))
                row[tag] = A.err(est.predict(X[te]), Y[te]); row[tag + "_decoder"] = name
                mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256), early_stopping=True,
                                    max_iter=500, random_state=0)).fit(X[tr], Y[tr])
                row[tag + "_mlp"] = A.err(mlp.predict(X[te]), Y[te])
            res["curve"][f"WATCH{k}_n{n}"] = row
            print(k, n, row, file=sys.stderr, flush=True)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        print(json.dumps(fit_curve(sys.argv[2]), indent=1))
