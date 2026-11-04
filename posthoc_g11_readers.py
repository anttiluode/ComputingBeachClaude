"""POST HOC (after P11e failed): why does the G11 read decode AB/BA across beds at 0.91 when
G1 (0.59) and G4 (0.75) could not?  Two candidate reasons, separated on the same held-out
reads (beds 1300-1347, low band, 1 % noise, G1's ridge LDA, leave one bed out):
  listeners: all 5 ports (12 pairs) vs the two harbours only (G1/G4's listeners, 6 pairs)
  beds:      how many training beds the reader gets (15 = the dev set size, 31, 47)
"""
import json
import numpy as np

from beach import dark as D

d = np.load("results/g11/across_1300.npz")
n = len(d["seed"])
Y = np.concatenate([d["low_yAB"], d["low_yBA"]]).reshape(2 * n, 16, len(D.PAIRS))
lab = np.r_[np.zeros(n), np.ones(n)].astype(int)
bed = np.r_[np.arange(n), np.arange(n)]
harb = [k for k, (i, j) in enumerate(D.PAIRS) if D.PORTS[j] in ("harbourN", "harbourS")]
sets = {"all5_listeners": list(range(len(D.PAIRS))), "harbours_only": harb,
        "probe_south_harbours (G1's read)": [k for k in harb if D.PROBES[D.PAIRS[k][0]] == "south"]}


def feats(cols, rng):
    Z = Y[:, :, cols].reshape(2 * n, -1)
    X = np.c_[Z.real, Z.imag]
    return X + 0.01 * np.sqrt(np.mean(X ** 2)) * rng.standard_normal(X.shape)


def loo_k(X, k, rng, reps=3):
    """Leave one bed out, but train on only k of the other beds (random, repeated)."""
    acc = []
    for _ in range(reps):
        for t in range(n):
            others = rng.permutation([b for b in range(n) if b != t])[:k]
            keep = np.isin(bed, np.r_[others, t])
            acc.append(_one(X, keep, t))
    return float(np.mean(acc))


def _one(X, keep, t):
    tr, te = keep & (bed != t), bed == t
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Z = (X - mu) / sd
    m0, m1 = Z[tr & (lab == 0)].mean(0), Z[tr & (lab == 1)].mean(0)
    R = Z[tr] - np.where((lab[tr] == 1)[:, None], m1, m0)
    S = R.T @ R / max(1, tr.sum() - 2)
    S += 0.01 * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    w = np.linalg.solve(S, m1 - m0)
    return float(np.mean((Z[te] @ w > w @ (m0 + m1) / 2).astype(int) == lab[te]))


rng = np.random.default_rng(2026)
out = {"note": __doc__.strip().splitlines()[0]}
for name, cols in sets.items():
    X = feats(cols, rng)
    out[name] = {"n_features": X.shape[1],
                 **{f"train_beds_{k}": loo_k(X, k, rng, reps=1 if k == n - 1 else 3) for k in (15, 31, n - 1)}}
    print(name, out[name], flush=True)
json.dump(out, open("results/g11/posthoc_readers.json", "w"), indent=1)
