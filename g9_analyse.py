"""G9a analysis: where did the storm hit?  Each memory gets the same decoder search.

Usage: python g9_analyse.py <out.json> <data.npz>

Decoder (fixed in PREREG_G9.md): standardise features, then pick ONE of
  ridge (alpha in 1e-3..1e3), kNN (k in 3, 5, 10), RBF kernel ridge (gamma in {0.3,1,3}/d,
  alpha in {0.01, 0.1, 1})
by 5-fold CV mean squared error on the training storms only.  Score = median Euclidean
location error (cells) on test storms.  Chance = median error of predicting the training mean.
"""
import json
import sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold

from beach import watch as W

HO = (1, 5, 9, 13)
DENSE_K = (10, 20, 40, 64)
ARMS = ["SAND", "BEACH5", "BEACH2", "WATCH5", "WATCH2", "WATCH1", "WATCH5d", "BEACH5_f8", "BEACH5_fs"] + \
    [f"DENSE{k}" for k in DENSE_K]
NU = 0.01


def features(d, rng):
    """All memories for every storm.  Noise: nu x RMS on every sensor reading."""
    r5, r2 = d["r5"], d["r2"]
    out = {}
    rms5 = np.sqrt(np.mean(np.abs(d["r5_0"]) ** 2))
    rms2 = np.sqrt(np.mean(np.abs(d["r2_0"]) ** 2))
    n5 = NU * rms5 * np.sqrt(2) * (rng.standard_normal(r5.shape) + 1j * rng.standard_normal(r5.shape)) / np.sqrt(2)
    n2 = NU * rms2 * np.sqrt(2) * (rng.standard_normal(r2.shape) + 1j * rng.standard_normal(r2.shape)) / np.sqrt(2)
    out["BEACH5"] = np.array([W.cplx(z) for z in r5 + n5])
    z5 = r5 + n5
    if int(d["ntones"]) == 1:
        # diagnostics: read at ONE frequency, either the storm's own or a fixed one (index 8)
        out["BEACH5_fs"] = np.array([np.r_[W.READ_W[f[0]], W.cplx(z[f[0]])] for z, f in zip(z5, d["f"])])
    out["BEACH5_f8"] = np.array([W.cplx(z[8]) for z in z5])
    out["BEACH2"] = np.array([W.cplx(z) for z in r2 + n2])
    out["SAND"] = d["dsand"].astype(float)
    for tag, key in (("", "heard0"), ("d", "heard")):
        # "" = frozen basin (ordinary weights replace the self-modifying bed: primary competitor)
        # "d" = hybrid: sensors listen while the bed is being dug
        heard = d[key]                                   # (S, tones, 5)
        rmsh = np.sqrt(np.mean(np.abs(heard) ** 2))
        heard = heard + NU * rmsh * (rng.standard_normal(heard.shape) + 1j * rng.standard_normal(heard.shape)) / np.sqrt(2)
        for k in (1, 2, 5):
            if tag == "d" and k != 5:
                continue
            reps = {"binned": np.array([W.watch_features(W.watch_memory(h, f, k)) for h, f in zip(heard, d["f"])])}
            if int(d["ntones"]) == 1:
                reps["pooled"] = np.array([W.watch_features_pooled(W.watch_memory(h, f, k))
                                           for h, f in zip(heard, d["f"])])
            out[f"WATCH{k}{tag}"] = reps
    hd = d["heard_dense"]
    rmsd = np.sqrt(np.mean(np.abs(hd) ** 2))
    hd = hd + NU * rmsd * (rng.standard_normal(hd.shape) + 1j * rng.standard_normal(hd.shape)) / np.sqrt(2)
    for k in DENSE_K:
        reps = {"binned": np.array([W.dense_features(h, f, k, False) for h, f in zip(hd, d["f"])])}
        if int(d["ntones"]) == 1:
            reps["pooled"] = np.array([W.dense_features(h, f, k, True) for h, f in zip(hd, d["f"])])
        out[f"DENSE{k}"] = reps
    return out


def fit(X, Y, seed=0, krr=True):
    cv = KFold(5, shuffle=True, random_state=seed)
    d = X.shape[1]
    fams = {
        "ridge": (Ridge(), {"ridge__alpha": list(np.logspace(-3, 3, 7))}),
        "knn": (KNeighborsRegressor(), {"kneighborsregressor__n_neighbors": [3, 5, 10]}),
        "krr": (KernelRidge(kernel="rbf"), {"kernelridge__gamma": [0.3 / d, 1.0 / d, 3.0 / d],
                                            "kernelridge__alpha": [0.01, 0.1, 1.0]}),
    }
    if not krr:
        fams.pop("krr")
    best = None
    for name, (est, grid) in fams.items():
        gs = GridSearchCV(make_pipeline(StandardScaler(), est), grid, cv=cv,
                          scoring="neg_mean_squared_error").fit(X, Y)
        if best is None or gs.best_score_ > best[1]:
            best = (name, gs.best_score_, gs.best_estimator_, gs.best_params_)
    return best


def fit_any(Xs, Y, idx, krr=True):
    """Xs: one feature matrix, or a dict of alternative representations (watchers): the
    representation and decoder are both chosen by inner CV on the training storms."""
    if not isinstance(Xs, dict):
        Xs = {"": Xs}
    best = None
    for rep, X in Xs.items():
        name, sc, est, _ = fit(X[idx], Y[idx], krr=krr)
        if best is None or sc > best[1]:
            best = (f"{rep}/{name}" if rep else name, sc, est, rep)
    return best


def take(Xs, rep, idx):
    return (Xs[rep] if isinstance(Xs, dict) else Xs)[idx]


def err(P, Y):
    return float(np.median(np.hypot(*(P - Y).T)))


def load(path):
    d = dict(np.load(path))
    if "heard_dense" not in d:                           # dev files: frozen basin, 64 sensors
        from beach.medium import rough_bed
        s0 = rough_bed(W.b, int(d["seed"]))
        d["heard_dense"] = np.array([W.heard_dense(s0, dict(r=l[0], c=l[1], f=list(f), m=m))
                                     for l, f, m in zip(d["loc"], d["f"], d["m"])])
    d["heard0"] = d["heard_dense"][..., :5]              # DENSE rows 0-4 are the 5 ports
    return d


def analyse(path, ntrain_list=None, arms=None, do_ho=True):
    d = load(path)
    nfreq = int(d.get("nfreq", 16))
    rng = np.random.default_rng(int(d["seed"]) * 13 + int(d["ntones"]))
    F = features(d, rng)
    Y = d["loc"].astype(float)
    S = len(Y)
    nte = S // 5
    tr_all, te = np.arange(S - nte), np.arange(S - nte, S)
    if ntrain_list is None:
        ntrain_list = [n for n in (100, 200, 400, 800) if n <= len(tr_all)]
    arms = arms or [a for a in ARMS if a in F]
    res = {"seed": int(d["seed"]), "ntones": int(d["ntones"]), "n_storms": S,
           "worst_probe_stress_over_sigma_c": float(d["probe"].max()), "curve": {}, "ho": {}}
    for n in ntrain_list:
        tr = tr_all[:n]
        row = {"CHANCE": err(np.tile(Y[tr].mean(0), (len(te), 1)), Y[te])}
        for a in arms:
            name, _, est, rep = fit_any(F[a], Y, tr)
            row[a] = err(est.predict(take(F[a], rep, te)), Y[te])
            row[a + "_decoder"] = name
        res["curve"][n] = row
        print(n, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, file=sys.stderr, flush=True)
    res["nfreq"] = nfreq
    if do_ho and int(d["ntones"]) == 1 and nfreq == 16:
        f = d["f"][:, 0]
        trm = np.isin(f, HO, invert=True)
        tr = tr_all[trm[tr_all]]
        te_in = te[trm[te]]
        te_ho = np.flatnonzero(~trm)
        row = {"n_train": int(len(tr)), "n_test_in": int(len(te_in)), "n_test_ho": int(len(te_ho)),
               "CHANCE_in": err(np.tile(Y[tr].mean(0), (len(te_in), 1)), Y[te_in]),
               "CHANCE_ho": err(np.tile(Y[tr].mean(0), (len(te_ho), 1)), Y[te_ho])}
        for a in arms:
            name, _, est, rep = fit_any(F[a], Y, tr)
            row[a + "_in"] = err(est.predict(take(F[a], rep, te_in)), Y[te_in])
            row[a + "_ho"] = err(est.predict(take(F[a], rep, te_ho)), Y[te_ho])
            row[a + "_decoder"] = name
        res["ho"] = row
        print("HO", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, file=sys.stderr, flush=True)
    return res


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    json.dump(analyse(sys.argv[2]), open(sys.argv[1], "w"), indent=1)
