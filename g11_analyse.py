"""G11 analysis.  Usage: python g11_analyse.py <across.npz> [<within.npz> ...] [--json out.json]"""
import sys
import json
import numpy as np

import g1_order as G1
from beach import dark as D

NU = 0.01


def R(z):
    z = np.asarray(z)
    return np.concatenate([z.real, z.imag], axis=-1)


def across(path):
    d = np.load(path)
    out = {"n_beds": int(len(d["seed"])), "worst_probe_over_sigma_c": float(d["probe_max"].max())}
    Dl = d["hBA"] - d["hAB"]
    c = 0.5 * (d["hAB"] + d["hBA"]) - d["h0"]
    out["sand_norm"] = {"order": float(np.median(np.linalg.norm(Dl, axis=1))),
                        "common": float(np.median(np.linalg.norm(c, axis=1))),
                        "storm": float(np.median(np.linalg.norm(d["dstorm"], axis=1)))}
    out["radial_k"] = {"order": float(np.median([D.radial_k(v) for v in Dl])),
                       "common": float(np.median([D.radial_k(v) for v in c])),
                       "storm": float(np.median([D.radial_k(v) for v in d["dstorm"]])),
                       "roughness": float(np.median([D.radial_k(v) for v in d["h0"]]))}
    out["sand_coherence_order"] = D.coherence(Dl)
    rng = np.random.default_rng(11)
    for band in ("low", "high"):
        o = {}
        gD = np.linalg.norm(d[f"{band}_JD"], axis=1) / np.linalg.norm(Dl, axis=1)
        gc = np.linalg.norm(d[f"{band}_Jc"], axis=1) / np.linalg.norm(c, axis=1)
        gr = np.linalg.norm(d[f"{band}_Jr"], axis=1) / np.linalg.norm(d["h0"] - d["h0"].mean(1, keepdims=True), axis=1)
        gs = np.linalg.norm(d[f"{band}_Js"], axis=1) / np.linalg.norm(d["dstorm"], axis=1)
        o["gain_ratio_order_over_common"] = float(np.median(gD / gc))
        o["gain_ratio_storm_over_common"] = float(np.median(gs / gc))
        o["gain_ratio_roughness_over_common"] = float(np.median(gr / gc))
        dy = d[f"{band}_yBA"] - d[f"{band}_yAB"]
        o["born_rel_err_order"] = float(np.median(np.linalg.norm(d[f"{band}_JD"] - dy, axis=1) / np.linalg.norm(dy, axis=1)))
        dys = d[f"{band}_ystorm"] - d[f"{band}_y0"]
        o["born_rel_err_storm"] = float(np.median(np.linalg.norm(d[f"{band}_Js"] - dys, axis=1) / np.linalg.norm(dys, axis=1)))
        Yall = R(np.concatenate([d[f"{band}_yAB"], d[f"{band}_yBA"]]))
        rms = np.sqrt(np.mean(Yall ** 2))
        nf = Yall.shape[1]
        # SNR of the order difference against the noise of two independent 1 % reads
        o["order_snr_median"] = float(np.median(np.linalg.norm(R(dy), axis=1) / (NU * rms * np.sqrt(2 * nf))))
        o["storm_snr_median"] = float(np.median(np.linalg.norm(R(dys), axis=1) / (NU * rms * np.sqrt(2 * nf))))
        # does the order point the same way in every bed's read?  (standardised features)
        sd = Yall.std(0) + 1e-30
        o["read_coherence_order"] = D.coherence(R(dy) / sd)
        o["coherence_null_random"] = 1.0 / len(dy)
        if f"{band}_T" in d.files:
            T = R(d[f"{band}_T"])                                # (i, j, feat)
            n = T.shape[0]
            cs = [abs(T[i, i] @ T[i, j]) / (np.linalg.norm(T[i, i]) * np.linalg.norm(T[i, j]))
                  for i in range(n) for j in range(n) if i != j]
            o["transplant_cos_median"] = float(np.median(cs))
            # same index, another bed's order pattern (how different are the patterns themselves)
            cs2 = [abs(T[i, i] @ T[j, i]) / (np.linalg.norm(T[i, i]) * np.linalg.norm(T[j, i]))
                   for i in range(n) for j in range(n) if i != j]
            o["same_index_other_order_cos_median"] = float(np.median(cs2))
        sv = d[f"{band}_sv"]
        o["index_rank_1pct"] = float(np.median((sv > 1e-2 * sv[:, :1]).sum(1)))
        o["index_rank_noise"] = float(np.median((sv > 0.05 * sv[:, :1]).sum(1)))
        out[band] = o
    # across-bed decoding, G1/G4's LOO-bed ridge LDA, 1 % read noise
    dec = {}
    y = np.r_[np.zeros(len(Dl)), np.ones(len(Dl))].astype(int)
    groups = np.r_[d["seed"], d["seed"]]
    feats = {}
    for band in ("low", "high"):
        X = R(np.concatenate([d[f"{band}_yAB"], d[f"{band}_yBA"]]))
        feats[band] = X + NU * np.sqrt(np.mean(X ** 2)) * rng.standard_normal(X.shape)
    feats["both"] = np.hstack([feats["low"], feats["high"]])
    feats["sand"] = np.concatenate([d["hAB"], d["hBA"]])
    if len(Dl) >= 4:
        for k, X in feats.items():
            dec[k] = G1.lda_loo(X, y, groups, rng)
    out["across_bed_decoding"] = dec
    return out


def within(path):
    d = np.load(path)
    rng = np.random.default_rng(13)
    X = R(d["y"])
    X = X + NU * np.sqrt(np.mean(X ** 2)) * rng.standard_normal(X.shape)
    return {"seed": int(d["seed"]), "n_pairs": int(d["pair"].max() + 1),
            "worst_probe_over_sigma_c": float(d["probe_max"].max()),
            "read": G1.lda_loo(X, d["lab"], d["pair"], rng),
            "sand": G1.lda_loo(d["sand"], d["lab"], d["pair"], rng)}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = {"across": across(args[0]), "within": [within(p) for p in args[1:] if p != sys.argv[-1] or "--json" not in sys.argv]}
    if "--json" in sys.argv:
        res["within"] = [within(p) for p in args[1:-1]]
        json.dump(res, open(args[-1], "w"), indent=1)
    print(json.dumps(res, indent=1))
