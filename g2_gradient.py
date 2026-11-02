"""G2: is the sand computing a gradient?  (PREREG.md, P2a-P2e)"""
import json
import numpy as np

from beach.medium import Beach, rough_bed, scour, exner

W = 0.40
BETA = 0.03
QS = [0.0, 0.25, 0.5, 0.75, 0.9]
SEEDS = range(100, 108)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run(damp):
    bch = Beach(damp=damp)
    src, rd = bch.port("west"), bch.port("harbourN")
    rows = []
    for seed in SEEDS:
        sand = rough_bed(bch, seed)
        g, y, eta = bch.grad_sand(sand, W, src, rd, lambda y: y)   # J = 1/2 |y|^2, e = y
        tau = bch.solve(sand, W, rd * (BETA * y))                   # physical teacher swell
        s_fwd = bch.stress(eta)
        s_p, s_m = bch.stress(eta, tau), bch.stress(eta, -tau)
        for q in QS:
            sc = np.quantile(s_fwd, q) if q > 0 else 0.0
            m = (s_fwd > sc).astype(float)
            pred = cos(-g, -g * m)
            for name, wr in (("scour", scour), ("exner", exner)):
                d2 = 0.5 * (wr(bch, s_p, sc, 1.0) - wr(bch, s_m, sc, 1.0))
                d1 = wr(bch, s_p, sc, 1.0) - wr(bch, s_fwd, sc, 1.0)
                rows.append(dict(seed=seed, q=q, writer=name,
                                 cos2=cos(d2, -g), cos1=cos(d1, -g), P=pred,
                                 descent2=float(d2 @ -g), n_active=int(m.sum())))
    return rows


def med(rows, **kw):
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    return sel


out = {}
lossless = run(0.0)
damped = run(0.006)
out["rows_lossless"], out["rows_damped"] = lossless, damped

c = [r["cos2"] for r in med(lossless, q=0.0, writer="scour")]
out["P2a_cos_median"] = float(np.median(c)); out["P2a_pass"] = out["P2a_cos_median"] >= 0.99
c = [r["cos2"] for r in med(damped, q=0.0, writer="scour")]
out["P2b_cos_median"] = float(np.median(c)); out["P2b_pass"] = out["P2b_cos_median"] >= 0.95
p2c = {}
for q in QS:
    s = med(damped, q=q, writer="scour")
    p2c[str(q)] = dict(cos=float(np.median([r["cos2"] for r in s])), P=float(np.median([r["P"] for r in s])))
out["P2c_by_q"] = p2c
out["P2c_pass"] = all(abs(v["cos"] - v["P"]) <= 0.05 for v in p2c.values())
s = med(damped, q=0.0, writer="exner")
out["P2d_descent_all"] = all(r["descent2"] > 0 for r in s)
out["P2d_cos_median"] = float(np.median([r["cos2"] for r in s]))
out["P2d_pass"] = out["P2d_descent_all"] and 0 < out["P2d_cos_median"] < 0.6
s = med(damped, q=0.0, writer="scour")
out["P2e_cos1_median"] = float(np.median([r["cos1"] for r in s]))
out["P2e_pass"] = abs(out["P2e_cos1_median"] - out["P2b_cos_median"]) <= 0.05
# extra, not pre-registered: lossless exner and lossless threshold sweep
out["extra_lossless_exner_cos_median_q0"] = float(np.median([r["cos2"] for r in med(lossless, q=0.0, writer="exner")]))
out["extra_lossless_P2c"] = {str(q): dict(cos=float(np.median([r["cos2"] for r in med(lossless, q=q, writer="scour")])),
                                         P=float(np.median([r["P"] for r in med(lossless, q=q, writer="scour")]))) for q in QS}
json.dump(out, open("results/g2.json", "w"), indent=1)
for k, v in out.items():
    if not k.startswith("rows"):
        print(k, v)
