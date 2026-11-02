"""POST HOC (not pre-registered), written after G2 failed P2b and P2c.

(1) Damping sweep: how fast does the teacher-sand / gradient alignment fall with loss?
    Also records the alternative teacher amplitude conj(e), the physical adjoint field of
    the lossy medium itself.
(2) P2c redone with a teacher that is weak relative to the threshold (beta = 1e-4).
"""
import json
import numpy as np

from beach.medium import Beach, rough_bed, scour

W = 0.40
SEEDS = range(100, 108)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


out = {"damping": {}, "weak_teacher_P2c": {}}
for damp in [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.01, 0.02]:
    bch = Beach(damp=damp)
    src, rd = bch.port("west"), bch.port("harbourN")
    ce, cc = [], []
    for seed in SEEDS:
        sand = rough_bed(bch, seed)
        g, y, eta = bch.grad_sand(sand, W, src, rd, lambda y: y)
        for amp, lst in ((y, ce), (np.conj(y), cc)):
            tau = bch.solve(sand, W, rd * 0.03 * amp)
            d = 0.5 * (scour(bch, bch.stress(eta, tau), 0, 1) - scour(bch, bch.stress(eta, -tau), 0, 1))
            lst.append(cos(d, -g))
    # quality factor proxy: omega / damping
    out["damping"][str(damp)] = dict(cos_teacher_e=float(np.median(ce)),
                                     cos_teacher_conj_e=float(np.median(cc)),
                                     Q=(W / damp if damp else None))
    print(damp, out["damping"][str(damp)], flush=True)

bch = Beach(damp=0.006)
src, rd = bch.port("west"), bch.port("harbourN")
for damp_name, b in (("lossless", Beach(damp=0.0)), ("damped", bch)):
    res = {}
    for q in [0.0, 0.25, 0.5, 0.75, 0.9]:
        cs, ps = [], []
        for seed in SEEDS:
            sand = rough_bed(b, seed)
            g, y, eta = b.grad_sand(sand, W, src, rd, lambda y: y)
            tau = b.solve(sand, W, rd * 1e-4 * y)
            sf = b.stress(eta)
            sc = np.quantile(sf, q) if q > 0 else 0.0
            d = 0.5 * (scour(b, b.stress(eta, tau), sc, 1) - scour(b, b.stress(eta, -tau), sc, 1))
            m = (sf > sc).astype(float)
            cs.append(cos(d, -g))
            # prediction relative to what the threshold leaves of the *achievable* direction
            ps.append(cos(-g, -g * m))
        res[str(q)] = dict(cos=float(np.median(cs)), P=float(np.median(ps)))
    out["weak_teacher_P2c"][damp_name] = res
    print(damp_name, res, flush=True)
json.dump(out, open("results/posthoc_g2.json", "w"), indent=1)
