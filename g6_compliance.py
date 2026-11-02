"""G6: plain scour vs the gradient of the source's own compliance (PREREG_G6G7.md)."""
import json
import numpy as np

from beach.medium import Beach, rough_bed, scour, apply
from beach.sliding import compliance_grad, cos, normed

DAMPS = [0.0, 0.0005, 0.002, 0.006]
SEEDS = range(600, 608)
PORTS = ["west", "north", "south"]
WS = [0.20, 0.30, 0.42]
QS = [0.0, 0.5]

out = {"by_damping": {}}
for damp in DAMPS:
    bch = Beach(damp=damp)
    rows = []
    for seed in SEEDS:
        sand = rough_bed(bch, seed)
        for p in PORTS:
            src = bch.port(p)
            for w in WS:
                g, Jc, eta = compliance_grad(bch, sand, w, src)
                st = bch.stress(eta)
                for q in QS:
                    sc = float(np.quantile(st, q)) if q > 0 else 0.0
                    d = scour(bch, st, sc, 1.0)
                    s2 = apply(sand, normed(d, 1e-4))
                    Jc2 = float(np.real(src @ bch.solve(s2, w, src)))
                    rows.append(dict(seed=seed, port=p, w=w, q=q, cos=cos(d, -g), Jc=Jc, Jc_step=Jc2,
                                     lowered=bool(Jc2 < Jc)))
    summ = {}
    for q in QS:
        s = [r for r in rows if r["q"] == q]
        summ[str(q)] = dict(median_cos=float(np.median([r["cos"] for r in s])),
                            min_cos=float(np.min([r["cos"] for r in s])),
                            frac_lowered=float(np.mean([r["lowered"] for r in s])), n=len(s))
    out["by_damping"][str(damp)] = dict(rows=rows, summary=summ)
    print(damp, summ, flush=True)

m = [out["by_damping"][str(d)]["summary"]["0.0"]["median_cos"] for d in DAMPS]
out["P6a_pass"] = bool(m[0] >= 0.98)
out["P6b_pass"] = bool(all(m[i] > m[i + 1] for i in range(len(m) - 1)))
out["P6c_pass"] = bool(m[-1] <= 0.3)
out["P6d_pass"] = bool(out["by_damping"]["0.0"]["summary"]["0.0"]["frac_lowered"] >= 0.9)
json.dump(out, open("results/g6.json", "w"), indent=1)
print({k: v for k, v in out.items() if k.startswith("P6")})
