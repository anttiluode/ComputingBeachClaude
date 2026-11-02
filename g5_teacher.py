"""G5: the numerical adjoint vs a physically injected teacher (PREREG_G4G5.md)."""
import json
import numpy as np

from beach.medium import Beach, rough_bed, scour

W, BETA = 0.40, 0.03
DAMPS = [0.0, 0.0005, 0.002, 0.006]
SEEDS = range(500, 508)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


out = {"by_damping": {}}
for damp in DAMPS:
    bch = Beach(damp=damp)
    src, rd = bch.port("west"), bch.port("harbourN")
    rows = []
    for seed in SEEDS:
        sand = rough_bed(bch, seed)
        g, y, eta = bch.grad_sand(sand, W, src, rd, lambda y: y)
        lam = bch.solve(sand, W, rd * np.conj(y))          # exact adjoint field: A lam = c conj(e)
        tau = bch.solve(sand, W, rd * (BETA * y))          # physical teacher swell
        ideal = BETA * np.conj(lam)                        # what the teacher would be if lossless
        ge, gl = np.abs(bch.Gr @ eta), np.abs(bch.Gr @ lam)
        wf = ge * gl
        dphi = np.angle((bch.Gr @ tau) * np.conj(bch.Gr @ ideal))
        phase_err = float(np.sum(wf * np.abs(dphi)) / np.sum(wf))
        nat = scour(bch, bch.stress(eta), 0.0, 1.0)
        on = scour(bch, bch.stress(eta, tau), 0.0, 1.0)   # the sand's single exposure, no subtraction
        J = lambda s: 0.5 * abs(rd @ bch.solve(s, W, src)) ** 2
        step = lambda d: sand + d * (0.005 / np.abs(d).max())
        rows.append(dict(seed=seed, phase_err=phase_err,
                         cos_nat=cos(nat, -g), cos_on=cos(on, -g), cos_diff=cos(on - nat, -g),
                         J0=J(sand), J_nat=J(step(nat)), J_on=J(step(on))))
    med = lambda k: float(np.median([r[k] for r in rows]))
    out["by_damping"][str(damp)] = dict(rows=rows, phase_err=med("phase_err"), cos_nat=med("cos_nat"),
                                        cos_on=med("cos_on"), cos_diff=med("cos_diff"),
                                        on_beats_nat=int(sum(r["cos_on"] > r["cos_nat"] for r in rows)),
                                        on_lowers_J_more=int(sum(r["J_on"] < r["J_nat"] for r in rows)))
    d = out["by_damping"][str(damp)]
    print(damp, {k: v for k, v in d.items() if k != "rows"}, flush=True)

bd = out["by_damping"]
pe = [bd[str(d)]["phase_err"] for d in DAMPS]
out["P5a_pass"] = bool(pe[0] <= 0.01)
out["P5b_pass"] = bool(all(pe[i] < pe[i + 1] for i in range(len(pe) - 1)))
out["P5c_pass"] = bool(bd["0.006"]["cos_on"] - bd["0.006"]["cos_nat"] <= 0.05)
out["P5d_pass"] = bool(bd["0.0"]["cos_diff"] >= 0.95 and bd["0.0"]["on_beats_nat"] >= 6)
json.dump(out, open("results/g5.json", "w"), indent=1)
print({k: v for k, v in out.items() if k.startswith("P5")})
