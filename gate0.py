"""Gate 0: the instrument is right before anything is claimed.

G0a  flat bed (sand 0) reproduces WallMemoryLaw's stadium cavity (no written walls).
G0b  the frequency-domain solve equals the time stepper's steady state.
G0c  the adjoint gradient equals finite differences.
G0d  the time-averaged stress of PHYSICALLY superposed swells (time domain) contains
     the cross term the frequency-domain formula predicts.
Writes results/gate0.json.
"""
import json
import sys
import numpy as np

from beach.medium import Beach, rough_bed, DT, SIGMA

W = 0.40          # swell angular frequency per time unit (wavelength ~10 cells at depth 1)
out = {}
bch = Beach()
sand = rough_bed(bch, seed=1)

# ---------------------------------------------------------------- G0a
wml = sys.argv[1] if len(sys.argv) > 1 else None
if wml:
    sys.path.insert(0, wml)
    from wml.cavity import Cavity
    cav = Cavity(K=24, fraction=0.6, shape="stadium")
    ref = cav.run(np.zeros((1, 24)), source=0, steps=300)[0, :, 0]
    sy, sx = cav.sources[0]
    rr, cc = bch.cells[:, 0], bch.cells[:, 1]
    pulse = np.exp(-((cc - sx) ** 2 + (rr - sy) ** 2) / (2 * SIGMA ** 2))
    rec = bch.idx[cav.receiver]
    tr = []
    bch.step_time(np.zeros(bch.n), [], 300, u0=pulse, record=lambda n, u: tr.append(u[rec]))
    d = float(np.max(np.abs(np.array(tr) - ref)))
    out["G0a_max_abs_diff_vs_WallMemoryLaw"] = d
    out["G0a_pass"] = d < 1e-12
    print("G0a", d)

# ---------------------------------------------------------------- G0b
src = bch.port("west")
eta = bch.solve(sand, W, src)
steps = 9000
u = bch.step_time(sand, [(src, 1.0, W)], steps)
pred = np.real(eta * np.exp(-1j * W * steps * DT))
raw = float(np.linalg.norm(u - pred) / np.linalg.norm(pred))
# The Neumann cavity has a zero-frequency uniform mode (mean water level).  The start-up
# kick leaves a constant offset there that never decays (u_tt + g u_t = 0).  It has no
# gradient, so it carries no stress.  Compare with the uniform offset removed.
off = np.mean(u - pred)
rel = float(np.linalg.norm(u - off - pred) / np.linalg.norm(pred))
out["G0b_rel_err_raw_incl_zero_mode"] = raw
out["G0b_zero_mode_offset"] = float(off)
out["G0b_rel_err_steady_state"] = rel
out["G0b_pass"] = rel < 1e-3
print("G0b", rel)

# ---------------------------------------------------------------- G0c
rd = bch.port("harbourN")
ystar = 0.0
loss_e = lambda y: y - ystar                    # J = 1/2 |y - y*|^2
J = lambda s: 0.5 * abs(rd @ bch.solve(s, W, src) - ystar) ** 2
gs, y, _ = bch.grad_sand(sand, W, src, rd, loss_e)
rng = np.random.default_rng(0)
errs = []
for k in rng.choice(bch.n, 12, replace=False):
    h = 1e-6
    sp_, sm_ = sand.copy(), sand.copy()
    sp_[k] += h
    sm_[k] -= h
    fd = (J(sp_) - J(sm_)) / (2 * h)
    errs.append((fd, gs[k]))
errs = np.array(errs)
rel = float(np.max(np.abs(errs[:, 0] - errs[:, 1])) / np.max(np.abs(errs[:, 0])))
out["G0c_fd_vs_adjoint_max_rel"] = rel
out["G0c_pass"] = rel < 1e-4
print("G0c", rel)

# ---------------------------------------------------------------- G0d
# forward swell + teacher swell injected at the harbour with complex amplitude Et.
Et = 0.3 * np.exp(0.7j)
tau = bch.solve(sand, W, rd * Et)
pred_cross = bch.stress(eta, tau) - bch.stress(eta) - bch.stress(tau)
T0, periods = 9000, 20
nT = int(round(2 * np.pi / (W * DT)))          # steps per period (approx)
Tavg = periods * nT


def avg_stress(drives):
    acc = np.zeros(bch.nf)
    cnt = [0]

    def rec(n, uu):
        if n > T0:
            acc[:] += (bch.Gr @ uu) ** 2
            cnt[0] += 1
    bch.step_time(sand, drives, T0 + Tavg, record=rec)
    sf = acc / cnt[0]
    return (bch.Fc @ sf) / bch.deg


s_both = avg_stress([(src, 1.0, W), (rd, Et, W)])
s_fwd = avg_stress([(src, 1.0, W)])
s_tch = avg_stress([(rd, Et, W)])
meas_cross = s_both - s_fwd - s_tch
cos = float(meas_cross @ pred_cross / np.linalg.norm(meas_cross) / np.linalg.norm(pred_cross))
rel = float(np.linalg.norm(meas_cross - pred_cross) / np.linalg.norm(pred_cross))
out["G0d_cos_time_avg_cross_vs_formula"] = cos
out["G0d_rel_err"] = rel
out["G0d_pass"] = cos > 0.999
print("G0d", cos, rel, "(period not an exact integer number of steps: residual from averaging window)")

json.dump(out, open("results/gate0.json", "w"), indent=1)
print(json.dumps(out, indent=1))
