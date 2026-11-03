"""G10 analysis.  Usage: python g10_analyse.py <out.json> <a.npz> <b.npz> <c.npz>

Readout (frozen).  For each amplitude (and medium/T), ONE decoder is chosen and fitted on the
calibration storms' FIRST read (G9's decoder search, g9_analyse.fit), then frozen and applied
to every later read.  Baseline = the pristine bed's first read at the same amplitude (what the
device recorded once, before any storm).  Score: median location error (cells) on test storms.

Noise.  An absolute sensor floor: NU x RMS of the pristine G9-strength read (= G9's 1 % at
rho 0.125).  Features are divided by the amplitude factor, so at amplitude rho the noise in
feature units is that floor x sqrt(0.125 / rho): a louder ping reads cleaner.

Other readouts (diagnostics, same frozen decoder):
  avg    mean of reads 1..k (each with its own noise) minus the frozen baseline
  eff    read k minus the PRISTINE bed's read k after the same read history (an efference
         copy: the device knows what its own reads did to an unwritten bed)
  ctrl   unpinged control at the same final time (panel a: the hard bed does not move between
         exposures, so this is read 1 with fresh noise; panel b: K-1 ambient-only slots, then 1 read)
Material.
  proj   <rec_k, r0> / |r0|^2, rec_k = storm bed - pristine bed after the same history,
         r0 = what the storm wrote.  1 = the storm's distinction is intact in the sand.
  SAND   frozen decoder fitted on r0 of calibration storms, applied to rec_k.
"""
import json
import sys
import warnings
import numpy as np

from g9_analyse import fit, err
from beach import watch as W

warnings.filterwarnings("ignore")
NU = 0.01


def cplx(Z):
    Z = np.asarray(Z)
    Z = Z.reshape(Z.shape[0], -1)
    return np.concatenate([Z.real, Z.imag], 1)


class Noise:
    def __init__(self, floor, seed):
        self.floor, self.rng = floor, np.random.default_rng(seed)

    def __call__(self, Z, rho):
        s = self.floor * np.sqrt(0.125 / rho)
        return Z + s * (self.rng.standard_normal(Z.shape) + 1j * self.rng.standard_normal(Z.shape))


def chance(Yc, Yt):
    return err(np.tile(Yc.mean(0), (len(Yt), 1)), Yt)


def decoder(X, Y):
    name, _, est, _ = fit(X, Y)
    return name, est


def sand_decoder(d, key="CR0"):
    X = d[key].astype(float)
    name, _, est, _ = fit(X, d["cal_loc"].astype(float), krr=False)
    return est


def panel_a(path):
    d = dict(np.load(path))
    rhos, K = list(d["rhos"]), int(d["K"])
    Yc, Yt = d["cal_loc"].astype(float), d["test_loc"].astype(float)
    floor = NU * np.sqrt(np.mean(np.abs(d["PF"][0, 0]) ** 2))
    nz = Noise(floor, int(d["seed"]) * 31 + 1)
    out = dict(seed=int(d["seed"]), n_cal=len(Yc), n_test=len(Yt), chance=chance(Yc, Yt), by_rho={})
    sest = sand_decoder(d)
    save_k = list(d["save_k"])
    for a, rho in enumerate(rhos):
        base = d["PF"][a, 0]
        name, est = decoder(cplx(nz(d["CF"][a], rho) - base), Yc)
        TF = d["TF"][a]                                   # (n_test, K, 16, 12)
        noisy = nz(TF, rho)
        single = [err(est.predict(cplx(noisy[:, k] - base)), Yt) for k in range(K)]
        avg = [err(est.predict(cplx(noisy[:, :k + 1].mean(1) - base)), Yt) for k in range(K)]
        eff = [err(est.predict(cplx(noisy[:, k] - d["PF"][a, k])), Yt) for k in range(K)]
        ctrl = err(est.predict(cplx(nz(TF[:, 0], rho) - base)), Yt)
        sand = {int(k): err(sest.predict(d["REC"][a, :, j].astype(float)), Yt) for j, k in enumerate(save_k)}
        out["by_rho"][str(rho)] = dict(
            decoder=name, single=single, avg=avg, eff=eff, ctrl_final=ctrl, sand=sand,
            proj_median=[float(np.median(d["PROJ"][a, :, k])) for k in range(K)],
            wrote_per_read_median=[float(np.median(d["WROTE"][a, :, k])) for k in range(K)],
            peak_probe_stress=float(d["PSTR"][a].max()),
            pristine_wrote=[float(x) for x in d["pristine_wrote"][a]])
        r = out["by_rho"][str(rho)]
        print(f"a rho {rho:5}: single {r['single'][0]:5.2f} -> {r['single'][-1]:5.2f} | avg8 {r['avg'][-1]:5.2f} | "
              f"eff8 {r['eff'][-1]:5.2f} | ctrl {ctrl:5.2f} | proj8 {r['proj_median'][-1]:.3f} | "
              f"sand1 {sand[1]:5.2f} sand8 {sand[K]:5.2f} | wrote {r['wrote_per_read_median'][0]:.1e} [{name}]", flush=True)
    return out


def panel_b(path):
    d = dict(np.load(path))
    rhos, Ts, K = list(d["rhos"]), list(d["Ts"]), int(d["K"])
    Yc, Yt = d["cal_loc"].astype(float), d["test_loc"].astype(float)
    floor = NU * np.sqrt(np.mean(np.abs(d["PF"][0, 0, 0]) ** 2))
    nz = Noise(floor, int(d["seed"]) * 31 + 2)
    out = dict(seed=int(d["seed"]), n_cal=len(Yc), n_test=len(Yt), chance=chance(Yc, Yt),
               rest_wrote=float(d["rest_wrote"]), by_T={})
    save_k = list(d["save_k"])
    for ti, T in enumerate(Ts):
        X = d["CR0"][ti].astype(float)
        _, _, sest, _ = fit(X, Yc, krr=False)
        sand_ctrl = err(sest.predict(d["CREC"][ti].astype(float)), Yt)
        res = dict(ctrl_proj_median=float(np.median(d["CPROJ"][ti])), sand_ctrl_final=sand_ctrl, by_rho={})
        for a, rho in enumerate(rhos):
            base = d["PF"][ti, a, 0]
            name, est = decoder(cplx(nz(d["CF"][ti, a], rho) - base), Yc)
            noisy = nz(d["TF"][ti, a], rho)
            single = [err(est.predict(cplx(noisy[:, k] - base)), Yt) for k in range(K)]
            avg = [err(est.predict(cplx(noisy[:, :k + 1].mean(1) - base)), Yt) for k in range(K)]
            eff = [err(est.predict(cplx(noisy[:, k] - d["PF"][ti, a, k])), Yt) for k in range(K)]
            ctrl = err(est.predict(cplx(nz(d["CTF"][ti, a], rho) - base)), Yt)
            ctrl_eff = err(est.predict(cplx(nz(d["CTF"][ti, a], rho) - d["PCF"][ti, a])), Yt)
            sand = {int(k): err(sest.predict(d["REC"][ti, a, :, j].astype(float)), Yt) for j, k in enumerate(save_k)}
            res["by_rho"][str(rho)] = dict(
                decoder=name, single=single, avg=avg, eff=eff, ctrl_final=ctrl, ctrl_final_eff=ctrl_eff, sand=sand,
                proj_median=[float(np.median(d["PROJ"][ti, a, :, k])) for k in range(K)],
                wrote_per_read_median=[float(np.median(d["WROTE"][ti, a, :, k])) for k in range(K)])
            r = res["by_rho"][str(rho)]
            print(f"b T {T:5} rho {rho:5}: single {single[0]:5.2f} -> {single[-1]:5.2f} | avg8 {avg[-1]:5.2f} | "
                  f"eff8 {eff[-1]:5.2f} | ctrl {ctrl:5.2f} (eff {ctrl_eff:5.2f}) | proj8 {r['proj_median'][-1]:.3f} "
                  f"(ctrl {res['ctrl_proj_median']:.3f}) | sand8 {sand[K]:5.2f} (ctrl {sand_ctrl:5.2f}) [{name}]", flush=True)
        out["by_T"][str(T)] = res
    return out


def panel_c(path, path_a):
    d = dict(np.load(path))
    da = dict(np.load(path_a))
    Yc, Yt = da["cal_loc"].astype(float), d["test_loc"].astype(float)
    assert np.array_equal(Yt, da["test_loc"].astype(float)), "panel c must use panel a's test storms"
    floor = NU * np.sqrt(np.mean(np.abs(da["PF"][0, 0]) ** 2))
    nz = Noise(floor, int(d["seed"]) * 31 + 3)
    rho = 0.125
    a0 = list(da["rhos"]).index(rho)
    base = da["PF"][a0, 0]
    name, est = decoder(cplx(nz(da["CF"][a0], rho) - base), Yc)
    sest = sand_decoder(da)
    arms, betas = [str(x) for x in d["arms"]], list(d["betas"])
    out = dict(seed=int(d["seed"]), n_test=len(Yt), chance=chance(Yc, Yt), decoder=name, by_beta={})
    for bi, beta in enumerate(betas):
        res = {}
        J0 = d["J"][bi, 0, :, 0]
        for ai, arm in enumerate(arms):
            noisy = nz(d["TF"][bi, ai], rho)
            JE = d["J"][bi, ai, :, -1]
            res[arm] = dict(
                read_frozen=err(est.predict(cplx(noisy - base)), Yt),
                read_eff=err(est.predict(cplx(noisy - d["PF"][bi, ai])), Yt),
                sand=err(sest.predict(d["REC"][bi, ai].astype(float)), Yt),
                proj_median=float(np.median(d["PROJ"][bi, ai])),
                J_ratio_median=float(np.median(JE / J0)),
                pristine_J_ratio=float(d["PJ"][bi, ai, -1] / d["PJ"][bi, ai, 0]),
                write_L2_total_median=float(np.median(d["STEP"][bi, ai].sum(1))))
            r = res[arm]
            print(f"c beta {beta}: {arm:10} read {r['read_frozen']:5.2f} (eff {r['read_eff']:5.2f}) | sand {r['sand']:5.2f} | "
                  f"proj {r['proj_median']:.3f} | J/J0 {r['J_ratio_median']:.3f} (pristine {r['pristine_J_ratio']:.3f}) | "
                  f"L2 {r['write_L2_total_median']:.3f}", flush=True)
        # learning beyond exposure, per storm bed
        Jt, Je = d["J"][bi, arms.index("teach"), :, -1], d["J"][bi, arms.index("expose"), :, -1]
        res["teach_beats_expose_frac"] = float(np.mean(Jt < Je))
        out["by_beta"][str(beta)] = res
    return out


if __name__ == "__main__":
    o, pa, pb, pc = sys.argv[1:5]
    res = dict(a=panel_a(pa), b=panel_b(pb), c=panel_c(pc, pa))
    json.dump(res, open(o, "w"), indent=1)
