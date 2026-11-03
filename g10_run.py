"""G10 data: does reading or teaching the beach erase what the storms wrote?  (PREREG_G10.md)

Usage: python g10_run.py <panel a|b|c> <bed seed> <n_cal> <n_test> <out.npz> [K]

  a  hard-threshold bed (G9's), read-amplitude sweep, K repeated physical reads per test storm
  b  sliding-threshold bed (G7's law, fixed rate, ambient sea), same idea, T in {inf, 48}
  c  hard bed, one teaching episode after the storm: none / expose / teach / ideal_full /
     ideal_diff, then one G9-strength read (rho 0.125, writes nothing on the hard bed)

Storm draws use a seed stream disjoint from G9's (G9: 10_000 + 7 bed + ...; here 20_000 + ...).
Panels a and c draw the SAME storms for a given bed (same stream), so c's 'none' arm is a's
first read at rho 0.125.
"""
import sys
import time
import numpy as np

from beach.medium import rough_bed
from beach import watch as W
from beach import destructive as D

RHO_A = [0.125, 0.5, 1.0, 2.0, 4.0, 8.0]
RHO_B = [0.125, 0.5, 2.0]
T_B = [np.inf, 48.0]
SAVE_K = (1, 2, 4, 8)
ARMS_C = ["expose", "teach", "ideal_full", "ideal_diff"]
E_C = 20
BETAS_C = [0.03, 0.3]


def storms(seed, n_cal, n_test, tag):
    rng = np.random.default_rng(20_000 + 7 * seed + tag)
    return [W.draw_storm(rng) for _ in range(n_cal)], [W.draw_storm(rng) for _ in range(n_test)]


def proj(rec, r0):
    return float(rec @ r0 / (r0 @ r0))


def panel_a(seed, n_cal, n_test, K, out):
    sand0 = rough_bed(W.b, seed)
    B0 = D.Bed(sand0)
    cal, test = storms(seed, n_cal, n_test, 1)
    t0 = time.time()
    R = dict(rhos=np.array(RHO_A), K=K, seed=seed)
    # pristine bed read K times at each amplitude (efference baseline, matched pristine state)
    PF = np.zeros((len(RHO_A), K, 16, len(D.PAIRS)), complex)
    PS = np.zeros((len(RHO_A), K + 1, W.b.n), np.float32)
    PW_ = np.zeros((len(RHO_A), K))
    for a, rho in enumerate(RHO_A):
        bed = B0.copy()
        PS[a, 0] = bed.s
        for k in range(K):
            PF[a, k], _, PW_[a, k] = D.read_phys(bed, rho)
            PS[a, k + 1] = bed.s
    R.update(PF=PF, pristine_wrote=PW_)
    # calibration storms: one physical read at each amplitude
    CF = np.zeros((len(RHO_A), n_cal, 16, len(D.PAIRS)), complex)
    CR0 = np.zeros((n_cal, W.b.n), np.float32)
    for i, st in enumerate(cal):
        SB = D.storm(B0, st)
        CR0[i] = SB.s - sand0
        for a, rho in enumerate(RHO_A):
            CF[a, i] = D.read_phys(SB.copy(), rho)[0]
        if i % 50 == 0:
            print("a cal", seed, i, f"{time.time() - t0:.0f}s", flush=True)
    R.update(CF=CF, CR0=CR0, cal_loc=np.array([(s["r"], s["c"]) for s in cal]))
    # test storms: K reads at each amplitude
    TF = np.zeros((len(RHO_A), n_test, K, 16, len(D.PAIRS)), complex)
    TR0 = np.zeros((n_test, W.b.n), np.float32)
    REC = np.zeros((len(RHO_A), n_test, len(SAVE_K), W.b.n), np.float32)
    PROJ = np.zeros((len(RHO_A), n_test, K))
    WROTE = np.zeros((len(RHO_A), n_test, K))
    PSTR = np.zeros((len(RHO_A), n_test, K))
    for i, st in enumerate(test):
        SB = D.storm(B0, st)
        r0 = (SB.s - sand0).astype(float)
        TR0[i] = r0
        for a, rho in enumerate(RHO_A):
            bed = SB.copy()
            for k in range(K):
                TF[a, i, k], PSTR[a, i, k], WROTE[a, i, k] = D.read_phys(bed, rho)
                rec = bed.s - PS[a, k + 1]
                PROJ[a, i, k] = proj(rec, r0)
                if k + 1 in SAVE_K:
                    REC[a, i, SAVE_K.index(k + 1)] = rec
        if i % 10 == 0:
            print("a test", seed, i, f"{time.time() - t0:.0f}s", flush=True)
    R.update(TF=TF, TR0=TR0, REC=REC, PROJ=PROJ, WROTE=WROTE, PSTR=PSTR, save_k=np.array(SAVE_K),
             test_loc=np.array([(s["r"], s["c"]) for s in test]))
    np.savez_compressed(out, **R)
    print("done a", seed, f"{time.time() - t0:.0f}s", flush=True)


def panel_b(seed, n_cal, n_test, K, out):
    sand0 = rough_bed(W.b, seed)
    B0 = D.habituated(sand0)
    rest = D.silent_slot(B0.copy())                      # must be ~0: the bed is at rest
    cal, test = storms(seed, n_cal, n_test, 2)
    t0 = time.time()
    nr, nt = len(RHO_B), len(T_B)
    R = dict(rhos=np.array(RHO_B), Ts=np.array(T_B), K=K, seed=seed, rest_wrote=rest)
    PF = np.zeros((nt, nr, K, 16, len(D.PAIRS)), complex)
    PS = np.zeros((nt, nr, K + 1, W.b.n), np.float32)
    PCF = np.zeros((nt, nr, 16, len(D.PAIRS)), complex)     # pristine, K-1 silent slots, then 1 read
    PCS = np.zeros((nt, W.b.n), np.float32)
    for ti, T in enumerate(T_B):
        for a, rho in enumerate(RHO_B):
            bed = B0.copy()
            PS[ti, a, 0] = bed.s
            for k in range(K):
                PF[ti, a, k] = D.read_phys(bed, rho, "sliding", T)[0]
                PS[ti, a, k + 1] = bed.s
        bed = B0.copy()
        for _ in range(K - 1):
            D.silent_slot(bed, T)
        PCS[ti] = bed.s
        for a, rho in enumerate(RHO_B):
            PCF[ti, a] = D.read_phys(bed.copy(), rho, "sliding", T)[0]
    R.update(PF=PF, PCF=PCF)
    CF = np.zeros((nt, nr, n_cal, 16, len(D.PAIRS)), complex)
    CR0 = np.zeros((nt, n_cal, W.b.n), np.float32)
    for i, st in enumerate(cal):
        for ti, T in enumerate(T_B):
            SB = D.storm(B0, st, "sliding", T)
            CR0[ti, i] = SB.s - sand0
            for a, rho in enumerate(RHO_B):
                CF[ti, a, i] = D.read_phys(SB.copy(), rho, "sliding", T)[0]
        if i % 50 == 0:
            print("b cal", seed, i, f"{time.time() - t0:.0f}s", flush=True)
    R.update(CF=CF, CR0=CR0, cal_loc=np.array([(s["r"], s["c"]) for s in cal]))
    TF = np.zeros((nt, nr, n_test, K, 16, len(D.PAIRS)), complex)
    CTF = np.zeros((nt, nr, n_test, 16, len(D.PAIRS)), complex)  # unpinged control, read at time K
    TR0 = np.zeros((nt, n_test, W.b.n), np.float32)
    PROJ = np.zeros((nt, nr, n_test, K))
    CPROJ = np.zeros((nt, n_test))
    WROTE = np.zeros((nt, nr, n_test, K))
    REC = np.zeros((nt, nr, n_test, len(SAVE_K), W.b.n), np.float32)
    CREC = np.zeros((nt, n_test, W.b.n), np.float32)
    for i, st in enumerate(test):
        for ti, T in enumerate(T_B):
            SB = D.storm(B0, st, "sliding", T)
            r0 = (SB.s - sand0).astype(float)
            TR0[ti, i] = r0
            for a, rho in enumerate(RHO_B):
                bed = SB.copy()
                for k in range(K):
                    TF[ti, a, i, k], _, WROTE[ti, a, i, k] = D.read_phys(bed, rho, "sliding", T)
                    rec = bed.s - PS[ti, a, k + 1]
                    PROJ[ti, a, i, k] = proj(rec, r0)
                    if k + 1 in SAVE_K:
                        REC[ti, a, i, SAVE_K.index(k + 1)] = rec
            bed = SB.copy()
            for _ in range(K - 1):
                D.silent_slot(bed, T)
            CREC[ti, i] = bed.s - PCS[ti]
            CPROJ[ti, i] = proj(CREC[ti, i].astype(float), r0)
            for a, rho in enumerate(RHO_B):
                CTF[ti, a, i] = D.read_phys(bed.copy(), rho, "sliding", T)[0]
        if i % 10 == 0:
            print("b test", seed, i, f"{time.time() - t0:.0f}s", flush=True)
    R.update(TF=TF, CTF=CTF, TR0=TR0, PROJ=PROJ, CPROJ=CPROJ, WROTE=WROTE, REC=REC, CREC=CREC,
             save_k=np.array(SAVE_K), test_loc=np.array([(s["r"], s["c"]) for s in test]))
    np.savez_compressed(out, **R)
    print("done b", seed, f"{time.time() - t0:.0f}s", flush=True)


def panel_c(seed, n_cal, n_test, out):
    """Teaching after a storm.  The pristine bed gets every arm too (matched-history baseline
    for material separation and for the efference readout)."""
    sand0 = rough_bed(W.b, seed)
    B0 = D.Bed(sand0)
    _, test = storms(seed, n_cal, n_test, 1)              # identical to panel a's test storms
    t0 = time.time()
    arms = ["none"] + ARMS_C
    na, nb = len(arms), len(BETAS_C)
    R = dict(seed=seed, E=E_C, betas=np.array(BETAS_C), arms=np.array(arms))
    PF = np.zeros((nb, na, 16, len(D.PAIRS)), complex)
    PS = np.zeros((nb, na, W.b.n))
    PJ = np.zeros((nb, na, E_C + 1))

    def run_arms(bed0, beta):
        """returns {arm: (bed, J traj, applied step L2s)}"""
        tb, Jt, full, diff = D.episode(bed0, "teach", E_C, beta)
        ref = dict(full=full, diff=diff)
        res = {"none": (bed0.copy(), [D.task_J(bed0.s)] * (E_C + 1), [0.0] * E_C), "teach": (tb, Jt, full)}
        for arm in ("expose", "ideal_full", "ideal_diff"):
            bed, J_, st_, _ = D.episode(bed0, arm, E_C, beta, ref=ref)
            res[arm] = (bed, J_, st_)
        return res

    for bi, beta in enumerate(BETAS_C):
        res = run_arms(B0, beta)
        for ai, arm in enumerate(arms):
            bed, J_, _ = res[arm]
            PS[bi, ai], PJ[bi, ai] = bed.s, J_
            PF[bi, ai] = D.read_phys(bed.copy(), D.RHO_G9)[0]
    R.update(PF=PF, PJ=PJ)
    TF = np.zeros((nb, na, n_test, 16, len(D.PAIRS)), complex)
    J = np.zeros((nb, na, n_test, E_C + 1))
    PROJ = np.zeros((nb, na, n_test))
    REC = np.zeros((nb, na, n_test, W.b.n), np.float32)
    STEP = np.zeros((nb, na, n_test, E_C))
    TR0 = np.zeros((n_test, W.b.n), np.float32)
    for i, st in enumerate(test):
        SB = D.storm(B0, st)
        r0 = (SB.s - sand0).astype(float)
        TR0[i] = r0
        for bi, beta in enumerate(BETAS_C):
            res = run_arms(SB, beta)
            for ai, arm in enumerate(arms):
                bed, J_, st_ = res[arm]
                J[bi, ai, i], STEP[bi, ai, i] = J_, st_
                TF[bi, ai, i] = D.read_phys(bed.copy(), D.RHO_G9)[0]
                rec = bed.s - PS[bi, ai]
                REC[bi, ai, i] = rec
                PROJ[bi, ai, i] = proj(rec, r0)
        if i % 10 == 0:
            print("c test", seed, i, f"{time.time() - t0:.0f}s", flush=True)
    R.update(TF=TF, J=J, PROJ=PROJ, REC=REC, STEP=STEP, TR0=TR0,
             test_loc=np.array([(s["r"], s["c"]) for s in test]))
    np.savez_compressed(out, **R)
    print("done c", seed, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    panel, seed, n_cal, n_test, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    K = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    if panel == "a":
        panel_a(seed, n_cal, n_test, K, out)
    elif panel == "b":
        panel_b(seed, n_cal, n_test, K, out)
    else:
        panel_c(seed, n_cal, n_test, out)
