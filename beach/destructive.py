"""G10 instruments: reads that touch the sand, a sliding-threshold beach, and teaching episodes
on a storm-written bed.  G9's code (beach/watch.py, g9_*.py) is imported, never edited.

Physical read.  G9's read5() solves the probe fields and leaves the bed alone.  Here every
probe exposure also passes through a writer, so a read can change what it reads:

  for each of the 16 read frequencies (one "block"):
      solve the 3 probe fields (west, north, south) on the CURRENT bed,
      record the 5-port responses (divided by the amplitude factor, so features are in
      G9's units at every amplitude),
      then apply 3 write substeps, one per probe, with the fields of that block
      (quasi-static within a block, the same approximation G9's storms make per substep).

  So one read = 48 probe substeps at a FIXED physical rate (the storm's scour rate RATE),
  never rescaled to a unit step.

Amplitude.  rho = peak probe stress / sigma_c on the calibration bed (seed 0), maximised over
the 16 frequencies, per probe port.  G9's PAMP is rho = 0.125 (its rule "<= 0.5 sigma_c, x4
margin" halves the AMPLITUDE, so stress is 0.5/4).  Amplitude factor k(rho) = sqrt(rho/0.125).

Two media.
  hard     G9's bed: scour d = -r max(sigma - sigma_c, 0).  No ambient sea.  Between
           exposures nothing moves, so an unpinged bed at a later time IS the unpinged bed.
  sliding  G7's law with a fixed physical rate:  d = -r (sigma - theta),
           dtheta = (sigma - theta) / T  (T in substeps, inf = theta never moves).
           An always-on ambient sea (west port, omega 0.20, outside the read band, peak
           stress = sigma_c on the calibration bed) is part of the medium; theta starts
           habituated to it (theta0 = sigma_amb(bed)), so the pristine bed is at rest under
           the ambient sea alone, exactly as G7's bed is at rest under its forward swell.
           Time = substeps of exposure; a "silent" slot is 48 ambient-only substeps, the
           same length as one read.
"""
import numpy as np
import scipy.sparse.linalg as spla

from beach.medium import scour, apply
from beach import watch as W

b = W.b
SIGC = W.SIGC
RATE = W.RATE
READ_W = W.READ_W
PROBES = W.PROBES
PAIRS = W.PAIRS
PMAT = W.PMAT
PW = W.PW
PAMP = W.PAMP
RHO_G9 = 0.125

# ambient sea for the sliding medium
W_AMB = 0.20
_eamb0 = b.solve(W.G1.s0, W_AMB, PW["west"])
A_AMB = float(np.sqrt(SIGC / b.stress(_eamb0).max()))
SRC_AMB = PW["west"] * A_AMB


def kfac(rho):
    return float(np.sqrt(rho / RHO_G9))


def _lu(sand, w):
    return spla.splu(b.operator(sand, w))


def amb_stress(sand):
    return b.stress(_lu(sand, W_AMB).solve(SRC_AMB.astype(complex)))


class Bed:
    """A bed state: sand, and (sliding medium only) the threshold field theta."""

    def __init__(self, sand, theta=None):
        self.s = sand.copy()
        self.th = None if theta is None else theta.copy()

    def copy(self):
        return Bed(self.s, self.th)


def habituated(sand):
    return Bed(sand, amb_stress(sand))


def _write(bed, sig, medium, T, rate):
    """One substep.  Returns max |d sand| actually applied."""
    if medium == "hard":
        d = scour(b, sig, SIGC, rate)
    else:
        d = -rate * (sig - bed.th)
        if np.isfinite(T):
            bed.th = bed.th + (sig - bed.th) / T
    new = apply(bed.s, d)
    m = float(np.abs(new - bed.s).max())
    bed.s = new
    return m


def read_phys(bed, rho, medium="hard", T=np.inf, rate=RATE):
    """One physical read.  Mutates bed.  Returns (out5 complex (16, 12) in G9 units,
    max probe stress / sigma_c, max |d sand| written during the read)."""
    k = kfac(rho)
    out5 = np.zeros((16, len(PAIRS)), complex)
    rhs = np.stack([PW[p] * PAMP[p] * k for p in PROBES], 1).astype(complex)
    smax, wrote = 0.0, 0.0
    for fi, w in enumerate(READ_W):
        eta = _lu(bed.s, w).solve(rhs)
        R = PMAT @ eta / k
        out5[fi] = [R[j, i] for i, j in PAIRS]
        sp = [b.stress(eta[:, q]) for q in range(3)]
        smax = max(smax, max(float(x.max()) for x in sp))
        samb = amb_stress(bed.s) if medium == "sliding" else 0.0
        for q in range(3):
            wrote = max(wrote, _write(bed, samb + sp[q], medium, T, rate))
    return out5, smax / SIGC, wrote


def silent_slot(bed, T=np.inf, rate=RATE):
    """Sliding medium only: 48 ambient-only substeps (same length as one read)."""
    wrote = 0.0
    for _ in range(16):
        samb = amb_stress(bed.s)
        for _ in range(3):
            wrote = max(wrote, _write(bed, samb, "sliding", T, rate))
    return wrote


def storm(bed0, st, medium="hard", T=np.inf, rate=RATE):
    """Write one storm into a COPY of bed0.  hard: identical to G9's run_storm.
    sliding: same forcing and amplitude rule, sliding writer, ambient sea on."""
    if medium == "hard":
        s, _ = W.run_storm(bed0.s, st, rate)
        return Bed(s)
    src = W.storm_src(st["r"], st["c"])
    ws = [READ_W[f] for f in st["f"]]
    amps = [np.sqrt(st["m"] * W.STA_MAX / len(ws) / b.stress(b.solve(bed0.s, w, src)).max()) for w in ws]
    bed = bed0.copy()
    for _ in range(W.NW):
        sig = amb_stress(bed.s)
        for w, a in zip(ws, amps):
            sig = sig + b.stress(_lu(bed.s, w).solve((src * a).astype(complex)))
        _write(bed, sig, "sliding", T, rate)
    return bed


# ------------------------------------------------------------------ teaching (panel c)
W_TEACH = 0.40                        # G1's swell A frequency, G5-G8's task frequency
SRC_F = PW["west"]                    # forward swell, amplitude 1 = G1/G5 (peak ~10 sigma_c)
RD = PW["harbourN"]                   # task readout


def task_J(sand):
    return float(0.5 * abs(RD @ _lu(sand, W_TEACH).solve(SRC_F.astype(complex))) ** 2)


def episode(bed0, arm, E, beta, rate=RATE, ref=None, amp=1.0):
    """One teaching episode on the hard medium.  arms:
       expose      forward swell only, scour writer                 (matched exposure, no target)
       teach       forward swell + physical teacher from harbourN,  tau = A^-1 (rd * beta * y),
                   scour writer, one exposure, no subtraction  (G5's 'natural' with a threshold)
       ideal_full  exact task gradient, step L2 = teach's applied step L2 (ref['full'][t])
       ideal_diff  exact task gradient, step L2 = |d_teach - d_expose| on teach's own state
                   (ref['diff'][t]): the size of the target-specific part of teach's write
    Returns bed, J trajectory, per-step applied L2 and (teach) the diff norms."""
    bed = bed0.copy()
    src = SRC_F * amp
    J = [task_J(bed.s)]
    full, diff = [], []
    for t in range(E):
        if arm in ("expose", "teach"):
            lu = _lu(bed.s, W_TEACH)
            eta = lu.solve(src.astype(complex))
            de = scour(b, b.stress(eta), SIGC, rate)
            if arm == "teach":
                y = RD @ eta
                tau = lu.solve((RD * (beta * y)).astype(complex))
                d = scour(b, b.stress(eta, tau), SIGC, rate)
                diff.append(float(np.linalg.norm(d - de)))
            else:
                d = de
        else:
            g, _, _ = b.grad_sand(bed.s, W_TEACH, src, RD, lambda y: y)
            size = ref["full" if arm == "ideal_full" else "diff"][t]
            ng = np.linalg.norm(g)
            d = -g * (size / ng) if ng > 0 else 0 * g
        new = apply(bed.s, d)
        full.append(float(np.linalg.norm(new - bed.s)))
        bed.s = new
        J.append(task_J(bed.s))
    return bed, J, full, diff
