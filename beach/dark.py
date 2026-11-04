"""G11 instruments: the beach's INDEX, measured as sensitivity kernels.

A weak ping from probe port p, heard at port q, at frequency w, is y = r_q^T A(w, sand)^-1 s_p.
To first order in a sand change ds (Born), dy = K_pqw . ds with the kernel

    K_pqw(i) = (g/2) * sum_{faces f touching cell i} (Gr lam_q)_f (Gr eta_p)_f,
    eta_p = A^-1 s_p  (the ping's field),   lam_q = A^-1 r_q  (the listener's field; A is
    complex symmetric, so the adjoint field is the listener's own field).

Stacking Re and Im of every (p, q, w) gives the Jacobian J (features x cells).  The span of
J's rows is everything a weak ping from these ports can ever see.  Sand outside it is stored
but unaddressable ("dark sand").

Reads are G9's BEACH-5 family (probes west/north/south, heard at all five ports, reciprocal
duplicates removed: 12 complex pairs per frequency), in two bands:
  LOW  = G1/G9's 16 frequencies 0.30-0.55
  HIGH = 16 frequencies 0.60-0.95 (shorter waves, finer kernels)
Probe amplitude per port and band: G1's rule (max probe stress <= 0.5 sigma_c on seed 0,
x4 margin, max over the band's frequencies).
"""
import numpy as np

import g1_order as G1
from beach.medium import G as GRAV, WRITERS, apply, rough_bed
from beach import watch as W

b = G1.bch
SIGC = G1.SIGC
BANDS = {"low": np.linspace(0.30, 0.55, 16), "high": np.linspace(0.60, 0.95, 16)}
PROBES, PORTS, PAIRS = W.PROBES, W.PORT_NAMES, W.PAIRS
PW = W.PW
RMAT = W.PMAT                      # (5, n) receivers

PAMP = {}
for _band, _ws in BANDS.items():
    for p in PROBES:
        sm = max(b.stress(b.solve(G1.s0, w, PW[p])).max() for w in _ws)
        PAMP[(_band, p)] = float(np.sqrt(0.5 * SIGC / sm) / 2.0)


def read(sand, band):
    """Complex read vector (16 freqs x 12 pairs) and max probe stress / sigma_c."""
    ws = BANDS[band]
    out = np.zeros((len(ws), len(PAIRS)), complex)
    smax = 0.0
    rhs = np.stack([PW[p] * PAMP[(band, p)] for p in PROBES], 1)
    for fi, w in enumerate(ws):
        eta = b.solve(sand, w, rhs)
        for k in range(len(PROBES)):
            smax = max(smax, float(b.stress(eta[:, k]).max()))
        R = RMAT @ eta
        out[fi] = [R[j, i] for i, j in PAIRS]
    return out.ravel(), smax / SIGC


def real(z):
    return np.r_[z.real, z.imag]


def jacobian(sand, band):
    """Complex Jacobian dy/dsand, (16*12, n), same row order as read()."""
    ws = BANDS[band]
    rows = []
    for w in ws:
        import scipy.sparse.linalg as spla
        lu = spla.splu(b.operator(sand, w))
        eta = lu.solve(np.stack([PW[p] * PAMP[(band, p)] for p in PROBES], 1).astype(complex))
        lam = lu.solve(RMAT.T.astype(complex))                       # (n, 5)
        ge, gl = b.Gr @ eta, b.Gr @ lam
        for i, j in PAIRS:
            rows.append(0.5 * GRAV * (b.Fc @ (gl[:, j] * ge[:, i])))
    return np.array(rows)


# ------------------------------------------------------------------ histories
def burst(sand, w, src, rate, nw=G1.NW):
    for _ in range(nw):
        st = b.stress(b.solve(sand, w, src))
        sand = apply(sand, WRITERS["scour"](b, st, SIGC, rate))
    return sand


RATE = G1.rate_for("scour", 0.04, SIGC)          # G1/G4's strongest scour


def history(sand0, order, jit=None, rate=RATE):
    """G4's shared benchmark: A = west swell w 0.40, B = north swell w 0.47 at amplitude ampB.
    jit = (dwA, dwB, mA, mB) jitters frequencies and amplitudes (within-bed decoding)."""
    dwA, dwB, mA, mB = (0.0, 0.0, 1.0, 1.0) if jit is None else jit
    sand = sand0.copy()
    for k in order:
        if k == "A":
            sand = burst(sand, G1.WA + dwA, G1.srcA * mA, rate)
        else:
            sand = burst(sand, G1.WB + dwB, G1.srcB * G1.ampB * mB, rate)
    return sand


def storm_write(sand0, rng):
    """One G9 storm (single tone, random interior location): the content that reads well."""
    st = W.draw_storm(rng, 1)
    sand, _ = W.run_storm(sand0, st)
    return sand - sand0, st


# ------------------------------------------------------------------ measures
def gain(J, x):
    return float(np.linalg.norm(J @ x) / (np.linalg.norm(x) + 1e-300))


def radial_k(v):
    """Power-weighted mean spatial wavenumber (cycles/cell) of a cell field, zero outside."""
    g = b.to_grid(v - v.mean(), fill=0.0)
    P = np.abs(np.fft.fft2(g)) ** 2
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(g.shape[0]), np.fft.fftfreq(g.shape[1]), indexing="ij"))
    P[0, 0] = 0
    return float((P * k).sum() / P.sum())


def coherence(V):
    """||mean v||^2 / mean ||v||^2 over rows: 1 = every bed points the same way, ~1/n = random."""
    return float(np.linalg.norm(V.mean(0)) ** 2 / np.mean(np.sum(V ** 2, 1)))
