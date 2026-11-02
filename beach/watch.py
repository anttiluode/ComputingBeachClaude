"""G9 instruments: storms that hit the beach while nobody watches, the beach read afterwards,
and the ordinary-weight competitor that *was* watching.

A storm is a localized steady forcing (Gaussian blob, sigma 2.5 cells) at a random interior
point, at one or more of the 16 read frequencies, with a random strength.  It writes the bed
with G1/G4's strongest scour rule (10 feedback substeps, s = 0.04), starting from a fresh copy
of the bed every time.

Three ways to remember it:

  SAND     the bed itself (what the medium stored; an upper bound nobody can read directly)
  BEACH-k  weak pings after the storm, minus the same pings before it (the device knows its own
           baseline).  BEACH-5: probes at west/north/south, heard at all 5 ports, reciprocal
           duplicates removed (12 complex pairs x 16 freqs = 384 reals).  BEACH-2: G1's read,
           probe south, heard at the two harbours (64 reals).  No sensor is on during the storm.
  WATCH-k  ordinary weights: k sensors are on during the storm and each frequency bin keeps a
           Hebbian outer-product memory C_f += x x^H of what they heard (x = complex amplitude
           at the k ports, averaged over the storm).  Stored per bin: log trace + C / trace
           (k^2 reals), 16 bins.  WATCH-5 uses the same 5 ports as BEACH-5.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

import g1_order as G1
from beach.medium import N, SIGMA, PORTS, WRITERS, apply

b = G1.bch
SIGC = G1.SIGC
RATE = G1.rate_for("scour", 0.04, SIGC)
STA_MAX = float(G1.stA.max())
READ_W = G1.READ_W                      # 16 frequencies in [0.30, 0.55]
NW = 10
STORM_SIGMA = 2.5
PORT_NAMES = ["west", "north", "south", "harbourN", "harbourS"]
PROBES = ["west", "north", "south"]
PW = {p: b.port(p) for p in PORT_NAMES}
PMAT = np.stack([PW[p] for p in PORT_NAMES], 0)          # (5, n) receivers

# probe amplitudes, G1's rule per probe port: max stress <= 0.5 sigma_c on seed 0, x4 margin
PAMP = {}
for p in PROBES:
    sm = max(b.stress(b.solve(G1.s0, w, PW[p])).max() for w in READ_W)
    PAMP[p] = float(np.sqrt(0.5 * SIGC / sm) / 2.0)

# unique (probe, receiver) pairs under reciprocity
PAIRS = []
for i, p in enumerate(PROBES):
    for j, q in enumerate(PORT_NAMES):
        if q in PROBES and PROBES.index(q) < i:
            continue
        PAIRS.append((i, j))

# interior cells at least 5 cells from the wall
_dist = distance_transform_edt(b.inside)
INTERIOR = np.flatnonzero(_dist[b.cells[:, 0], b.cells[:, 1]] >= 5.0)


def storm_src(r0, c0, sigma=STORM_SIGMA):
    rr, cc = b.cells[:, 0], b.cells[:, 1]
    return np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * sigma ** 2))


def draw_storm(rng, ntones=1, fset=None):
    """Location (cell row, col), frequency indices, strength factor m (stress ratio).
    fset: the frequency indices storms may use (default all 16)."""
    k = INTERIOR[rng.integers(len(INTERIOR))]
    r0, c0 = (float(v) for v in b.cells[k])
    pool = np.arange(16) if fset is None else np.asarray(fset)
    fidx = rng.choice(pool, size=ntones, replace=False)
    m = float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
    return dict(r=r0, c=c0, f=[int(f) for f in fidx], m=m)


def run_storm(sand0, st, rate=None):
    """Write the storm into a copy of sand0 with feedback.  Returns new sand and the complex
    port amplitudes the watchers heard per tone (mean over the 10 substeps)."""
    src = storm_src(st["r"], st["c"])
    ws = [READ_W[f] for f in st["f"]]
    # amplitude per tone: total max stress on the initial bed = m * A's max stress,
    # split equally over tones
    amps = []
    for w in ws:
        smax = b.stress(b.solve(sand0, w, src)).max()
        amps.append(np.sqrt(st["m"] * STA_MAX / len(ws) / smax))
    sand = sand0.copy()
    heard = np.zeros((len(ws), len(PORT_NAMES)), complex)
    for _ in range(NW):
        stress = np.zeros(b.n)
        for t, (w, a) in enumerate(zip(ws, amps)):
            eta = b.solve(sand, w, src * a)
            stress += b.stress(eta)            # different frequencies: no time-averaged cross term
            heard[t] += PMAT @ eta / NW
        sand = apply(sand, WRITERS["scour"](b, stress, SIGC, RATE if rate is None else rate))
    return sand, heard


def read5(sand):
    """BEACH-5 and BEACH-2 complex reads, and the max probe stress / sigma_c."""
    out5 = np.zeros((16, len(PAIRS)), complex)
    out2 = np.zeros((16, 2), complex)
    smax = 0.0
    rhs = np.stack([PW[p] * PAMP[p] for p in PROBES], 1)
    for fi, w in enumerate(READ_W):
        eta = b.solve(sand, w, rhs)                      # (n, 3)
        for k in range(3):
            smax = max(smax, float(b.stress(eta[:, k]).max()))
        R = PMAT @ eta                                   # (5 receivers, 3 probes)
        out5[fi] = [R[j, i] for i, j in PAIRS]
        out2[fi] = R[3:5, 2]                             # probe south -> harbourN, harbourS
    return out5, out2, smax / SIGC


def watch_memory(tones_heard, fidx, k):
    """Ordinary-weight memory: per bin, C += x x^H over the first k ports (port order
    harbourN, harbourS, south, west, north so k = 1, 2 are the harbours)."""
    order = [3, 4, 2, 0, 1][:k]
    C = np.zeros((16, k, k), complex)
    for x, f in zip(tones_heard, fidx):
        v = x[order]
        C[f] += np.outer(v, v.conj())
    return C


def watch_features(C):
    k = C.shape[1]
    iu = np.triu_indices(k, 1)
    out = []
    for f in range(16):
        tr = np.real(np.trace(C[f]))
        if tr <= 0:
            out.append(np.zeros(2 + k + 2 * len(iu[0])))
            continue
        M = C[f] / tr
        out.append(np.r_[1.0, np.log(tr), np.real(np.diag(M)), M[iu].real, M[iu].imag])
    return np.concatenate(out)


def cplx(z):
    z = np.asarray(z).ravel()
    return np.r_[z.real, z.imag]


def watch_features_pooled(C):
    """Single-tone storms only: the same memory without frequency bins, so a decoder can
    share what it learns across frequencies: [omega, log trace, C / trace]."""
    k = C.shape[1]
    iu = np.triu_indices(k, 1)
    tr = np.real(np.einsum("fii->f", C))
    f = int(np.argmax(tr))
    M = C[f] / tr[f]
    return np.r_[READ_W[f], np.log(tr[f]), np.real(np.diag(M)), M[iu].real, M[iu].imag]


def heard_initial(sand0, st):
    """Idealised watcher for the big training sets: the storm's field on the unwritten bed
    (one solve per tone, no drift while the bed is being dug)."""
    src = storm_src(st["r"], st["c"])
    out = []
    for f in st["f"]:
        w = READ_W[f]
        eta = b.solve(sand0, w, src)
        a = np.sqrt(st["m"] * STA_MAX / len(st["f"]) / b.stress(eta).max())
        out.append(PMAT @ eta * a)
    return np.array(out)


# Dense watchers: the 5 ports first, then 59 fixed random interior cells (same on every bed).
_rs = np.random.default_rng(4242)
DENSE_IDX = _rs.choice(INTERIOR, size=59, replace=False)
DENSE = np.vstack([PMAT, np.stack([np.exp(-((b.cells[:, 0] - b.cells[i, 0]) ** 2 +
                                           (b.cells[:, 1] - b.cells[i, 1]) ** 2) / (2 * SIGMA ** 2))
                                   for i in DENSE_IDX], 0)])          # (64, n)


def heard_dense(sand0, st):
    """Frozen basin: complex amplitude at all 64 dense sensors, per tone."""
    src = storm_src(st["r"], st["c"])
    out = []
    for f in st["f"]:
        eta = b.solve(sand0, READ_W[f], src)
        a = np.sqrt(st["m"] * STA_MAX / len(st["f"]) / b.stress(eta).max())
        out.append(DENSE @ eta * a)
    return np.array(out)


def dense_features(tones_heard, fidx, k, pooled):
    """k sensors, rank-1 memory per bin stored as the phase-referenced vector
    v = x conj(x_0)/|x_0| (x x^H carries exactly this).  pooled: no bins, [omega, Re v, Im v]."""
    if pooled:
        x = tones_heard[0][:k]
        v = x * np.conj(x[0]) / (abs(x[0]) + 1e-30)
        return np.r_[READ_W[fidx[0]], v.real, v.imag]
    out = np.zeros((16, 2 * k))
    for x, f in zip(tones_heard, fidx):
        x = x[:k]
        v = x * np.conj(x[0]) / (abs(x[0]) + 1e-30)
        out[f] += np.r_[v.real, v.imag]
    return out.ravel()


def mfp_replicas(sand0, sensors, sigma=STORM_SIGMA):
    """Matched-field replicas for a digital watcher that has a PERFECT model of the frozen
    basin: R[f][x, s] = response at sensor s to a unit storm at interior cell x, frequency f.
    By reciprocity (A symmetric) it is the sensor's own field blurred by the storm shape."""
    rr, cc = b.cells[:, 0], b.cells[:, 1]
    srcs = np.stack([storm_src(rr[i], cc[i], sigma) for i in INTERIOR], 1)  # (n, nx)
    out = []
    for w in READ_W:
        eta = b.solve(sand0, w, sensors.T)                                  # (n, nsens)
        out.append(srcs.T @ eta)                                            # (nx, nsens)
    return np.array(out)


def mfp_locate(R, tones_heard, fidx, k):
    """Bartlett matched-field processor on the first k sensors, summed over tones."""
    score = 0.0
    for x, f in zip(tones_heard, fidx):
        x = x[:k]
        Rf = R[f][:, :k]
        num = np.abs(Rf.conj() @ x) ** 2
        score = score + num / (np.sum(np.abs(Rf) ** 2, 1) * np.sum(np.abs(x) ** 2) + 1e-300)
    i = INTERIOR[int(np.argmax(score))]
    return np.array([b.cells[i, 0], b.cells[i, 1]], float)
