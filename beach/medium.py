"""The beach: a damped linear wave field over a sand bed, plus sand writers.

Geometry and discretisation come from anttiluode/WallMemoryLaw (wml/cavity.py, the
Bunimovich stadium): 96x96 grid, wave speed 0.65 at reference depth, dt 0.5, damping
0.006, Neumann walls, leapfrog stepping

    a u[n+1] = 2 u[n] - b u[n-1] - dt^2 L(D) u[n] + dt^2 f[n],   a,b = 1 +- gamma dt/2

What is new: the floor is sand.  Water depth D = D0 - sand, and every face between two
water cells carries g * D_face, so (L u)_i = sum_j g D_ij (u_i - u_j).  With uniform
depth D0 = 1 and g = 0.65^2 this is WallMemoryLaw's stencil exactly (shallow water:
wave speed^2 = g * depth).  Sand therefore sets the local wave speed: it is the maze.

Steady swells are solved in the frequency domain with the EXACT symbol of the time
stepper.  For a drive f[n] = Re(F exp(-i w n dt)) the steady state is
u[n] = Re(eta exp(-i w n dt)) with

    A(w, D) eta = F,     A = (a z - 2 + b / z) / dt^2 + L(D),     z = exp(-i w dt)

A is complex symmetric (reciprocity), and real when gamma = 0 (lossless).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N = 96
C0 = 0.65
G = C0 ** 2          # so that depth 1 gives WallMemoryLaw's wave speed
DT = 0.5
DAMP = 0.006
SIGMA = 1.55
CENTER = (N - 1) / 2
STADIUM = (18.0, 22.0)
D0 = 1.0
SAND_MIN, SAND_MAX = 0.0, 0.6   # depth stays in [0.4, 1.0]: wave speed varies x0.63..1


def stadium_mask():
    yy, xx = np.mgrid[:N, :N]
    X, Y = xx - CENTER, yy - CENTER
    L, R = STADIUM
    return ((np.abs(X) <= L) & (np.abs(Y) <= R)) | ((np.abs(X) - L) ** 2 + Y ** 2 <= R ** 2)


# Ports (row, col).  READ and the two harbours are where weak pings are sent / heard.
PORTS = {
    "west": (50, 18),      # = WallMemoryLaw stadium source 0
    "north": (30, 48),
    "south": (66, 40),
    "harbourN": (36, 78),
    "harbourS": (60, 78),
}


class Beach:
    def __init__(self, damp=DAMP):
        self.inside = stadium_mask()
        self.idx = -np.ones((N, N), int)
        cells = np.argwhere(self.inside)
        self.cells = cells
        self.n = len(cells)
        self.idx[cells[:, 0], cells[:, 1]] = np.arange(self.n)
        # faces between water cells (each once)
        fi, fj = [], []
        for dr, dc in ((1, 0), (0, 1)):
            r2, c2 = cells[:, 0] + dr, cells[:, 1] + dc
            ok = (r2 < N) & (c2 < N)
            ok[ok] &= self.inside[r2[ok], c2[ok]]
            fi.append(np.flatnonzero(ok))
            fj.append(self.idx[r2[ok], c2[ok]])
        self.fi = np.concatenate(fi)
        self.fj = np.concatenate(fj)
        self.nf = len(self.fi)
        # incidence matrix: (grad u)_f = u_i - u_j
        rows = np.r_[np.arange(self.nf), np.arange(self.nf)]
        cols = np.r_[self.fi, self.fj]
        vals = np.r_[np.ones(self.nf), -np.ones(self.nf)]
        self.Gr = sp.csr_matrix((vals, (rows, cols)), shape=(self.nf, self.n))
        # cell <- face averaging (each cell: mean of its faces)
        deg = np.bincount(np.r_[self.fi, self.fj], minlength=self.n).astype(float)
        self.deg = deg
        self.Fc = sp.csr_matrix((np.ones(2 * self.nf), (np.r_[self.fi, self.fj],
                                 np.r_[np.arange(self.nf), np.arange(self.nf)])),
                                shape=(self.n, self.nf))
        self.damp = damp
        self._lu = {}

    # ------------------------------------------------------------ fields
    def port(self, name, sigma=SIGMA):
        r, c = PORTS[name]
        assert self.inside[r, c], name
        rr, cc = self.cells[:, 0], self.cells[:, 1]
        return np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma ** 2))

    def face_depth(self, sand):
        return D0 - 0.5 * (sand[self.fi] + sand[self.fj])

    def laplacian(self, sand):
        w = G * self.face_depth(sand)
        return (self.Gr.T @ sp.diags(w) @ self.Gr).tocsc()

    def symbol(self, w, damp=None):
        g = self.damp if damp is None else damp
        a, b = 1 + g * DT / 2, 1 - g * DT / 2
        z = np.exp(-1j * w * DT)
        return (a * z - 2 + b / z) / DT ** 2

    def operator(self, sand, w):
        return (self.laplacian(sand) + self.symbol(w) * sp.identity(self.n, format="csc")).tocsc()

    def solve(self, sand, w, rhs):
        """Steady complex field(s) for drive(s) rhs: (n,) or (n, k)."""
        lu = spla.splu(self.operator(sand, w))
        return lu.solve(np.asarray(rhs, complex))

    # ------------------------------------------------------------ stress
    def face_grad(self, eta):
        return self.Gr @ eta

    def stress(self, *fields):
        """Time-averaged bed stress proxy per cell for the SUPERPOSITION of steady
        fields of the same frequency: <(grad u)^2> = 1/2 |grad eta|^2 on each face,
        averaged onto cells.  (Shallow water: bed velocity ~ (g/w) grad eta.)"""
        tot = sum(fields)
        sf = 0.5 * np.abs(self.Gr @ tot) ** 2
        return (self.Fc @ sf) / self.deg

    def to_grid(self, v, fill=np.nan):
        out = np.full((N, N), fill, dtype=np.result_type(v, float))
        out[self.cells[:, 0], self.cells[:, 1]] = v
        return out

    def lap_cells(self, v):
        """Neumann graph Laplacian on cells, unit weights: (Lap v)_i = sum_j (v_j - v_i)."""
        return -(self.Gr.T @ (self.Gr @ v))

    # ------------------------------------------------------------ adjoint
    def grad_sand(self, sand, w, src, readout, loss_e):
        """Exact gradient dJ/dsand for J with dJ = Re(conj(e) dy), y = readout . eta.
        A lam = readout * conj(e);  dJ/dD_f = -g Re(grad lam grad eta);
        D_f = D0 - (s_i + s_j)/2  =>  dJ/ds_i = (g/2) sum_f Re(...)."""
        A = self.operator(sand, w)
        lu = spla.splu(A)
        eta = lu.solve(src.astype(complex))
        y = readout @ eta
        e = loss_e(y)
        lam = lu.solve(readout * np.conj(e))
        gf = -G * np.real((self.Gr @ lam) * (self.Gr @ eta))
        gs = -0.5 * (self.Fc @ gf)
        return gs, y, eta

    # ------------------------------------------------------------ time domain
    def step_time(self, sand, drives, steps, damp=None, record=None, u0=None):
        """Leapfrog on the full grid with face coefficients g*D_face.
        drives: list of (weights over cells, complex amplitude F, w); f[n] = Re(F e^{-iwn dt}).
        record: callable(n, u_cells) or None.  Returns u (cells)."""
        g = self.damp if damp is None else damp
        a, b = 1 + g * DT / 2, 1 - g * DT / 2
        wface = G * self.face_depth(sand)
        u = np.zeros(self.n) if u0 is None else u0.astype(float).copy()
        prev = np.zeros(self.n)
        Lmat = (self.Gr.T @ sp.diags(wface) @ self.Gr).tocsr()
        for nstep in range(steps):
            f = np.zeros(self.n)
            for wts, F, w in drives:
                f += wts * np.real(F * np.exp(-1j * w * nstep * DT))
            nxt = (2 * u - b * prev - DT ** 2 * (Lmat @ u) + DT ** 2 * f) / a
            prev, u = u, nxt
            if record is not None:
                record(nstep + 1, u)
        return u


# ---------------------------------------------------------------- sand writers
def scour(beach, stress, sigma_c, rate):
    """W1, local scour (NOT sand-conserving): stress above the critical value lowers the
    bed in place.  Toy rule, labelled as such."""
    return -rate * np.maximum(stress - sigma_c, 0.0)


def exner(beach, stress, sigma_c, rate):
    """W2, conservative Exner toy: excess stress E = max(stress - sigma_c, 0) is sand
    mobility; flux q = -rate grad E (sand leaves agitated water and settles in calm
    water), d sand = -div q = rate * Lap E.  Total sand is conserved exactly."""
    E = np.maximum(stress - sigma_c, 0.0)
    return rate * beach.lap_cells(E)


WRITERS = {"scour": scour, "exner": exner}


def apply(sand, d):
    return np.clip(sand + d, SAND_MIN, SAND_MAX)


def rough_bed(beach, seed, mean=0.3, amp=0.06, corr=3.0):
    """Initial bed: smooth random roughness (Gaussian-filtered noise), fixed by seed."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((N, N))
    from scipy.ndimage import gaussian_filter
    g = gaussian_filter(g, corr)
    v = g[beach.cells[:, 0], beach.cells[:, 1]]
    v = (v - v.mean()) / v.std()
    return np.clip(mean + amp * v, SAND_MIN, SAND_MAX)
