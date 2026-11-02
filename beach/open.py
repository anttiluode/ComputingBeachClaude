"""Open-beach variant: loss lives in an absorbing sponge at the boundary instead of in the
bulk.  Same stencil, but the damping gamma is a per-cell array.  With gamma = 0 everywhere
this is Beach(damp=0) exactly."""
import numpy as np
import scipy.sparse as sp

from beach.medium import Beach, DT, N, CENTER, STADIUM


def sponge(bch, where, strength, width=6.0):
    """gamma per cell.  where = 'east' (an absorbing beach face beyond the harbours) or
    'rim' (open water all round).  Quadratic ramp over `width` cells to `strength`."""
    rr, cc = bch.cells[:, 0].astype(float), bch.cells[:, 1].astype(float)
    if where == "east":
        x0 = CENTER + STADIUM[0] + STADIUM[1] - width        # last `width` columns
        d = np.clip(cc - x0, 0, None) / width
    elif where == "rim":
        X, Y = cc - CENTER, rr - CENTER
        L, R = STADIUM
        dist_in = np.where(np.abs(X) <= L, R - np.abs(Y), R - np.hypot(np.abs(X) - L, Y))
        d = np.clip((width - dist_in) / width, 0, None)
    else:
        raise ValueError(where)
    return strength * np.clip(d, 0, 1) ** 2


class OpenBeach(Beach):
    def __init__(self, gamma):
        super().__init__(damp=0.0)
        self.gamma = np.asarray(gamma, float)

    def symbol(self, w, damp=None):
        g = self.gamma if damp is None else damp
        a, b = 1 + g * DT / 2, 1 - g * DT / 2
        z = np.exp(-1j * w * DT)
        return (a * z - 2 + b / z) / DT ** 2

    def operator(self, sand, w):
        return (self.laplacian(sand) + sp.diags(self.symbol(w))).tocsc()
