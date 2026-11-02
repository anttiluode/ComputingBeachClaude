"""Trial-and-keep learning (ObjektiYksi Gates 2-3 on the beach): perturb the bed, measure the
harbour, keep the change if the loss went down, otherwise revert.  No adjoint and no
teacher wave, so loss in the medium cannot scramble a teacher's phase.  What it does need,
and the sand does not have by itself: a global scalar "better / worse" signal and a way to
undo a change.  That part is an agent, labelled as such.

Perturbation families, all rescaled to the same max |d sand|:
  white   iid Gaussian per cell
  smooth  Gaussian-filtered noise (correlation 3 cells, like rough_bed)
  storm   scour pattern of a random swell (random port, random frequency), random sign
          (sign + = scour where the storm was agitated, - = accretion there)
  exner   conservative storm: sand leaves the storm's agitated water and settles in its calm
          water (Lap of its stress), random sign
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from beach.medium import apply, N

STORM_PORTS = ["west", "north", "south", "harbourN", "harbourS"]


def perturbation(bch, sand, family, rng):
    if family == "white":
        return rng.standard_normal(bch.n)
    if family == "smooth":
        g = gaussian_filter(rng.standard_normal((N, N)), 3.0)
        return g[bch.cells[:, 0], bch.cells[:, 1]]
    p = STORM_PORTS[rng.integers(len(STORM_PORTS))]
    w = rng.uniform(0.20, 0.50)
    st = bch.stress(bch.solve(sand, w, bch.port(p)))
    sgn = rng.choice([-1.0, 1.0])
    if family == "storm":
        return -sgn * st
    if family == "exner":
        return sgn * bch.lap_cells(st)
    raise ValueError(family)


def keep_or_revert(task_J, bch, sand0, family, trials, step, seed):
    rng = np.random.default_rng(seed)
    sand = sand0.copy()
    J = task_J(sand)
    hist, kept = [J], 0
    for _ in range(trials):
        d = perturbation(bch, sand, family, rng)
        cand = apply(sand, d * (step / np.abs(d).max()))
        Jc = task_J(cand)
        if Jc < J:
            sand, J = cand, Jc
            kept += 1
        hist.append(J)
    return dict(J=hist, kept=kept, sand_final=None)
