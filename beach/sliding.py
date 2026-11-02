"""Shared code for G6 (compliance law) and G7 (sliding-threshold sand).

G6 identity.  For a drive F and steady field eta = A(s)^-1 F, the source's own compliance
J_c = Re(F . eta) has dJ_c/ds = -(eta^T dA/ds eta) = (g/2) sum_faces (grad eta)^2 when A is
real symmetric (lossless).  Natural scour lowers the bed by |grad eta|^2, so it is -grad J_c
up to the per-cell face averaging.  With loss A is complex symmetric, and |grad eta|^2 is
no longer the gradient of anything of this form.

G7 medium.  Each cell carries sand s and a slow threshold theta that tracks its own stress:
    ds     = -r (sigma - theta)          (erode above the habitual stress, deposit below)
    dtheta = (sigma - theta) / T
After long exposure to the forward swell alone, theta = sigma_nat and the bed is at rest.
With a weak teacher tau on, sigma - theta = Re(grad eta . conj grad tau) + 1/2 |grad tau|^2
(+ the drift of sigma_nat as the bed moves): the self-term is cancelled by the medium's own
memory, not by code.
"""
import numpy as np

from beach.medium import Beach, apply


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def compliance_grad(bch, sand, w, src):
    """dJ_c/dsand for J_c = Re(F . eta), F = src (real).  readout = src, e = 1."""
    g, y, eta = bch.grad_sand(sand, w, src, src, lambda y: np.ones_like(y))
    return g, float(np.real(y)), eta


def normed(d, size):
    m = np.abs(d).max()
    return d * (size / m) if m > 0 else d


class Task:
    """J = sign * 1/2 |y|^2, y = harbour readout of the forward swell.
    sign +1: quiet the harbour (descend |y|).  sign -1: make it loud."""

    def __init__(self, bch, w, src, rd, sign, beta):
        self.bch, self.w, self.src, self.rd, self.sign, self.beta = bch, w, src, rd, sign, beta

    def J(self, sand):
        return self.sign * 0.5 * abs(self.rd @ self.bch.solve(sand, self.w, self.src)) ** 2

    def fields(self, sand):
        """forward eta, exact gradient g = dJ/dsand, and the physical teacher swell tau
        launched from the harbour with amplitude beta * e (e = sign * y)."""
        bch = self.bch
        g, y, eta = bch.grad_sand(sand, self.w, self.src, self.rd, lambda y: self.sign * y)
        tau = bch.solve(sand, self.w, self.rd * (self.beta * self.sign * y))
        return eta, g, tau, y


def run_arm(task, sand0, arm, epochs, step, T=np.inf):
    """One training run.  Every arm's per-epoch update is rescaled to the same max |d sand|
    (`step`), so arms are compared on direction, not on rate.  Returns the J trajectory and
    the per-epoch cosine of the update with -dJ/dsand.

    arms:
      gd        exact gradient (reference ceiling)
      natural   teacher on, fixed threshold 0: d = -sigma(eta + tau)        (G5's single exposure)
      sliding   teacher on, sliding threshold: d = -(sigma(eta + tau) - theta)
      sliding0  teacher OFF, sliding threshold (control: should not move)
      twophase  code subtraction, the G2 reference: d = -(sigma(eta+tau) - sigma(eta-tau))/2
    """
    bch = task.bch
    sand = sand0.copy()
    eta0 = bch.solve(sand, task.w, task.src)
    theta = bch.stress(eta0)                       # habituated to the forward swell
    Js, coss, raw = [task.J(sand)], [], []
    for _ in range(epochs):
        eta, g, tau, y = task.fields(sand)
        if arm == "gd":
            d = -g
        elif arm == "natural":
            d = -bch.stress(eta, tau)
        elif arm in ("sliding", "sliding0"):
            sig = bch.stress(eta, tau) if arm == "sliding" else bch.stress(eta)
            d = -(sig - theta)
            theta = theta + (sig - theta) / T if np.isfinite(T) else theta
        elif arm == "twophase":
            d = -0.5 * (bch.stress(eta, tau) - bch.stress(eta, -tau))
        else:
            raise ValueError(arm)
        raw.append(float(np.abs(d).max()))
        coss.append(cos(d, -g))
        sand = apply(sand, normed(d, step)) if raw[-1] > 1e-14 else sand
        Js.append(task.J(sand))
    return dict(J=Js, cos=coss, raw_max=raw)
