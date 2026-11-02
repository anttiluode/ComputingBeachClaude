"""POST HOC: does each swell frequency print sand bars at its own spacing (half a wavelength)?
Flat bed (sand 0.3, depth 0.7), one scour step, no threshold.  For each written swell, the share
of the print's spatial power in a +-15% band around each swell's predicted bar wavenumber 2k.
(Antti, 2 Nov 2026: "small waves leave fine print, large waves larger print".)"""
import numpy as np
from beach.medium import Beach, scour
b = Beach(); sand = np.full(b.n, 0.3); c = 0.65 * np.sqrt(0.7)
W = (0.25, 0.40, 0.60)
ky, kx = np.mgrid[-48:48, -48:48] * 2 * np.pi / 96; kr = np.hypot(kx, ky)
for src_name in ("west", "north", "south"):
    M = np.zeros((3, 3))
    for i, w in enumerate(W):
        d = scour(b, b.stress(b.solve(sand, w, b.port(src_name))), 0, 1.0)
        g = np.nan_to_num(b.to_grid(d - d.mean(), fill=0.0))
        P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
        for j, w2 in enumerate(W):
            K = 2 * w2 / c
            M[i, j] = P[(kr > 0.85 * K) & (kr < 1.15 * K)].mean()
    print(src_name, "(rows: written swell 0.25/0.40/0.60; cols: share at each bar frequency)")
    print(np.round(M / M.sum(1, keepdims=True), 2))
