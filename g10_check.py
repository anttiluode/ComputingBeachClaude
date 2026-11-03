"""G10 instrument checks (run before any G10 data).  Exits non-zero on failure.

1. A physical read at G9's amplitude (rho 0.125) on the hard bed reproduces G9's read5()
   exactly and writes nothing.
2. A hard-bed storm through destructive.storm() is G9's run_storm() bit for bit.
3. The habituated sliding bed is at rest under the ambient sea alone (G7's P7c).
4. A read well above threshold (rho 8) really moves sand on the hard bed.
"""
import sys
import numpy as np

from beach.medium import rough_bed
from beach import watch as W
from beach import destructive as D

ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, info, flush=True)
    ok &= bool(cond)


s0 = rough_bed(W.b, 960)
bed = D.Bed(s0)
out5, pm, wrote = D.read_phys(bed, D.RHO_G9)
g9, _, pm9 = W.read5(s0)
check("rho 0.125 read == G9 read5", np.allclose(out5, g9, rtol=1e-10, atol=1e-14),
      f"max |diff| {np.abs(out5 - g9).max():.2e}")
check("rho 0.125 read writes nothing", wrote == 0.0 and np.array_equal(bed.s, s0), f"wrote {wrote}")
check("peak probe stress matches G9's", abs(pm - pm9) < 1e-12, f"{pm:.4f} vs {pm9:.4f}")

st = W.draw_storm(np.random.default_rng(1))
a = D.storm(D.Bed(s0), st).s
b_, _ = W.run_storm(s0, st)
check("hard storm == G9 run_storm", np.array_equal(a, b_))

H = D.habituated(s0)
w = D.silent_slot(H.copy())
check("habituated sliding bed at rest", w < 1e-12, f"max |d sand| {w:.2e}")

for rho in (0.5, 1.0, 2.0, 4.0, 8.0):
    bb = D.Bed(s0)
    _, pm, wr = D.read_phys(bb, rho)
    print(f"   hard rho {rho}: peak probe stress {pm:.3f} sigma_c, max write {wr:.2e}, "
          f"cells moved {int((np.abs(bb.s - s0) > 0).sum())}")
bb = D.Bed(s0)
_, _, wr = D.read_phys(bb, 8.0)
check("rho 8 read moves sand", wr > 0)
for rho in (0.125, 2.0):
    hh = H.copy()
    _, _, wr = D.read_phys(hh, rho, "sliding", np.inf)
    print(f"   sliding rho {rho}: max write {wr:.2e}, total |d sand| {np.abs(hh.s - s0).sum():.3e}")
sys.exit(0 if ok else 1)
