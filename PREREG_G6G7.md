# Pre-registration: G6 (the compliance law) and G7 (sand with a sliding threshold)

Written and pushed before either script was run on its held-out beds. Medium, ports, swell
frequency and solver are unchanged (`beach/medium.py`). New shared code: `beach/sliding.py`.

## Why

Both teacher-wave gates so far (G2, G5 here; G2b–G2d in ComputingBeachSol) hit the same
wall: the gradient lives in the **cross term** of the stress, and getting it alone needs
two exposures subtracted **in code**. Plain scour without a teacher was not a neutral
control either: it was already partly aligned with the task. G6 tests an explanation
of that. G7 tests a physical way round the wall.

## G6: is plain scour gradient descent on the source's own compliance?

Claim (derivation in `beach/sliding.py`). For a drive F and a steady swell η = A(s)⁻¹F with A
real symmetric (lossless), the compliance J_c = Re(F·η) has dJ_c/ds = (g/2) Σ_faces (∇η)².
Scour lowers sand by |∇η|², so **scour = −∇J_c** up to the per-cell face average. With loss,
A is complex symmetric and the identity breaks. A scratch check on beds 200–204 gave cos
0.994 lossless and 0.10 at damping 0.006. The held-out test below uses new beds.

- Beds 600–607. Drives from **west, north, south**, at ω ∈ {0.20, 0.30, 0.42}.
- Damping ∈ {0, 0.0005, 0.002, 0.006}. Scour with threshold at the 0 and 0.5 quantile of stress.

| | prediction | pass rule (72 cases per damping) |
|---|---|---|
| P6a | lossless scour is the compliance gradient | median cos(scour, −∇J_c) ≥ 0.98 at damping 0, no threshold |
| P6b | loss breaks it, monotonically | median cos falls at each damping step |
| P6c | at this cavity's loss it is mostly gone | median ≤ 0.3 at damping 0.006 |
| P6d | so a small lossless scour step lowers J_c | J_c decreases in ≥ 90 % of cases (step max 1e-4) |

## G7: a sliding threshold does the subtraction

Each cell carries sand s and a threshold θ that follows its own stress:

    ds = −r(σ − θ),   dθ = (σ − θ)/T

After a long time with the forward swell alone, θ equals that swell's stress and the bed is
at rest (homeostasis, like an axon initial segment that moves with chronic activity, or a
gravel bed that hardens under sub-threshold flow). With a teacher swell τ on, σ − θ is the
cross term plus ½|∇τ|², so **the medium's own memory removes the self-term**.

### What the dev beds (900–905, never to be reused) already showed, and what it changed

- G5's teacher β = 0.03 is **not weak**: max|τ|/max|η| ≈ 1.5, so ½|∇τ|² is not small.
  Single exposure minus habit had median cos 0.75, min −0.95. At **β = 0.003** (ratio 0.15):
  median 0.991, min 0.978. **β = 0.003 is fixed for G7.** This revises P2c and P5d's
  "weak teacher" (both were run at β = 0.03).
- Over epochs the threshold goes stale. The bed moves, the natural stress changes, and θ
  no longer cancels it. The cosine decays from 0.99, faster for larger steps. A finite T
  (θ tracking) did worse on dev, because θ also learns the teacher.

### Protocol

- Beds 600–607. Forward swell west, ω 0.40. Teacher: a swell from harbourN with amplitude
  β·e, as in G5.
- Two tasks: **quiet** (J = ½|y|², lower the swell at harbourN) and **loud** (J = −½|y|²).
- 20 epochs, each update rescaled to max |Δs| = 0.0005 (arms compared on direction).
- Damping ∈ {0, 0.0005, 0.006}.
- Arms:
  - `gd`: exact gradient (ceiling).
  - `natural`: teacher on, fixed threshold, i.e. G5's single exposure.
  - `sliding`: teacher on, θ frozen at the habituated value (T = ∞).
  - `sliding_T3`: T = 3.
  - `sliding0`: sliding threshold with the teacher **off** (control).
  - `twophase`: subtraction in code (G2 reference).
- Score: improvement fraction f = (J₀ − J_end) / (J₀ − J_end,gd).

| | prediction | pass rule (8 beds, each task) |
|---|---|---|
| P7a | one exposure with a sliding threshold points along the gradient | damping 0: median epoch-1 cos(sliding, −g) ≥ 0.95 |
| P7b | it learns, where plain sand does not | damping 0: J_end(sliding) < J_end(natural) on ≥ 7/8 beds |
| P7c | without a teacher the habituated sand does not move | sliding0: Δs exactly 0 over all 20 epochs, all beds |
| P7d | it gets at least half way to exact gradient descent | damping 0: median f(sliding) ≥ 0.5 |
| P7e | a tracking threshold learns the teacher away | damping 0: median f(sliding_T3) < median f(sliding) |
| P7f | the threshold goes stale as the bed moves | damping 0: median epoch-20 cos < median epoch-1 cos − 0.05 |
| P7g | the loss wall is not fixed by this | damping 0.006: median epoch-1 cos(sliding) ≤ 0.4 |

P7a and P7c are close to the derivation. P7b and P7d are the claims that matter. P7g is
the honest limit: a sliding threshold removes the **subtraction** wall, not the **loss** wall.
