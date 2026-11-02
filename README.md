# ComputingBeachClaude

**Can waves write a sand bed that later waves read, and can the bed learn?**

**Live page: [anttiluode.github.io/ComputingBeachClaude](https://anttiluode.github.io/ComputingBeachClaude/)**. Play swells and watch them write the floor, run two beaches through the same storms in opposite order, and play notes through the trained bay. The page runs the same physics as the Python, ported to [`beach.js`](beach.js) and checked against it by `node tests/check_js.mjs` (ping traces to 5e-13, router contrasts to 0.02 dB). It needs a web server for its JavaScript modules: GitHub Pages, or `python -m http.server`.

Antti's beach picture (2 Nov 2026): waves hit a beach, move sand, and the moved sand changes
the next wave. In that picture, routing is the sand acting as a maze, looping is the
basins, and memory is the sand itself. This repo builds the simplest honest version and
tests three things with pre-registered gates ([PREREG.md](PREREG.md), pushed as
`b963e00` before any G1–G3 run). Sol works the same question in a separate repo.

Built by Claude (Opus 5.5).

![summary](results/summary.png)

## The medium

[`beach/medium.py`](beach/medium.py). WallMemoryLaw's Bunimovich stadium (96×96,
3112 water cells), its leapfrog scheme, damping 0.006, Neumann walls. New: **the floor
is sand.** Depth = 1 − sand, and each face carries g·depth (shallow water,
speed² = g·depth). Sand in [0, 0.6] makes the wave speed vary by a factor of 0.63–1, so
the sand is the maze. Steady swells are solved in the frequency domain with the exact
symbol of the time stepper.

Two sand writers, both **toy rules**: stress = time-averaged (grad u)², and
E = max(stress − σ_c, 0).
- **scour**: sand drops where the water is agitated. Δsand = −rate·E.
- **exner**: conservative. Sand leaves agitated water and settles in calm water.
  Δsand = rate·Lap(E).

The threshold σ_c makes a weak ping read without writing: below threshold, nothing moves.

## Gate 0: the instrument is right (all pass)

| check | result |
|---|---|
| flat bed equals WallMemoryLaw's stadium cavity | 1.0e-14 |
| swell solver equals time stepper steady state | 1.1e-6 (after removing the Neumann zero mode, a non-decaying uniform offset that carries no stress) |
| adjoint gradient equals finite differences | 2.1e-8 |
| time-averaged stress of physically superposed swells contains the predicted cross term | cos 0.99997 |

## G1: does the beach remember the order of two swells?

Swell A (west, ω 0.40), then swell B (north, ω 0.47), versus B then A, with matched
exposure. 48 beds of random roughness. A weak ping from the south port is heard at two
harbours over 16 frequencies. The decoder is trained on 47 beds and tested on the held-out one.

| | prediction | result |
|---|---|---|
| P1a | order readable at the strongest write (both writers) | **FAIL**: read 0.59 (scour), 0.44 (exner) |
| P1b | without feedback, order is not stored | **PASS**: open-loop AB and BA beds identical to 3e-15, read 0.51 |
| P1c | the weakest write is not readable | **PASS** |
| P1d | AB − BA grows as the square of write strength | **FAIL**: slope 1.0 (scour), 1.6 (exner) |

**What is true:** the sand does store the order. From the sand itself, a linear decoder
separates AB from BA on held-out beds with 0.98 accuracy (scour, s = 0.04). The open-loop
control shows that this comes entirely from **feedback**, because waves are routed by
sand that earlier waves wrote.

**What fails:** the ping can't read it. Post hoc ([posthoc_g1_diffread.py](posthoc_g1_diffread.py)),
the read stays at 0.60–0.64 with **zero** noise, with before/after differencing and
with probes from all three ports. So noise is not the limit. In this reverberant cavity
the echo is such a scrambled function of the sand that the order difference points a
different way in every bed's echo, and no reader trained on other beds finds it. The
P1d slope failure is the same fact seen from the write side: even the weakest write is
past the perturbative regime. Reads were confirmed nondestructive: the worst probe
stress was 0.27 σ_c.

**Listening window (post hoc, after Sol's [ComputingBeachSol](https://github.com/anttiluode/ComputingBeachSol)
read order at 0.97 with windowed time-domain echoes from 8 writable wall patches).**
[posthoc_g1_window.py](posthoc_g1_window.py) reads the same G1 beds with time-domain
pings from three ports, in cumulative and sliding windows. Its predictions were
written before running (`1dce5b8`), and two of them failed for a reason I should have
caught. I timed the windows on WallMemoryLaw's wave speed, but this bed is about 0.7
deep, so waves are about 16 % slower. First arrivals at the harbours are at steps
90–211, so windows ending at 100–200 hold mostly silence. W1 and W2 fail as registered
but did not test the tour law. What the data does show:

| scour, s = 0.04 | read accuracy |
|---|---|
| steady-state swells (G1, infinite window) | 0.59 |
| time-domain pings, [0, 600) | **0.75** |
| sliding [100,200) / [200,300) / [350,450) / [500,600) | 0.64 / **0.74** / 0.63 / 0.57 |
| exner, any window | ≤ 0.58 (W3 pass) |

The order information seemed to sit **just after first arrival** and dilute with later
reverberation. That looked like the tour-law shape measured from arrival, but it was
post-hoc-on-post-hoc, and **it did not replicate** in pre-registered G4 on fresh beds
(arrival-early 0.64 vs whole transient 0.75). Even the best window reaches 0.75, not Sol's 0.97, so the
window is only part of the gap. The other part is probably 3112 writable floor
values versus Sol's 8 wall patches.

### G4: the floor on the shared benchmark ([PREREG_G4G5.md](PREREG_G4G5.md), fresh beds 300–347)

The split was agreed with Sol: Sol scales Robin wall patches (K = 8…48), and this repo
covers the distributed floor. Sol also proposed **ABX vs BAX**: an identical final wave
X after both orders. That checks that the reader isn't just recognising whichever wave
came last. Measured arrivals: west 202, north 92, south 116 steps.

| history | sand | R1 steady swell | R2 whole transient | R3 early (arrival + 0–100) | R4 late (arrival + 300–400) |
|---|---|---|---|---|---|
| AB vs BA | 0.98 | 0.64 | **0.75** | 0.64 | 0.54 |
| ABX vs BAX, X × 0.5 | 0.99 | 0.64 | **0.73** | 0.52 | 0.59 |
| ABX vs BAX, X × 1 | 0.99 | 0.55 | **0.74** | 0.54 | 0.55 |
| ABX vs BAX, X × 2 | **1.00** | 0.65 | **0.70** | 0.56 | 0.66 |

- **P4a PASS**: the transient read replicates on fresh beds (0.75), early > late.
- **P4b FAIL**: the arrival-early window is *not* better than the steady swell in every
  condition. The "order lives just after first arrival" reading from the post-hoc window
  run does **not** replicate. The whole transient is the best reader everywhere.
- **P4c PASS**: a common final wave does not erase the order in the sand. The sand
  decoder stays 0.99–1.00 even when X is twice as strong. The floor holds the
  *sequence*, not just the last event.
- Post hoc, does X *amplify* the history (steered by it, carving it deeper)? No. On 8
  beds the AB/BA sand difference ‖sand_ABX − sand_BAX‖ / ‖sand_AB − sand_BA‖ is 1.00,
  1.02 and 1.03 for X at ×0.5, ×1 and ×2, growing on only 3–4/8 beds. X writes about
  the same thing onto both beaches and leaves the earlier difference intact: neither
  erased nor amplified.
- **P4d PASS** (barely: 0.05 drop): a stronger X makes the order harder to read,
  0.75 → 0.70.

Reads were nondestructive throughout (worst probe 0.19 σ_c). So for a distributed floor:
order is stored almost perfectly, survives overwriting, and is readable at about 0.70–0.75
with no single privileged time window. Sol's 8 patches read at 0.97. If Sol's K-sweep
shows readability falling as K grows, the gap is the dimension of the writable state.

## G2: does the sand compute a gradient?

The physical teacher is a swell sent back from the output harbour with amplitude set
by the error. For steady swells in a lossless reciprocal medium, the time-averaged
cross stress between forward and teacher swells **is** the gradient of the loss with
respect to depth. Scour then takes a gradient step, with no backprop anywhere.
(This holds for swells, not pings: for a pulse the adjoint is the time-reversed field,
which is never in the water at the same moment. That was corrected before the prereg.)

| | prediction | result |
|---|---|---|
| P2a | lossless: sand step = gradient | cos **0.989**, FAIL against the 0.99 bar. Effectively right: the gap is cells averaging their faces while the gradient sums them |
| P2b | at this cavity's damping it still nearly is | **FAIL: 0.24** |
| P2c | the threshold removes only the sub-threshold part | **FAIL as set up**: my "weak" teacher was 6× the threshold near the harbour. Post hoc with a truly weak teacher, it holds within 0.016 at every quantile (lossless) |
| P2d | conservative exner is a descent direction | **FAIL** |
| P2e | one-phase ≈ two-phase | PASS |

**The result is the loss curve** (left panel, post hoc
[posthoc_g2.py](posthoc_g2.py)). Alignment is above 0.98 down to Q ≈ 800, falls around
Q ≈ 400, and is gone by this cavity's Q ≈ 67. Sending the teacher with amplitude
conj(e), which is the true adjoint field of the lossy medium, degrades more slowly
(0.97 at Q 400, 0.88 at Q 200) but also fails at Q 67. In a reverberant cavity a wave
crosses the medium many times, so a little loss per crossing is a lot of phase error
by the time forward and teacher meet. Prior art: in-situ photonic backprop (Pai et al.
2023) and Hamiltonian echo backpropagation (López-Pastor & Marquardt 2023) assume, or
engineer, the lossless/time-reversible case. This measures what happens when you don't.

### G5: the physical teacher, and no subtraction in code ([PREREG_G4G5.md](PREREG_G4G5.md))

Sol's review (`ComputingBeachSol`) pointed out that the 0.989 above comes from
writer(u + λ) − writer(u − λ), and that **the code, not the sand, does that subtraction**.
A real beach with the teacher on only ever does writer(u + τ). G5 measures that single
exposure, and the field the physical teacher actually makes, against the exact adjoint.
Eight fresh beds, β = 0.03, no threshold.

| damping | phase error, physical teacher vs conj(adjoint) | cos(natural, −g) | cos(teacher on, −g) | on beats natural (direction) | on lowers loss more |
|---|---|---|---|---|---|
| 0 | **1e-12 rad** | −0.24 | +0.16 | 8/8 | **3/8** |
| 0.0005 | 1.33 rad | +0.17 | +0.19 | 6/8 | 3/8 |
| 0.002 | 1.48 rad | −0.08 | −0.02 | 6/8 | 6/8 |
| 0.006 | 1.56 rad | −0.14 | −0.11 | 5/8 | 5/8 |

- **P5a PASS**: lossless, the injected teacher swell *is* the conjugate adjoint, to 1e-12 rad.
- **P5b PASS**: phase error grows with loss, reaching 1.56 rad at this cavity's loss.
  That is about π/2, the mean of a random phase: the teacher's phase no longer has
  anything to do with the adjoint's. Caveat: at 0.0005 the field phase error is already
  1.33 rad, while the update alignment there (post-hoc G2) was still 0.98. So this
  field-wise metric is harsher than what the update feels. Near-resonant modes flip
  phase under tiny loss but carry little of the cross term.
- **P5c PASS**: at 0.006 a single teacher exposure is no better than none (−0.11 vs −0.14).
- **P5d FAIL**: lossless, the sand's own on − natural difference aligns at only 0.82
  (the teacher's self-term). The physical clause is the telling one: the teacher tilts
  the step the right way on 8/8 beds, but that step lowers the loss more than plain
  scour on only **3/8**. The cross term rides on a step dominated by the wave's own
  stigmergy.

So, even lossless, **a beach with one sand law doesn't learn from a teacher**. It needs
a contrastive mechanism: something that writes the difference between teacher-plus and
teacher-minus exposures (a two-sign material, a slow/fast pair of sand fractions), as
coupled learning and equilibrium propagation require. Sol's point, measured.

### G6: plain scour is gradient descent on the source's own compliance ([PREREG_G6G7.md](PREREG_G6G7.md))

Why was "natural" sand already partly aligned with the tasks (G3 here, +0.21 forward-only
in ComputingBeachSol)? For a drive F and steady swell η = A(s)⁻¹F, take the source's
compliance J_c = Re(F·η). When A is real symmetric (lossless), the backward field for
J_c is η itself, so

    dJ_c/ds = −ηᵀ (∂A/∂s) η = (g/2) Σ_faces (∇η)²

Scour lowers the bed by |∇η|², which makes it exactly −∇J_c. **Stigmergy is gradient
descent with a built-in loss.** This is the self-adjoint case known from structural
topology optimisation, where strain-energy-driven remodelling (bone) minimises compliance.
Fresh beds 600–607, three drive ports × three frequencies:

| damping | median cos(scour, −∇J_c) | min | one small step lowers J_c |
|---|---|---|---|
| 0 | **0.995** (0.93 at the 0.5-quantile threshold) | 0.980 | **72/72** |
| 0.0005 | 0.982 | −0.99 | 83 % |
| 0.002 | 0.717 | −0.97 | 78 % |
| 0.006 | 0.121 | −0.83 | 58 % |

**P6a–P6d all PASS.** The beach optimises *something* by itself: the compliance of
whatever drives it. A task only gets help where it overlaps with that. And at this
cavity's loss, even that self-optimisation is gone.

### G7: sand with a sliding threshold ([PREREG_G6G7.md](PREREG_G6G7.md))

The subtraction G5 needed can be stored in the medium instead of done in code. Give each
cell a threshold θ that follows its own stress: ds = −r(σ − θ), dθ = (σ − θ)/T. After a
long time with the forward swell alone, θ = σ_nat and the bed is at rest. Turn on a
teacher and σ − θ is the cross term plus ½|∇τ|². Nature has this rule twice: the axon
initial segment moves and resizes to counter chronic activity (homeostasis), and gravel
beds harden under sub-threshold flow (the stress-history effect).

Found on dev beds 900–905 and fixed in the prereg before the held-out run: **G5's teacher
was not weak.** At β = 0.03 the teacher swell is 1.5× the forward swell, so its own
½|∇τ|² term is large. At β = 0.003 (ratio 0.15), a single exposure minus the habituated
threshold aligns at a median of 0.991. That also revises P2c and P5d, which ran at β = 0.03.

Fresh beds 600–607, β = 0.003, 20 epochs, each update rescaled to the same max step.
f is the fraction of exact gradient descent's improvement reached:

| damping, task | gd | natural (teacher on, fixed threshold) | **sliding** (θ frozen at habit) | sliding, T = 3 | sliding, teacher off | two-phase (code subtraction) |
|---|---|---|---|---|---|---|
| 0, quiet harbour | 1 | 0.27 | **0.92** | 0.81 | 0 (exactly) | 1.00 |
| 0, loud harbour | 1 | 0.00 | **0.11** | 0.10 | 0 | 1.03 |
| 0.0005, quiet | 1 | −0.06 | 0.75 | 0.33 | 0 | 0.95 |
| 0.0005, loud | 1 | 0.18 | 0.41 | 0.24 | 0 | 0.60 |
| 0.006, quiet | 1 | −0.25 | −0.10 | 0.14 | 0 | 0.02 |
| 0.006, loud | 1 | 0.20 | −0.01 | 0.03 | 0 | 0.02 |

- **P7a PASS**: lossless, the first single exposure points along the gradient (median cos
  0.988 quiet, 0.986 loud). No subtraction in code: the medium's memory of its usual
  stress removes the self-term.
- **P7b FAIL** (quiet 7/8, loud **5/8**) and **P7d FAIL** (quiet f 0.92, loud **0.11**).
  It learns the quiet task. On the loud task it raises the harbour's |y|² by about 4.6×
  (median), against about 2× for plain sand, but gd hunts resonances and reaches ×3.5 to
  ×200 000 in this lossless cavity. So f is tiny.
- **P7c PASS**: with the teacher off, the habituated sand moves by exactly 0 on every bed.
- **P7e PASS**: a threshold that keeps tracking (T = 3) learns the teacher away and does worse.
- **P7f PASS**, and this is the limit: **the threshold goes stale.** As the bed moves, the
  natural stress changes and θ no longer cancels it. On half the loud beds the cosine
  flips sign between epochs 5 and 10. Making the harbour loud changes the whole field
  faster, so the stale part grows faster. A dev-bed try of "rest" interludes (teacher
  off, θ re-habituates) gave mixed results, so it isn't claimed.
- **P7g PASS**: at this cavity's loss, nothing works, code subtraction included (f 0.02).
  The physical teacher's phase is random there (G5).

So a sliding threshold **removes the subtraction wall** for a weak teacher and a short
walk. It does **not** remove the loss wall, and it has a range: the bed can only move so far
before its memory of "normal" is out of date.

## G3: can the beach learn to route?

Two swells from the west: ω₁ should reach harbourN and ω₂ harbourS. Five beds, 300
epochs, with each arm's step rescaled to the same size.

| | prediction | result (mean gain) |
|---|---|---|
| P3a | exact gradient on the sand can route | **PASS: +25.5 dB** (+15.7 to +36.2) |
| P3b | teacher-scour learns ≥ half of that | **FAIL: +4.5 dB** |
| P3c | teacher-exner learns | **FAIL: −3.6 dB** |
| P3d | natural sand (no teacher) stays within ±3 dB | **FAIL: +6.5 dB**, positive on all 5 beds |

Post hoc ([posthoc_g3_alignment.py](posthoc_g3_alignment.py)): at damping 0.006 the
teacher's first step has cos −0.06…+0.09 with the true gradient, which is nothing.
Plain scour has cos +0.11…+0.30. So "natural" was not a neutral control for this
routing task: ordinary scouring under these two swells happens to point partly
downhill. I should have checked that before registering it as a control. A low-loss
rerun (damping 0.0005) was **inconclusive**: even exact GD only gained +7.1 dB, because
the near-lossless contrast landscape is too rough for fixed steps. The first attempt
at step 0.005 was aborted for the same reason, and its log is kept.

![beds](results/beds.png)

**What GD carved** (seed 202, +38.6 dB): concentric sand rings around the source and
around the north harbour. These are Fresnel-style lenses that focus the swell. That is
inverse design (known), but it's the picture of what a beach that *can* be trained
looks like.

## Ledger

**Verified**
- The sand bed reproduces WallMemoryLaw and the swell solver is exact (Gate 0).
- Waves write order into sand through feedback (sand decoder 0.98; open loop 3e-15).
- In a lossless medium the sand computes the gradient (cos 0.989), with the threshold
  behaving as predicted (post hoc, within 0.016).
- Sand depth is a strong enough maze to route two swells to two harbours (+25.5 dB by exact gradient).
- Plain scour is gradient descent on the driving source's own compliance when lossless
  (G6, cos 0.995 on 72 held-out cases), and that law fades with loss (0.12 at 0.006).
- A sliding threshold (sand that erodes above its habitual stress and deposits below it)
  turns one weak-teacher exposure into a gradient step with no code subtraction (G7, cos
  0.99), and trains the quiet-harbour task to 92 % of exact descent, lossless.

**Failed**
- Reading order back with pings (P1a); quadratic order growth (P1d).
- Teacher learning at this cavity's real loss (P2b, P3b, P3c); exner as descent (P2d).
- My natural-sand control (P3d): it is aligned with the task. G6 explains why.
- Sliding-threshold learning on the loud-harbour task (P7b, P7d): its threshold goes
  stale within ~5–10 epochs. And at this cavity's loss, sliding threshold learning, like
  every teacher method here, does nothing (P7g).
- G5's β = 0.03 teacher was 1.5× the forward swell, not weak (found in G7's dev runs).

**Bugs found on the way**
- G1's `history()` never applied `ampB`, so swell B ran at amplitude 1, not the
  registered 1.18. Its peak stress was 0.72× A's instead of equal. AB and BA still got
  identical exposure to each other, so the order comparison stays fair, but the
  registered "B matched to A" did not hold. G4 uses the matched amplitude.
- G1 crashed writing JSON (numpy bool) after every cell had run. `g1.json` was rebuilt
  from the log, and the lost probe-stress check was recomputed (0.27 σ_c).

**Not done**
- Sol's step 4: WallMemoryLaw's tour law on continuous sand.
- An open beach, where loss is radiation out of the domain, not bulk damping.
- Fast gating (overtopping berms), real sediment physics, time-domain pings for learning.
- A nonlinear reader for G1, which would make sense given that the echo scrambles a
  linearly stored difference.

## Run

```
pip install -r requirements.txt
python gate0.py <path to a WallMemoryLaw checkout>   # G0a needs it; the rest runs without
python g2_gradient.py                 # seconds
python g1_order.py slope              # ~1 min
python g1_order.py main               # ~25 min
python g3_router.py                   # ~20 min
python posthoc_g2.py; python posthoc_g1_diffread.py
PYTHONPATH=. python posthoc_g3_alignment.py
python posthoc_g3_lowloss.py 0.0005 0.001
python g6_compliance.py               # ~1 min
python g7_sliding.py                  # ~3 min
python make_figures.py
node tests/check_js.mjs              # the web page's physics against the Python
```

Do not hype. Do not lie. Just show.
