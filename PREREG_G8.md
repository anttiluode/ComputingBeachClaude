# Pre-registration: G8 (does an open beach fix the loss wall? does trial-and-keep cross it?)

Written and pushed before the held-out beds (700–707) were run. Dev beds 900–905 were used
to build the instruments, and what they showed is stated below.

## Why

My G5/G7 conclusion was: "the next thing is an open beach, where loss is radiation out of
the domain once per pass, not bulk damping per crossing." Sol challenged it: escaping energy
is still loss, and the physical teacher still won't be the adjoint. Here is the derivation
that says Sol is right. The sand feels Re(∇η·conj ∇τ). For that to be the gradient,
conj(τ) must be the adjoint λ = A⁻¹(conj e·readout), so the teacher would have to
propagate under conj(A). That is the time-reversed medium, **with gain where the real
one has loss**. Any passive loss breaks that, wherever it sits. The open question is only
how much the *location* of the loss matters at equal total loss.

Second question: if no teacher wave survives loss, can the beach learn without one?
ObjektiYksi (Gates 2–3) learned by perturb, measure, keep-or-revert. That needs no
adjoint, so loss can't scramble it. It does need what passive sand lacks: a global
better/worse signal and a way to undo a change, i.e. an agent.

## G8a: boundary loss vs bulk loss at matched Q

`beach/open.py`: per-cell damping. Sponges: **east** (an absorbing beach face beyond the
harbours, last 6 columns) or **rim** (open water all round, 6-cell ramp). Q is measured by
ring-down from a west pulse at ω 0.40 on each bed's initial sand.

Dev (beds 900–905, weak teacher β = 0.003, single exposure minus habit, i.e. G7's epoch-1 cosine):

| | Q | median cos |
|---|---|---|
| bulk 0.0006 / east 0.05 | 667 / 693 | 0.98 / 0.95 |
| bulk 0.0016 / east 0.3 | 250 / 249 | 0.88 / 0.79 |
| bulk 0.002 / east 1.0 | 200 / 193 | 0.83 / 0.71 |
| bulk 0.006 / rim 0.05 | 67 / 67 | 0.31 / 0.12 |
| rim 0.3 | 19 | −0.14 |

Held-out: beds 700–707, both tasks (quiet, loud), the same five pairs.

| | prediction | pass rule |
|---|---|---|
| P8a | an open/absorbing boundary is no better than the same Q in the bulk | in every matched pair, median cos(sponge) ≤ median cos(bulk) + 0.05 |
| P8b | a truly open beach (strong rim) kills the physical teacher | rim 0.3: median cos ≤ 0.3 |
| P8c | Q, not location, mostly sets alignment | over all 9 conditions, Spearman ρ(Q, median cos) ≥ 0.8 |

## G8b: trial-and-keep under this cavity's loss (damping 0.006, Q ≈ 67)

`beach/trial.py`. Each trial perturbs the bed (max |Δs| = 0.005), measures J at harbourN,
keeps the change if J fell, otherwise reverts. 200 trials. Perturbation families:
**white** (iid per cell), **smooth** (Gaussian-filtered, 3 cells), **storm** (the scour
pattern of a random swell, random port among 5 and ω ∈ [0.2, 0.5], random sign), and
**exner** (the same storm, conservative). Reference: exact-gradient descent at the same
step, 200 steps, scored at its **best** point (fixed steps overshoot nulls).
Score in dB of |y| at harbourN, as Sol suggested. Quiet = lower, loud = higher.

Dev (beds 900, 901, 150 trials):
- quiet: storm −42/−18 dB, smooth −43/−43, white −5/−4, exner −2/−2, GD final −15/−20.
- loud: storm +5.5/+5.1, smooth +10/+7, white +2.7/+2.2, exner +1.6/+1.1, GD final +31/+29.

| | prediction | pass rule (8 beds) |
|---|---|---|
| P8d | trial-and-keep learns where every teacher method failed | storm: ≥ 3 dB in the right direction on ≥ 7/8 beds, in both tasks |
| P8e | waves make better trial moves than noise | median storm better than median white, both tasks |
| P8f | conservation costs again | median exner worse than median storm, both tasks |
| P8g | trials hunt resonances poorly | loud: median storm < 0.5 × median GD-best (dB) |

Not predicted, reported: storm vs smooth (dev had smooth ahead on loud), trials kept,
and the number of wave solves each method used.
