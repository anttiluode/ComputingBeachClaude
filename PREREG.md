# Pre-registration: ComputingBeachClaude G1–G3

Written and pushed **before** any G1, G2 or G3 run. Gate 0 (solver checks) was run first
and is reported in `results/gate0.json`. Nothing else has been simulated on the sand.

## The medium (fixed)

`beach/medium.py`. WallMemoryLaw's Bunimovich stadium (96×96, 3112 water cells), its
leapfrog scheme, damping 0.006, Neumann walls. New: the floor is sand. Depth
D = 1 − sand, and each face between water cells carries g·D_face (shallow water,
speed² = g·depth, g = 0.65²). Sand is clipped to [0, 0.6], so wave speed varies by a
factor of 0.63–1. With sand = 0 this is WallMemoryLaw's stencil to 1e-14 (G0a).

Steady swells are solved in the frequency domain with the exact symbol of the time
stepper (G0b: 1e-6 after removing the non-decaying zero mode). **Stress** on a cell is
the time average of (grad u)² over its faces, ½|grad η|², for the superposition of all
swells present at that frequency. G0d confirmed in the time domain that physically
superposed swells carry exactly the cross term this formula predicts.

Initial beds: `rough_bed(seed)`, mean 0.3, smooth roughness (sd 0.06, correlation
3 cells). Ports: west (50,18), north (30,48), south (66,40), harbourN (36,78),
harbourS (60,78). Each is a Gaussian of σ 1.55, used both as a source and as a receiver.

**Two sand writers, both toy rules, labelled as such.** E = max(stress − σ_c, 0):
- **scour** (W1): Δsand = −rate·E. Local and not sand-conserving.
- **exner** (W2): Δsand = rate·Lap(E). Conservative Exner form with flux −rate·grad E:
  sand leaves agitated water and settles in calm water.

## A correction made before this file

In chat I said the teacher cross term gives the gradient. That is true only for
**steady swells**. For a ping, the adjoint field is the *time-reversed* forward field,
and it is never present in the water at the same moment. For one frequency, time
reversal is phase conjugation, so in a lossless medium the teacher is a physical swell
from the output port with complex amplitude e = 2∂J/∂conj(y). The time-averaged cross
stress is then exactly −(2/g)·∂J/∂D per face. Damping breaks this slightly. G1 uses
pings for reading, and G2/G3 use swells for learning.

---

## G1 — does the beach remember order?

Swell **A**: west, ω = 0.40, amplitude 1. Swell **B**: north, ω = 0.47, amplitude scaled
so its maximum cell stress on seed 0's initial bed equals A's (matched exposure).
A burst is 10 write steps: solve the field on the current bed, then update the sand.
The feedback is included.

- σ_c = median of A's cell stress on seed 0's initial bed (q = 0.5), fixed for all seeds.
- rate is set once per writer and strength s, so that A's first write step on seed 0
  has max |Δsand| = s. s ∈ {0.005, 0.01, 0.02, 0.04}.
- Histories: **AB** and **BA** on 48 beds (seeds 0–47), with matched total exposure.
- **Read**: a weak probe from south at 16 frequencies in [0.30, 0.55], heard at both
  harbours, gives 64 real features. The probe amplitude is set so its maximum stress on
  any bed is ≤ 0.5·σ_c, so a read writes **exactly nothing**. This is asserted on every
  read and logged.
- Read noise: fresh Gaussian on every read, ν ∈ {0.01, 0.05} × the RMS of the features.
- **Decoder**: ridge LDA (λ = 0.01·trace/d) on standardised features. Leave one bed out:
  train on 47 beds × 2 orders, test the held-out bed's AB and BA reads. 96 decisions.
- **Open-loop control**: each burst's writes are computed from fields on the
  *initial* bed (no feedback), then applied in order with clipping. Without feedback
  the order can matter only through clipping.
- Upper bound: the same decoder applied to the sand itself.

| | prediction | pass rule |
|---|---|---|
| P1a | closed loop at s = 0.04, ν = 0.01: order is readable | LOO ≥ 0.90 for **both** writers |
| P1b | the open-loop control is near chance | LOO ≤ 0.65 at s = 0.04, ν = 0.01, both writers |
| P1c | the weakest write is not readable at the higher noise | LOO ≤ 0.75 at s = 0.005, ν = 0.05 |
| P1d | order is a commutator, second order in the write | with no threshold (σ_c = 0), log-log slope of mean ‖sand_AB − sand_BA‖ against s over the four levels is 2 ± 0.3, both writers |

My confidence is low for P1a. The order signal points in a direction set by each bed's
own roughness. If that direction is not shared across beds, a linear decoder fails
even when every single bed's sand differs. The sand upper bound will show whether that
is what happened.

## G2 — is the sand computing a gradient?

Eight beds (seeds 100–107). Forward swell west, ω = 0.40. Loss J = ½|y|² at harbourN
(silence the north harbour). The teacher is a swell from harbourN with amplitude β·e,
e = y, β = 0.03. The sand's step direction is compared with the exact adjoint descent
direction −∂J/∂sand:
- **two-phase**: (Δ(teacher +) − Δ(teacher −))/2, which is pure cross term;
- **one-phase**: Δ(teacher on) − Δ(teacher off).

Thresholds: σ_c = q-quantile of the forward swell's cell stress, q ∈ {0, 0.25, 0.5,
0.75, 0.9}. Predicted mask m_q = 1[forward stress > σ_c] and predicted cosine
P(q) = cos(−g, −g⊙m_q).

| | prediction | pass rule (median over 8 beds) |
|---|---|---|
| P2a | lossless (damping 0), q = 0, two-phase scour is the gradient | cos ≥ 0.99 (not 1: cells average their faces, while the gradient sums them, so wall cells differ) |
| P2b | with damping 0.006 it nearly is | cos ≥ 0.95 |
| P2c | the threshold removes exactly the sub-threshold part | \|cos − P(q)\| ≤ 0.05 at every q, two-phase scour |
| P2d | conservative Exner is a descent direction (preconditioned by −Lap) | ⟨Δ, −g⟩ > 0 on **all** 8 beds at q = 0, two-phase; cos in (0, 0.6) |
| P2e | a weak teacher makes one-phase equal to two-phase | \|cos₁ − cos₂\| ≤ 0.05, scour, q = 0, damped |

## G3 — can the beach learn to route?

A frequency router. Two swells from west: ω₁ = 0.40 should go to harbourN, and
ω₂ = 0.47 to harbourS. J = Σ_k (|y_wrong,k|² − |y_right,k|²)/P_k, where P_k is the
initial-bed total power at both harbours. Contrast C = mean_k 10·log10(|y_right|²/|y_wrong|²).

Arms, five beds (seeds 200–204), 300 epochs, each epoch covering both frequencies. Every
arm's update is rescaled to max |Δsand| = 0.005 per epoch, so the arms compare
*directions*. A fixed physical rate would only change the time scale.
1. **GD**: exact adjoint gradient.
2. **teacher-scour, two-phase**, σ_c at q = 0.5 of each frequency's initial forward stress.
3. **teacher-scour, one-phase** (β = 1).
4. **teacher-exner, two-phase**.
5. **natural scour**: the swells alone, no teacher.
6. initial bed.

| | prediction | pass rule (mean over 5 beds) |
|---|---|---|
| P3a | the instrument can route | GD gain ≥ +10 dB |
| P3b | sand with a teacher learns most of it | teacher-scour two-phase gain ≥ 50 % of GD's gain |
| P3c | conservative sand learns something | teacher-exner gain ≥ +2 dB |
| P3d | without a teacher, nothing is learned on purpose | \|natural gain\| ≤ 3 dB |

## Not tested here

WallMemoryLaw's tour law on continuous sand (Sol's step 4). Real sediment transport,
meaning direction from wave asymmetry and streaming. Time-domain learning with pings.
Fast gating (overtopping berms): that is G4, not pre-registered yet.
