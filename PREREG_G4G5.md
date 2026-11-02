# Pre-registration: G4 (distributed floor on the shared benchmark) and G5 (physical teacher)

Written and pushed before either script was run. Agreed split with Sol
([ComputingBeachSol](https://github.com/anttiluode/ComputingBeachSol)). Sol takes Robin-wall
resolution K = 8…48, arrival-relative readout and a gradient-matched local writer. This repo
takes the **distributed floor** and the **physically injected teacher**.

Medium, writer and constants are unchanged from G1 (`beach/medium.py`, scour, rate rule,
σ_c, swell A west ω 0.40, swell B north ω 0.47, bursts of 10 write steps, feedback on).

## G4: the distributed floor on Sol's shared benchmark

- Beds: **fresh seeds 300–347** (48), never used before.
- New common final event **X**: a swell from south, ω 0.43, its amplitude matched to A by
  max stress on seed 0's bed, 10 write steps at rate × m_X, with m_X ∈ {0.5, 1, 2}.
- Histories: AB vs BA, and ABX vs BAX at each m_X.
- Readouts. All are read with the material frozen, and all are below threshold (asserted):
  - **R1 steady swell**: as in G1 (south probe, 16 frequencies, both harbours).
  - **R2 whole transient**: pings from west, north and south, traces at both harbours,
    steps [0, 700), every 2nd step.
  - **R3 early, arrival-relative**: for each ping source, [a_s, a_s + 100), where a_s is that
    source's first arrival (1 % of max) at its nearer harbour, measured once on seed 300's
    initial bed and recorded.
  - **R4 late**: [a_s + 300, a_s + 400).
- Decoder, noise and evaluation as in G1: ridge LDA, leave one bed out, ν = 0.01 of feature
  RMS, fresh noise per read. Plus the decoder on the sand itself (upper bound).

| | prediction | pass rule |
|---|---|---|
| P4a | the window result replicates on fresh beds | AB/BA: R2 ≥ 0.65 and R3 > R4 |
| P4b | the early arrival window beats the steady swell | R3 ≥ R1 in **every** history condition |
| P4c | a common final event does not erase order **in the sand** | ABX/BAX sand decoder ≥ 0.80 at m_X = 1 |
| P4d | but it reduces what a ping can read, more for stronger X | best-of-R1–R4 read: ABX/BAX at m_X = 2 is at least 0.05 below AB/BA |

## G5: the numerical adjoint vs a physically injected teacher

Damping ∈ {0, 0.0005, 0.002, 0.006}, eight beds (seeds 500–507), forward swell west ω 0.40,
loss J = ½|y|² at harbourN, scour with no threshold (so that only the field question
is tested). On identical beds, compare:
- λ: the exact adjoint. A_λ = c·conj(e).
- τ: the physical teacher, a swell from harbourN with amplitude β·e (β = 0.03).
- **phase error**: in the lossless case τ = β·conj(λ) exactly. Report the stress-weighted
  mean |arg(τ̂) − arg(β·conj(λ̂))| over cells, weights |∇η|·|∇λ|.
- **single exposure** (no subtraction in code): the update the sand actually makes with the
  teacher on, writer(stress(η + τ)), against the natural update writer(stress(η)). Score:
  the cosine of each with −∂J/∂sand, and the loss after an equal-size step.

| | prediction | pass rule (median over 8 beds) |
|---|---|---|
| P5a | the physical teacher equals conj(λ) when lossless | phase error ≤ 0.01 rad at damping 0 |
| P5b | phase error grows with loss | monotone increase over the four dampings |
| P5c | a single-exposure teacher is only slightly better than none at this cavity's loss, because the self-term dominates | at 0.006: cos(on) − cos(natural) ≤ 0.05 |
| P5d | when lossless, a single exposure with a weak teacher still adds the right direction | at damping 0: cos(on − natural, −g) ≥ 0.95 (the sand's own difference, a diagnostic) and cos(on) > cos(natural) on ≥ 6/8 beds |

P5d's first clause is the subtraction Sol pointed out. It is a diagnostic, not a physical
claim. Its second clause is the physical one.
