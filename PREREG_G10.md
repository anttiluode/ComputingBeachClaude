# Pre-registration: G10 (can the beach stay readable while it learns?)

Written and pushed before the held-out beds (1010, 1011) were run. Dev bed 960 was used to
build and debug the instruments; what it showed is stated below. G9's code and results are
untouched: G10 imports `beach/watch.py` and `g9_analyse.fit`, it does not edit them.

## Why

G9 found the beach's one honest advantage: a storm writes the sand with nothing switched on,
and pinging the bed later localizes it. But G9's reads were free **by construction**:
`read5()` solves the probe fields and never passes them through a writer, and its probes sit
at 0.125 σ_c (the rule "≤ 0.5 σ_c with a ×4 margin" halves the *amplitude*, so stress is
0.5/4). That is the flash-memory rule (read voltage below program voltage), but nobody tested
what happens without it.

BrahClaude (3 Nov 2026) found that in a nonlinear synapse model, repeated neutral pings erase
the memory they read. The "Learning Beach" Sol proposed needs a bed that learns from
experience, and anything that learns from experience also learns from its probes and its
teachers. G10 asks: **can the beach remain readable while learning, or does the activity that
teaches it inevitably destroy its earlier distinctions?**

## Shared setup

- Medium, ports, frequencies, damping 0.006, solver: unchanged (`beach/medium.py`).
- Storms: G9's (`watch.draw_storm`, single tone, 16 frequencies, strength m ∈ [0.7, 1.4],
  scour s = 0.04, 10 substeps), from a seed stream disjoint from G9's (20 000 + 7·bed + tag).
- **Physical read** (`beach/destructive.py`): G9's 3 probes × 16 frequencies, heard at 5 ports,
  12 reciprocal pairs, but every probe exposure is also one write substep at the **storm's own
  fixed scour rate** (no rescaling to a unit step). Fields are re-solved on the current bed at
  each frequency block; a read is 48 substeps. Features are divided by the amplitude factor so
  they are in G9's units at every amplitude.
- **Amplitude** ρ = peak probe stress / σ_c on the calibration bed (seed 0). G9 is ρ = 0.125.
- **Noise**: an absolute sensor floor, 1 % of the RMS of the pristine G9-strength read (G9's
  noise). In feature units it scales as √(0.125/ρ): louder pings read cleaner.
- **Frozen readout**: per amplitude (and medium), one decoder is chosen by G9's search
  (ridge / kNN / RBF kernel ridge, 5-fold CV) on the calibration storms' **first** read,
  baseline = the pristine bed's first read, then frozen for every later read. Score: median
  storm-location error in cells (G9's task). Chance ≈ 19–21.
- **Material separation**: rec_k = storm bed − pristine bed after the **same** history;
  r₀ = what the storm wrote. proj_k = ⟨rec_k, r₀⟩/|r₀|² (1 = the storm's distinction is intact).
  SAND = a frozen decoder fitted on calibration r₀, applied to rec_k (what an ideal reader of
  the material could still get).
- **Diagnostics** (same frozen decoder): *avg* = mean of reads 1..k; *eff* = read k minus the
  pristine bed's read k after the same reads (an efference copy: the device knows what its own
  reads did to an unwritten bed). If eff rescues a read that frozen fails, the loss is **reader
  misalignment**; if SAND and proj also fall, it is **memory loss**.

## Panel A: the hard-threshold bed (G9's), read-amplitude sweep

ρ ∈ {0.125, 0.5, 1, 2, 4, 8}, K = 8 consecutive physical reads per test storm. The hard bed
does not move between exposures, so the **unpinged control at the same final time** is the
storm bed read once (read 1 with fresh noise); it is reported as `ctrl`.

## Panel B: the sliding-threshold bed (G7's law, fixed rate)

d s = −r(σ − θ), dθ = (σ − θ)/T, r = the storm's scour rate (fixed, never normalised).
T ∈ {∞, 48 substeps (one read)}. An always-on ambient sea (west, ω 0.20, outside the read band,
peak stress σ_c on the calibration bed) is part of the medium; θ starts habituated to it, so
the pristine bed is at rest (checked: `g10_check.py`). Storms are written with the same law.
ρ ∈ {0.125, 0.5, 2}, K = 8. **Unpinged control**: K − 1 silent slots (48 ambient-only
substeps each, the length of a read), then one read at slot K, against a pristine bed given
the same silent slots. Calibration storms are written in this medium too.

## Panel C: teaching after a storm (hard bed)

The same test storms as panel A. After the storm, one 20-substep episode, scour writer at the
fixed rate, forward swell = G1's swell A (west, ω 0.40, amplitude 1, peak ≈ 10 σ_c, the same
strength as the storms). Task (G5/G7's): quiet harbourN, J = ½|y|².

| arm | what moves the sand |
|---|---|
| none | nothing |
| expose | forward swell only (**matched exposure, no teaching target**) |
| teach | forward swell + physical teacher τ = A⁻¹(harbourN · β·y), one exposure, no subtraction |
| ideal_full | the exact task gradient, each step's L2 = teach's applied step L2 |
| ideal_diff | the exact task gradient, each step's L2 = ‖d_teach − d_expose‖ on teach's own state (the size of the target-specific part of teach's write) |

β ∈ {0.03 (G5), 0.3}. Each arm also runs on the pristine bed (matched-history baseline). Read
afterwards at ρ = 0.125 (writes nothing on this bed; checked), decoded by panel A's frozen
ρ = 0.125 decoder. Learning = J falls more under teach than under expose.

## Predictions

*Draft: predictions and pass rules are added from the dev-bed (960) results before any held-out run. Until that commit lands, nothing on beds 1010–1011 has been run.*
