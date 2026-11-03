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
fixed rate, forward swell = G1's swell A (west, ω 0.40). Two forward strengths: amplitude 1
(peak ≈ 10 σ_c, the same strength as the storms) and amplitude 0.45 (stress × 0.2, peak ≈ 2 σ_c,
the level panel A's dev found nearly harmless to read at). Task (G5/G7's): quiet harbourN,
J = ½|y|².

| arm | what moves the sand |
|---|---|
| none | nothing |
| expose | forward swell only (**matched exposure, no teaching target**) |
| teach | forward swell + physical teacher τ = A⁻¹(harbourN · β·y), one exposure, no subtraction |
| ideal_full | the exact task gradient, each step's L2 = teach's **applied** step L2 |
| ideal_diff | the exact task gradient, each step's L2 = ‖applied(d_teach) − applied(d_expose)‖ on teach's own state (the size of the target-specific part of teach's write) |

Ideal arms stop writing once J ≤ 0.05 J₀ (so they do not wander after the task is solved).
β ∈ {0.03 (G5), 0.3}, crossed with the two forward strengths: 4 conditions. Each arm also runs on the pristine bed (matched-history baseline). Read
afterwards at ρ = 0.125 (writes nothing on this bed; checked), decoded by panel A's frozen
ρ = 0.125 decoder. Learning = J falls more under teach than under expose.

## What dev bed 960 showed (150 calibration / 30 test storms; panel B 100 / 20)

Medians, cells (chance ≈ 17–20). Small test sets: ±1.5 cells is noise-draw variation on dev.

**A, hard bed.** ρ ≤ 0.5 writes nothing; ρ = 1 writes 5e-4 per read, ρ = 8 writes 0.03.
Frozen read 1 → read 8: ρ ≤ 2 flat (8.0–9.5); ρ = 4: 9.3 → 10.2; **ρ = 8: 9.3 → 17.9**.
Material proj₈: 1.000 (ρ ≤ 1), 0.998 (2), 0.934 (4), **0.772 (8)**. The SAND reader at
ρ = 8 still gets 7.0 (5.9 at read 1). The efference baseline does **not** rescue ρ = 8 (19.0).
Smoke run (bed 961, K = 3): a recalibrated reader at ρ = 8 got 12.9 against the frozen 27.0.
At 10 % noise, averaging 8 free reads at ρ = 0.125 helps a little (10.0 → 8.9).

**B, sliding bed.** T = 48: the **unpinged** record vanishes by itself (control proj 0.002,
frozen 31.8 vs chance 19.9); reads are irrelevant. T = ∞: the ambient sea erodes the record
too (control proj 0.73); weak reads (ρ 0.125) cost no more than time (16.8 vs control 17.0,
proj 0.83 vs 0.73); strong reads (ρ 2) cost more (proj 0.64, frozen 28.0 vs control 19.9).

**C, teaching.** Amplitude 1: matched exposure alone takes the frozen read 9.3 → 16.7 while
the sand keeps proj 0.92 (SAND 6.0); teach β 0.03 erases more (SAND 16.0, proj 0.84) and
learns little beyond exposure (J/J₀ 2.56 vs 2.89); β 0.3 erases everything (proj 0.00).
**Ideal** gradient learning reaches J/J₀ ≈ 0.01 with ~1 unit of L2 and keeps the record
(read 8.2, proj 0.94). Amplitude 0.45: exposure writes almost nothing (L2 0.10, read 9.5);
teach β 0.03 lowers J to 0.89 (exposure 1.00) with the record intact (read 10.2, proj 1.00);
teach β 0.3 is itself a storm (proj 0.00).

## Held-out runs

Beds **1010 and 1011**. Panel A and C: 400 calibration / 150 test storms, K = 8 (A also reads
its calibration storms K times at ρ 4 and 8 for the recalibrated reader). Panel B: 200 / 60.
Run with `run_g10.sh`, analysed with `g10_analyse.py` (one JSON per bed).

## Predictions (each must hold on **both** held-out beds)

| | prediction | pass rule |
|---|---|---|
| P10a | below threshold, reading is free | A, ρ ∈ {0.125, 0.5}: every read writes exactly 0, proj₈ median ≥ 0.9999 (float32 storage), and frozen read-8 error = read-1 error ± 1 cell |
| P10b | the threshold is where reading starts to cost | A: proj₈ median ≥ 0.99 at ρ = 2 and ≤ 0.90 at ρ = 8 |
| P10c | loud reads blind a frozen reader | A, ρ = 8: frozen read-8 error ≥ 1.5 × read-1 error and ≥ 0.75 × chance |
| P10d | ...while most of the record is still in the sand | A, ρ = 8: SAND at read 8 ≤ 1.5 × SAND at read 1 and ≤ 0.5 × chance |
| P10e | an efference baseline does not rescue it (the lens changed, not an offset) | A, ρ = 8: eff read-8 error ≥ 0.75 × chance |
| P10f | the record is still in the reads: a recalibrated reader recovers it | A, ρ = 8: recalibrated read-8 error ≤ 0.6 × frozen read-8 error |
| P10g | a homeostatic bed forgets with nobody reading | B, T = 48: unpinged control proj ≤ 0.1 and control frozen error ≥ 0.9 × chance |
| P10h | with a fixed habit, weak reads cost no more than time | B, T = ∞, ρ = 0.125: proj₈ ≥ control proj − 0.1, and frozen read-8 ≤ control + 3 cells |
| P10i | ...strong reads do | B, T = ∞, ρ = 2: proj₈ < control proj, and frozen read-8 > control |
| P10j | storm-strength exposure blinds the frozen reader without erasing the sand | C, amp 1, expose: frozen read ≥ 1.5 × none, SAND ≤ 1.5 × none, proj ≥ 0.85 |
| P10k | the physical teacher erases more than its matched exposure | C, amp 1, β 0.03: teach SAND > expose SAND and teach proj < expose proj |
| P10l | an ideal learner solves the task and keeps the record | C, amp 1, β 0.03, ideal_diff: J/J₀ ≤ 0.05, proj ≥ 0.9, frozen read ≤ 1.3 × none |
| P10m | **readable while learning, physically**: at sub-storm strength the teacher learns beyond exposure and the record survives | C, amp 0.45, β 0.03: teach J/J₀ < expose J/J₀ by ≥ 0.05, teach beats expose on ≥ 60 % of storm beds, teach proj ≥ 0.95, frozen read ≤ 1.3 × none |
| P10n | a strong teacher is itself a storm | C, amp 0.45, β 0.3: teach proj ≤ 0.3 |

**Registered reading of the central question.**
- If P10m passes: **yes, in a narrow window.** The beach can stay readable while learning,
  when writing stays near threshold; it learns slowly there. Destruction is a property of
  non-selective writing at storm strength, not of learning (P10l, P10k).
- If P10m fails on the learning half but P10l passes: **learning and reading are compatible
  in principle, but this physical teacher cannot do it**: at the strength where it learns, it
  erases (P10k, P10n).
- If P10l fails too: **no**: on this bed even ideal learning destroys the record.
- Panel B is reported alongside: a medium that can re-habituate (the precondition G7 found
  for learning without subtraction) forgets its records with nobody reading (P10g).

All three panels report every arm, including those without pass rules. No threshold above is
changed after the held-out run; failures are reported as failures.

Do not hype. Do not lie. Just show.
