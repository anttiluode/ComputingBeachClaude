# Pre-registration: G9 (is a self-written medium worth more than stored weights?)

Written and pushed before the held-out beds (1000–1002) were run. Dev beds 950–951 were used
to build the instruments; what they showed is stated below.

## Why

Sol's closing line (2 Nov 2026): *"The next thing to prove is that making those operators
physically self-modifying gives us something useful that we couldn't get as cheaply by storing
and updating ordinary weights."*

G1–G8 already answer half of it. As a **learner**, the beach loses to ordinary weights: teacher
waves die under loss (G5, G8a), the sliding threshold goes stale (G7), and trial-and-keep needs
an agent (G8b). As a **memory of symbols**, it loses too: order of A and B is one bit, which a
digital counter with one decay stores perfectly in 2 numbers, while the floor's 3112 cells read
at 0.70–0.75 (G4).

So the only place left to look is where ordinary weights have a cost the medium doesn't pay. A
digital weight is updated by a measurement: something has to be **on and watching** when the
event happens, and it can only store **what its sensors heard, in the band they heard it**. The
sand is written by the event itself, with nothing on. And what it stores is an operator change
that can be **interrogated later in any band, from any port, as often as you like**. G9 tests
whether that is worth anything, against ordinary-weight competitors that get every fair advantage,
including a physics model of the basin.

## G9a: where did the storm hit?

**Storms** (`beach/watch.py`): a steady forcing, Gaussian blob σ 2.5 cells, at a random interior
cell (≥ 5 cells from the wall, 2276 candidates), at one of the 16 read frequencies, strength
m ∈ [0.7, 1.4] (log-uniform) × swell A's peak stress, split over tones. Each storm hits a fresh
copy of the bed and writes it with G1/G4's strongest scour (s = 0.04, 10 feedback substeps,
damping 0.006, Q ≈ 67). Task: regress the storm's (row, col). Score: median Euclidean error in
cells on the last 20 % of storms. Chance: predict the training mean (≈ 19–20 cells).

**Memories**, each with 1 % RMS noise on every sensor reading:

| arm | what it is | on during the storm |
|---|---|---|
| SAND | the bed change itself (upper bound; nobody can read it directly) | nothing |
| BEACH5 | weak pings after minus before the storm: probes west/north/south, heard at all 5 ports, 16 freqs, reciprocal duplicates removed (384 reals) | **nothing** |
| BEACH2 | G1's read: probe south, heard at the 2 harbours (64 reals) | nothing |
| WATCHk | **ordinary weights, frozen basin**: k = 1, 2, 5 port sensors hear the storm in a basin whose bed does not move; per frequency bin, Hebbian outer-product memory C += x xᴴ (stored as log trace + C/trace), binned or pooled across bins | k sensors |
| WATCH5d | hybrid: the same, but listening while the bed is being dug | 5 sensors |
| DENSEk | frozen basin, k = 10, 20, 40, 64 sensors (5 ports + fixed random interior cells), rank-1 memory as the phase-referenced vector | k sensors |
| BEACH5_f8, BEACH5_fs | mechanism diagnostics: BEACH5 read at ONE fixed frequency (index 8), or only at the storm's own frequency | nothing |

**Decoder**, identical for every data-driven arm: standardise, then choose ridge / kNN / RBF kernel
ridge (grids in `g9_analyse.py`), and for watchers also binned vs pooled representation, by 5-fold
CV on the training storms only. Big-watcher curve (`g9_watchbig.py`) also tries a 256×256 MLP.

**Model-based competitor** (`g9_mfp.py`): a digital watcher that has a **model of the basin**
and localizes with a Bartlett matched-field processor, no training storms at all. Models: exact
bed; true bed + smooth error ε ∈ {0.01, 0.03, 0.06, 0.1} (the bed's roughness sd is 0.06);
a flat bed at the mean; a different random bed; a flat bed with the storm assumed point-like.
k ∈ {2, 5, 20, 64} sensors.

### Dev (beds 950–951)

Bed 950, 500 single-tone storms, 16 frequencies, n_train = 400 (median error, cells):

| SAND | BEACH5 | BEACH2 | WATCH5 | WATCH2 | WATCH1 | WATCH5d | DENSE20 | DENSE64 | BEACH5_f8 | BEACH5_fs | chance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.3 | **6.0** | 7.4 | 17.2 | 17.6 | 19.7 | 12.0 | 16.0 | 14.1 | 8.2 | 19.9 | 20.1 |

- Frozen-basin watcher, 5 ports, idealised, more data: 12.6 cells at 6400 storms (MLP 12.1).
- One storm frequency only (bed 950): BEACH5 4.5, WATCH5 12.5, DENSE64 6.2.
- Weak writes s = 0.01 (bed 951): BEACH5 7.0, WATCH5 15.0.
- Unseen storm frequencies (train on 12, test on 4 interleaved): BEACH5 7.8 → 7.1.
- 4-tone storms (bed 951, 300 storms, n = 200): BEACH5 6.5, WATCH5 16.5, WATCH5d 14.5.
- **Matched-field with a model**: exact model, 2 ports: 0.0 cells. 5 ports, single tone: exact
  0, ε 0.03 → 1.0, ε 0.06 → 18.7, flat bed 15.7–16.6, other bed 17–23. 20 sensors: ≤ 3.2 with
  every model including flat. 4-tone storms, 5 ports, flat model: 1.0.

So on dev the honest shape is already visible: the beach beats every **data-driven** ordinary-weight
memory by ~3×, including 64 always-on sensors, but a digital watcher with a **good model** of the
basin, or with 20 interior sensors, or with broadband storms, beats the beach.

### Held-out runs

| run | beds | storms | condition |
|---|---|---|---|
| H1 | 1000, 1001, 1002 | 1000 each | single tone, 16 frequencies, s = 0.04 |
| H2 | 1000 | 1000 | one storm frequency only (index 8) |
| H3 | 1001 | 1000 | weak writes, s = 0.01 |
| H4 | 1002 | 600 | 4-tone storms |
| H5 | 1000 | 17 000 | idealised frozen-basin watcher only, learning curve to 16 000 |

Learning curves at n_train ∈ {100, 200, 400, 800}; test = last 200 storms.

### Predictions

| | prediction | pass rule |
|---|---|---|
| P9a | memory without a watcher beats ordinary weights fed by the same ports | H1: BEACH5 ≤ 0.6 × WATCH5 at n = 400 and n = 800, all 3 beds |
| P9b | the bed is worth many always-on sensors (data-driven) | H1: BEACH5 < DENSE64 at n = 400 on ≥ 2 of 3 beds |
| P9c | more data doesn't rescue the 5-port watcher | H5: best watcher at 16 000 storms > BEACH5 at n = 400 (bed 1000) |
| P9d | the mechanism is interrogation in a fixed basis | H1: BEACH5_fs ≥ 2 × BEACH5_f8 at n = 400, all 3 beds |
| P9e | the record is invariant to the storm's frequency | H1: BEACH5 on unseen storm frequencies ≤ 1.25 × on seen ones, all 3 beds |
| P9f | not an artefact of a heavy write | H3: BEACH5 ≤ 0.6 × WATCH5 at n = 400 and 800 |
| P9g | a perfect model kills the advantage | H1: matched field, exact model, 5 ports ≤ 1 cell, all 3 beds |
| P9h | the model has to be good | H1, 5 ports: flat-bed model error ≥ BEACH5 (n = 800) AND ε = 0.03 model ≤ BEACH5, all 3 beds |
| P9i | or the sensors have to be inside | H1: 20 sensors, flat-bed model ≤ 3 cells, all 3 beds |
| P9j | broadband storms remove the niche | H4: 5 ports, flat-bed model ≤ BEACH5 at n = 400 |
| P9k | one storm frequency helps the watcher but not enough | H2: WATCH5 lower than in H1 bed 1000 at n = 400, and BEACH5 ≤ 0.6 × WATCH5 |

**Registered verdict rule for Sol's question.**
- If P9a fails: **no**. The self-written bed is not better than ordinary weights even data-driven.
- If P9a and P9b pass but the flat-model part of P9h fails (a crude model plus 5 ports already
  beats the beach): **only against straw men**.
- If P9a, P9b and P9h pass: **yes, in a bounded regime**: when nothing can watch, the medium can't
  be modelled to within about half its own roughness, and the events are narrowband. The bounds
  are P9g, P9i and P9j.

## Cost ledger (filled in from the runs)

Per storm, for each memory: sensors powered during the event, numbers stored, digital
operations per event, calibration storms needed, model needed.

Do not hype. Do not lie. Just show.
