# PREREG G11: dark sand, or a scrambled index?

Written after the dev beds (1200–1215, and 1200/1201 within-bed) and **before** any held-out run.

## The question

G1/G4 left one wound open: the sand stores AB vs BA order (sand decoder 0.98) but weak pings
read it only at 0.70–0.80. Two explanations compete:

- **Dark sand** (Claude, in conversation, Nov 4): the order lives in sand directions the
  pings' sensitivity kernels barely touch. It is stored but outside the beach's index.
- **Scrambled index** (README G1, Nov 2): the order *is* in each bed's echo, but every bed
  files it under a different address, so a reader trained on other beds can't find it.

## The instrument

A weak ping read y depends on the sand through a kernel: to first order dy = J ds, with
`J_pqw(i) = (g/2) Σ_f∋i (∇λ_q)_f (∇η_p)_f` (ping field × listener field, `beach/dark.py`).
The rows of J are the beach's index. Reads are G9's BEACH-5 family (3 probes, 5 listeners,
12 complex pairs) in two bands, LOW 0.30–0.55 and HIGH 0.60–0.95, 16 frequencies each.
Storms: G4's shared benchmark (A west 0.40, B north 0.47 × ampB, scour s = 0.04, 10 feedback
substeps each).

Instrument checks (`g11_run.py check`, pass): J vs central differences, rel. err 8e-7 (low),
2e-6 (high).

**A correction found by the check.** In conversation I proposed reading the order with an
interaction of two pings, y_pq − y_p − y_q + y_0. In this linear medium that is identically
zero (measured 2.4e-15): two weak pings superpose exactly. It can be nonzero only if a ping
writes, which is the write-to-read pilot that already failed (0.77). That proposal is withdrawn.

## Dev values (beds 1200–1215, low band unless stated)

| measure | dev |
|---|---|
| order SNR per bed (‖Δy‖ over noise of two 1 % reads) | 28 |
| gain ‖JΔ‖/‖Δ‖ relative to the common write | 0.385 |
| mean wavenumber, order / common write | 0.182 / 0.114 cycles per cell |
| transplant: cos(J_i Δ_i, J_j Δ_i), bed i's order through bed j's index | 0.19 (high band 0.105) |
| same index, other bed's order: cos(J_i Δ_i, J_i Δ_j) | 0.75 |
| Born rel. err, order / one G9 storm | 0.37 / 1.88 |
| across-bed LOO decoding, low / high / sand | 0.63 / 0.69 / 0.94 |
| within-bed decoding (40 jittered pairs, ±0.015 in ω, ×0.9–1.1 in amplitude) | 0.925, 0.925 |

## Predictions (held-out: across 1300–1347, within 1300–1307 × 40 pairs)

- **P11a, not dark.** Median order SNR (low) ≥ 10.
- **P11b, dimmer and finer.** Gain ratio order/common (low) ≤ 0.6, and order's mean
  wavenumber ≥ 1.3 × the common write's.
- **P11c, the index scrambles, the content doesn't.** Median transplant cos (low) ≤ 0.4 and
  median same-index-other-order cos (low) ≥ 0.6.
- **P11d, a bed-calibrated reader works.** Within-bed decoding ≥ 0.85 on at least 6 of 8 beds.
- **P11e, an across-bed reader fails.** Across-bed LOO decoding (low) ≤ 0.80.
- **P11f, the first-order index holds for the order, not for storms.** Born rel. err
  order ≤ 0.5 and storm ≥ 1.0 (low).
- **P11g, finer waves scramble more.** Transplant cos high < transplant cos low.

Reads must write nothing: worst probe stress < 1 σ_c (reported, all reads).

**Verdict rule.** Dark sand is rejected if P11a passes. The scrambled index is supported if
P11c, P11d and P11e all pass.

Decoder: G1's ridge LDA, leave one bed (across) or one jittered pair (within) out, 1 % read
noise. Seeds of storms and jitter are fixed in `g11_run.py`.
