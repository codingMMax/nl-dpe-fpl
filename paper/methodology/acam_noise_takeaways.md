# ACAM / Crossbar Noise — Key Takeaways

One-pager summary of the noise-analysis work, for advisor review.
Full derivations in `acam_crossbar_noise_analysis.md`; tooling in
`paper/scripts/acam_crossbar_snr.py`.

---

## TL;DR

Within our DSE range (R = 128 – 1024), **output precision is limited
by the ACAM's 8-bit quantizer, not by analog noise**. The R-axis of
the DSE can be chosen for throughput / density reasons without a
fundamental precision trade-off, until IR drop cliffs in around
R ≈ 2048.

---

## Scope

Three ACAM usage modes; only two need study:

| ACAM mode                        | Noise behaviour                                                                 | In scope? |
|----------------------------------|---------------------------------------------------------------------------------|-----------|
| log ↔ exp (attention core)       | paired inverses self-cancel; softmax scale-invariance absorbs the residual       | **no** — assume ideal after NAF |
| ADC / identity (FC, Q/K/V, FFN)  | output feeds CLB reduction, noise amplified by √(V·H)                          | **yes**    |
| activation (ReLU / GELU inline)   | kink sensitivity — sign flips near zero                                         | **yes** (secondary) |

---

## The physics, in one breath each

- **σ_ideal ∝ √R** — independent cell-to-cell errors add in quadrature
  (central limit theorem). Adding rows *improves* relative noise.
- **σ_correlated ∝ R** — driver / supply / thermal noise that hits all
  rows coherently. Adds linearly. In our model, σ_corr / LSB is
  R-independent because LSB also scales with R.
- **IR drop ∝ R²** — row-wire resistance × row-wire current, both ∝ R,
  so their product is R². **The only effect that scales faster than
  the ACAM's LSB-per-R growth**, and it behaves like a cliff, not a
  slope.
- **Sneak path ∝ √R** — same family as σ_ideal (independent OFF-cell
  variance); negligible for 2T2R ACAM cells.

---

## The counter-intuitive result

Under ideal physics, **more rows = better SNR**, because signal grows
with R while independent noise only grows with √R. This is the same
reason long camera exposures produce cleaner images. The
"big crossbar is noisy" intuition in the IMC literature traces back
to **IR drop specifically**, not to noise.

---

## What the NL-DPE paper (arXiv:2511.13950) gives and doesn't give

**Gives:**
- Per-cell σ_G = 0.4 μS peak
- ACAM: 130 rows → 8-bit capability (Gray-coded)
- Noise-Aware Fine-tuning (NAF) recovers <1 % accuracy loss
- Empirical precision cliff: 7-bit catastrophic, 8-bit fine

**Doesn't give:**
- A row-count scaling formula. The paper's model is per-cell and
  input-independent; array-level R-dependence was not analysed.
  This is exactly the gap our experiment would fill.

---

## Answers to 5.1, 5.2, 5.3

| Question                                | Answer                                                                                           |
|-----------------------------------------|--------------------------------------------------------------------------------------------------|
| **5.1** Crossbar rows → precision        | Ideal physics: improves as √R. Non-ideal: cliff from IR drop at some R (process-dependent, not a gradual slope). |
| **5.2** ACAM rows → precision            | 130 rows Gray-coded = 8-bit capability (paper Table I). ACAM-σ alone supports >8 bits — not the binding constraint. |
| **5.3** Precision requirement & training | 8-bit needed; 7-bit is catastrophic. NAF absorbs static per-cell σ. Within DSE range, precision is ACAM-capped, so a single NAF pass should hold. Per-R retraining may help if the cliff moves into range. |

---

## What's in the repo

- `paper/methodology/acam_crossbar_noise_analysis.md` — full methodology
  with derivations, coefficient derivations, and experiment plan.
- `paper/methodology/acam_noise_takeaways.md` — this doc.
- `paper/scripts/acam_crossbar_snr.py` — SNR projection tool with
  physically-derived defaults (α = 3e-7, σ₀ = 0.05 nA/cell,
  δ = 0.05 nA/row). Sweep R with `--project`; override via CLI flags.

---

## What's open / pitch for next

1. **Calibrate α, σ₀, δ against SPICE** of the actual NL-DPE layout.
   The cliff location in real silicon depends entirely on α, which
   depends on the metal stack.
2. **Inject the paper's noise model into `azurelily/IMC/`** and run
   accuracy-vs-R on BERT-Tiny / VGG / LeNet under NAF. This produces
   the publishable 5.1 / 5.3 answer.
3. **Confirm NAF envelope holds across R** — per-R retraining may or
   may not be needed. Empirical question.

---

## One slide, if asked

- Under realistic process assumptions, precision does **not** degrade
  gradually with R in our DSE range — it stays 8-bit ACAM-capped.
- The only R-dependent degradation mechanism that matters is
  **IR drop**, and it behaves as a cliff.
- Paper's 5.1 answer: "precision depends on IR-drop cliff location,
  which is process-specific and needs silicon calibration."
- Experiment to nail this down is scoped in the methodology doc; a
  SPICE calibration of α is the long pole.
