# ACAM / Crossbar Noise Analysis — Precision vs Architecture Size

**Scope.** Grounding for the paper's §5 noise-modelling questions:
- **5.1** How do crossbar rows affect output precision?
- **5.2** How do ACAM rows affect output precision?
- **5.3** What is the precision requirement to maintain model accuracy
  as crossbar rows grow, and can this be compensated by training the
  ACAM differently?

Primary references:
- Zhao et al., *NL-DPE: An Analog In-memory Non-Linear Dot Product
  Engine for Efficient CNN and LLM Inference*, arXiv:2511.13950.
- Zhao et al., *Noise Aware Finetuning for Analog Non-Linear Dot
  Product Engine*, OpenReview id `hvimyM6JOe`.

---

## 0. Architectural context

Both NL-DPE papers follow the same design: the ACAM **replaces the
ADC and the digital activation** in one step. Each ACAM implements a
decision tree that maps column current directly to an activated
int8 output. There is no separate "ADC mode" in the paper proper —
every ACAM is configured for some nonlinear function, which may be
ReLU/GELU/sigmoid/log/exp/tanh/identity, etc.

For our architecture we distinguish three usage modes and the noise
effect of each is qualitatively different:

1. **ACAM as ADC-like quantizer** — identity-mapping ACAM feeding CLB
   reduction + CLB activation. Noise on the ACAM output is additive
   and gets amplified by the CLB adder tree (√(V·H) scaling).
2. **ACAM as activation** — ACAM implements ReLU/GELU/... directly.
   Noise is point-wise (no reduction amplification) but
   operating-point-sensitive (worst near ReLU's kink).
3. **ACAM as log / exp** — paired-inverse structure in the attention
   core. Multiplicative noise self-cancels through the log↔exp pair;
   softmax adds scale-invariance as a second defence.

Modes 1 and 2 are the objects of noise study; mode 3 can be treated
as ideal-int8 post-NAF within the R range we care about.

---

## 1. What the Pedretti papers give us

### 1.1 Per-cell conductance noise model

The paper models conductance error as the sum of programming and
read-fluctuation terms:

```
G = G_target + G_write + G_read                             (Eq. 6)
σ_x = exp( a_x · log( G.clip(0, c_x) ) + b_x )              (Eq. 5)
```

`σ_x` grows log-linearly with `G` and saturates at high conductance.
Both `G_write` and `G_read` are drawn from zero-mean Gaussians with
this σ.

### 1.2 ACAM threshold transfer

Each ACAM cell's decision threshold is itself a programmed ReRAM
conductance with its own variance:

```
TH = exp( a_ACAM · log(G) + b_ACAM ) + c_ACAM               (Eq. 7)
```

### 1.3 Device parameters (TaOx RRAM, from the paper's fab data)

| Parameter                    | Value                                    |
|------------------------------|------------------------------------------|
| Conductance range            | 0.01 μS – 150 μS                          |
| Programming σ (peak)         | ~0.4 μS                                   |
| Programming tolerance        | ±0.55 μS for targets > 1 μS; proportional for smaller |
| ACAM cells per column        | 130                                       |
| Output bit-width (Gray-coded)| 8-bit, achievable in 2^(n-1) = 128 rows   |
| ACAM cell area               | 0.72 μm²                                  |
| Search energy / latency      | 0.44 fJ / 300 ps                          |

### 1.4 Noise-Aware Fine-tuning (NAF) procedure

Four sequential steps:

1. **Inject crossbar noise** (Eq. 6) into weights and run 10 iterations
   of end-to-end fine-tuning with analog-slicing loss:
   ```
   Loss = MSE(y, ŷ) + λ₁·‖W‖∞ + λ₂·‖ε‖∞        (Eq. 8)
   ```
   L∞ regularisation pushes weights toward smaller conductances
   (lower noise regime).
2. **Convert non-linear ops** (softmax, depth-wise multiply) to
   single-variable functions (log, exp).
3. **Train independent decision trees** per output bit via
   scikit-learn on 5,000 sampled inputs.
4. **Fine-tune each DT** using a differentiable ACAM approximation
   (paper's Algorithm 1) with injected ACAM noise. Fewer than 10
   epochs typically sufficient.

Key claim: **no post-deployment in-device fine-tuning** needed.

### 1.5 Empirical precision cliff

Table III / Fig. 14 of the paper:

| Precision | Accuracy impact                                    |
|-----------|----------------------------------------------------|
| 6-bit     | Catastrophic                                        |
| 7-bit     | >50% accuracy drop                                  |
| 8-bit     | <1% loss **(operating point)**                      |
| 9-bit     | <1% loss with <1% energy increase                   |
| End-to-end w/ NAF | <1% loss                                    |
| End-to-end w/o NAF | 40–60% loss (ACAM noise alone)             |

### 1.6 Gap this analysis fills

The paper's noise model is **per-cell and input-independent**. It
does not provide an explicit scaling law from crossbar rows R to
column-current σ or to effective precision — only a per-cell σ that
does not depend on array size or input activity. The paper also
does not run an accuracy-vs-R experiment.

Our DSE sweeps R across {128, 256, 512, 1024}. To make claims about
this range we need to (a) start from ideal physics using the paper's
per-cell σ, (b) layer in non-ideal effects (IR drop, sneak paths,
correlated noise) whose scaling depends on layout, and (c) measure
the crossover in simulation + accuracy experiments. §2 establishes
the ideal scaling; §5 lays out the measurement plan.

---

## 2. Crossbar rows R → output precision (Question 5.1)

### 2.1 Physical model

Column current is the analog sum over R row cells:

```
I_col = V_read · Σ_{i=1..R} g_i · x_i
```

Noise sources and their R-scaling:

| Source             | Scaling (independent errors) | Scaling (worst case) |
|--------------------|------------------------------|-----------------------|
| Thermal (Johnson)  | √R                           | √R                    |
| Shot noise         | √R                           | √R                    |
| Programming σ_G    | √R                           | √R                    |
| Read fluctuation   | √R                           | √R                    |
| IR drop            | grows faster than √R         | ~R (at tail row)      |
| Sneak paths        | grows faster than √R         | super-linear          |

Ideal-case column current SNR scales as R / √R = √R: signal grows
with R, independent noise with √R, so SNR **improves** with R in
ideal physics. This is the classic "analog crossbar SNR gain with
row count" result in the literature. Large R does *not* degrade
precision by itself — the degradation (if any) comes from the
non-ideal effects (IR drop, sneak paths, correlated noise) which
grow faster than √R.

### 2.2 Worked numerical example

Using paper parameters: σ_G ≈ 0.4 μS, g_max = 150 μS, V_read = 0.3 V.
Bit-sliced input (each row driven to 0 or 0.3 V). ACAM thresholds
programmed via Eq. 7 to span the expected I_col range in 128 bins.

**Case R = 128:**
- Per-cell max current: g_max × V_read = 150 × 0.3 = **45 nA**
- Per-cell σ (current): σ_G × V_read = 0.4 × 0.3 = **0.12 nA**
- I_col_max = 128 × 45 = **5,760 nA**
- LSB = I_col_max / 128 = **45 nA**
- Cumulative σ_I = √128 × 0.12 = **1.36 nA**
- **σ_I / LSB = 0.030 LSB** — well below 8-bit noise floor

**Case R = 1024:**
- Per-cell max current: same, **45 nA** (V_read fixed per bit-slice)
- Per-cell σ: same, **0.12 nA**
- I_col_max = 1024 × 45 = **46,080 nA**
- ACAM re-programmed → LSB = 46,080 / 128 = **360 nA** (grows with R)
- Cumulative σ_I = √1024 × 0.12 = **3.84 nA**
- **σ_I / LSB = 0.011 LSB** — *cleaner* than R=128

**Result: σ_I / LSB improves by √8 ≈ 2.83× going R=128 → R=1024**,
because signal scales with R while independent noise scales with √R,
and the ACAM's LSB tracks the signal range.

### 2.3 Where the "bigger R is noisier" intuition actually comes from

Ideal physics predicts SNR gain with R. The commonly-cited "big
crossbar hurts precision" intuition is real but is driven by
**non-ideal effects** that grow faster than √R:

| Non-ideal effect        | Scaling                    | Consequence                             |
|-------------------------|----------------------------|-----------------------------------------|
| IR drop on row wire     | ~R² at tail cell           | Tail cells see reduced V_read, systematic bias |
| Sneak-path leakage      | ~R × OFF-state g           | At R=1024, ~10 μS of leakage vs 150 μS g_max |
| Correlated cell σ       | Can approach R (not √R)    | Ideal √R-averaging breaks down          |
| Line-capacitance delay  | Grows with R               | Slower settling, more read-time noise   |
| Column readout saturation | Hardware-fixed            | At I_col_max ≈ 46 μA (R=1024), sense amp may clip |

The actual R at which these effects overtake the √R ideal SNR gain
is **process- and layout-dependent**. Without measurement, we do
not know whether it sits at R=256, R=512, R=1024, or beyond.

### 2.4 What this means for 5.1

The paper's per-cell noise model alone **does not predict precision
degradation with R** — in fact, it predicts improvement. The
experiment should measure:

1. Whether ideal √R SNR gain holds across R ∈ {128, 256, 512, 1024}
   when the full noise model (programming + read-fluctuation +
   ACAM-threshold σ) is injected without non-ideal effects.
2. How much IR drop and sneak-path contributions erode the ideal
   gain when modelled alongside — i.e., at what R does the crossover
   happen where non-ideal effects dominate.
3. How accuracy tracks effective precision across this range.

Only the measured crossover — not an a-priori scaling formula — can
produce a publishable number for §5.1.

---

## 3. ACAM rows → output precision (Question 5.2)

### 3.1 Row budget vs output precision

From the paper: for Gray-coded decision-tree output of n bits, the
ACAM requires **2^(n-1) rows in the worst case**. Binary encoding is
~2× worse.

| Output precision | ACAM rows (Gray) | ACAM rows (binary) |
|------------------|------------------|--------------------|
| 6-bit            | 32               | 63                 |
| 7-bit            | 64               | 127                |
| **8-bit**        | **128**          | 248                |
| 9-bit            | 256              | 495                |

NL-DPE chooses **130 rows = 8-bit capability + 2 rows of slack**.
This is the worst-case budget across all activation functions the
paper evaluated (sigmoid, exp, tanh, GELU, log, identity).

### 3.2 Why not more rows?

| Row count change | Effect                                                |
|------------------|-------------------------------------------------------|
| ×2 rows          | +1 bit precision                                       |
| ×2 rows          | +1 programming σ surface per threshold → more bin-cross risk |
| ×2 rows          | ×2 search latency at fixed frequency                   |
| ×2 rows          | ×2 area (0.72 μm² per cell)                           |
| ×2 rows          | ×2 NAF training effort (one decision tree per output bit) |

Each doubling buys ~1 bit of nominal precision but the σ budget
across threshold boundaries gets correspondingly tighter. Effective
bit gain saturates: 256 rows may only deliver ~8.5 effective bits.

### 3.3 Effective precision from ACAM σ alone

With σ_prog = 0.55 μS over a 150 μS range, the ACAM can discriminate
~150/0.55 ≈ 273 distinguishable threshold levels in principle.
Since 273 > 128, **8-bit resolution is achievable at the ACAM itself**
for the 130-row configuration. The precision floor is therefore not
the ACAM's row count but the **crossbar column-current noise**
covered in §2.

### 3.4 Row count design conclusion

**130 rows is close to the architectural optimum** for an 8-bit
output target under the paper's device parameters. More rows give
diminishing returns; fewer rows cap out below 8 bits directly.

---

## 4. Precision requirement and training compensation (Question 5.3)

### 4.1 The precision-vs-R map is undetermined without measurement

Per §2, ideal physics predicts **precision improves with R** as √R
SNR gain. Non-ideal effects (IR drop, sneak paths, correlated σ)
can flip the sign at some R, but the crossover is
process-dependent and not predictable from the paper's per-cell
model alone.

What we *can* pin down from §1.5 is the **accuracy cliff on
effective precision**:

| Effective precision at ACAM output | Post-NAF accuracy regime (from paper) |
|------------------------------------|----------------------------------------|
| ≥ 8 bits                           | <1% loss                                |
| 7 bits                             | >50% loss                               |
| < 7 bits                           | Catastrophic                            |

What we **cannot** pin down without our experiment: the mapping
from R to effective precision, since this depends on whether
non-ideal effects dominate the ideal √R SNR gain at that R.

So the honest form of the 5.3 answer is two-part:

- **Precision requirement**: ≥ 8 bits effective at the ACAM output
  for <1% accuracy loss. 7 bits is already the catastrophic regime.
- **R-dependence**: undetermined from paper; measurement required.
  Ideal physics says R does not drive degradation — the cliff is
  driven by effective precision, which tracks (ideal SNR gain) −
  (non-ideal degradation). The question is *where* non-ideal
  effects overtake ideal gain.

### 4.2 What NAF can and cannot compensate

**NAF absorbs** (demonstrated in the paper):
- Static per-cell programming σ, when the σ distribution at training
  matches deployment.
- Cumulative small programming errors across all cells in a column,
  averaged over a training-time sample of noisy realisations.
- Weight-space sensitivity: L∞ regularisation pushes toward weights
  that sit in the low-noise conductance range.

**NAF does not absorb:**
- **Distributional shift across R.** NAF's compensating weights are
  tuned against a specific σ distribution at the ACAM input. If R
  changes at deployment, cumulative σ_I changes in both magnitude
  and possibly shape (IR-drop non-linearity at large R). The network
  is then operating outside the training envelope.
- **Per-sample stochastic noise that exceeds the training envelope.**
  Read fluctuation is uncorrelated with training-time noise; if its
  σ × √R exceeds the envelope NAF was trained for, the accuracy
  floor drops sharply.
- **Sign-preserving errors near activation kinks.** ReLU's derivative
  is 0 below 0, so a noise realisation that flips ReLU(x) from x to 0
  is an information loss NAF cannot recover — the gradient signal
  is absent.

### 4.3 Training strategies that extend the envelope

| Strategy                       | Mechanism                                     | Expected reach           |
|--------------------------------|-----------------------------------------------|--------------------------|
| Single NAF                     | Static σ injection during fine-tuning          | ~7-bit equivalent        |
| Per-R NAF                      | Separate NAF pass per DSE config              | Adds ~0.5 bit            |
| Per-layer NAF                  | Each layer tuned for its tiled R              | Adds ~0.5 bit            |
| Larger conductance range       | ACAM discriminates more levels                | Hardware change, +1 bit  |
| Multi-pass averaging           | Σ of n reads reduces σ by √n                   | +0.5 log₂(n) bit         |
| Redundant ACAM columns         | Majority-vote across k replicas                | +0.5 log₂(k) bit         |
| Lower V_read, more passes      | Shifts the SNR curve at the cost of latency   | Trades area for bits     |

### 4.4 Architectural recommendation (to be calibrated by experiment)

The recommendation sits on where the crossover between ideal √R
SNR gain and non-ideal degradation lands:

- If the crossover is **beyond R=1024**: ideal gain wins across the
  DSE range; standard single-pass NAF is sufficient at all R.
  Big-crossbar designs are pure upside.
- If the crossover is **around R=512**: single-pass NAF works up to
  R=256; per-R NAF recommended at R=512; mitigation (multi-pass,
  redundancy, partial-product decomposition) required at R=1024.
- If the crossover is **around R=256**: non-ideal effects dominate
  early. Smaller crossbars preferred; large-R designs need
  mitigation even with per-R NAF.

The experiment in §5 determines which regime we actually occupy.

---

## 5. Proposed experiment to close the gap

The paper gives the per-cell noise model; what we need is the
R-dependence of effective precision at the ACAM output. Proposed
steps:

1. **Extend the simulator** (`azurelily/IMC/`) to inject the paper's
   noise model:
   - Per-cell σ_G via Eq. 5 using the paper's fitted (a_x, b_x, c_x).
   - Per-sample read fluctuation σ_fluct.
   - Per-ACAM-cell threshold σ via Eq. 7.
2. **Add non-ideal effects** as separate, toggleable terms so their
   contributions can be decomposed:
   - IR drop along row wires (parameterised by wire resistance and
     row length at each R).
   - Sneak-path leakage through OFF cells.
   - Optional: correlated cell σ model (process-variation clusters).
3. **Sweep R ∈ {128, 256, 512, 1024}** at fixed workload (BERT-Tiny,
   VGG-16, LeNet). For each R, measure:
   - Effective bit-precision at ACAM output under (a) ideal model
     only, (b) ideal + IR drop, (c) full non-ideal model.
   - Task accuracy before and after single-pass NAF.
   - Per-layer accuracy contribution, since different layers may
     tile to different R.
4. **Identify the crossover**: the R at which (full non-ideal)
   effective precision dips below the 8-bit operating point and
   approaches the 7-bit cliff.
5. **Assess NAF reach**: per-R retraining recovers accuracy up to
   what R_max? Where does hardware mitigation become necessary?
6. **Mitigation sensitivity**: show accuracy vs area/latency for
   multi-pass averaging and redundant columns, so the paper can
   quote a Pareto frontier.

This produces paper-citeable answers:
- **5.1**: ideal physics predicts SNR improves as √R; actual
  R-dependence of effective precision is set by where non-ideal
  effects overtake the ideal gain — measured by the experiment, not
  predicted a priori.
- **5.2**: 130 rows = 8-bit capability (Gray-coded); precision floor
  from ACAM σ is above 8 bits, so row count is not the binding
  constraint. This is directly from the paper (Table I).
- **5.3**: 8-bit effective precision required for <1% accuracy loss;
  7-bit is already catastrophic. NAF recovers within its training
  distribution but per-R retraining is needed if effective precision
  varies across DSE configs. Whether any R requires additional
  hardware mitigation depends on where the non-ideal crossover sits,
  measured by the experiment.
