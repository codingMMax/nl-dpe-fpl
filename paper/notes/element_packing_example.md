# Element Packing in DPE DIMM: How Identity Crossbar Computes Attention

## Background

In BERT attention, we compute `Score = Q × K^T`:

```
GEMM:  (M × K) × (K × N) → (M × N)
       Q(S,d)  × K^T(d,S) → Score(S,S)

score[i,j] = Σ_k  Q[i,k] * K^T[k,j]

  i ∈ [0, S)       → row of Score     (M dimension)
  j ∈ [0, S)       → column of Score  (N dimension)
  k ∈ [0, d_head)  → reduction dim    (K dimension)
```

In NL-DPE's log domain, this becomes:

```
score[i,j] = Σ_k exp( log(Q[i,k]) + log(K^T[k,j]) )
```

The three steps per output element:
1. **CLB add**: compute `s[k] = log(Q[i,k]) + log(K^T[k,j])` for each k
2. **DPE(I|exp)**: feed `s[k]` through identity crossbar → ACAM computes `exp(s[k])`
3. **CLB reduce**: sum all `exp(s[k])` to get `score[i,j]`

The DPE crossbar is loaded with an **identity matrix** — it just routes each input to its corresponding column, where the ACAM applies `exp()`. All C columns fire in parallel every pass (same energy regardless of how many carry useful data).

---

## Example Setup

- **d_head = 3** (3-dimensional attention head, simplified)
- **Crossbar: 6 rows × 6 columns** (C = 6)
- **Goal**: compute score[i, j=0] and score[i, j=1]

---

## Without Packing (K_id = 1, 50% utilization)

We compute **one element per pass**, using only 3 of 6 columns:

```
Computing score[i, j=0]:

  s[k] = log(Q[i,k]) + log(K^T[k, j=0])    for k = 0, 1, 2

Identity crossbar (6×6):          Input vector (6 rows):

col:  0  1  2  3  4  5
     [1  0  0  0  0  0]  ← row 0   s[0]
     [0  1  0  0  0  0]  ← row 1   s[1]
     [0  0  1  0  0  0]  ← row 2   s[2]
     [0  0  0  1  0  0]  ← row 3   0  (unused)
     [0  0  0  0  1  0]  ← row 4   0  (unused)
     [0  0  0  0  0  1]  ← row 5   0  (unused)

ACAM output per column:
  col 0: exp(s[0]) = Q[i,0] * K^T[0,0]   ✓ useful
  col 1: exp(s[1]) = Q[i,1] * K^T[1,0]   ✓ useful
  col 2: exp(s[2]) = Q[i,2] * K^T[2,0]   ✓ useful
  col 3: exp(0) = 1                       ✗ wasted
  col 4: exp(0) = 1                       ✗ wasted
  col 5: exp(0) = 1                       ✗ wasted

CLB reduce (cols 0-2):
  score[i,0] = Q[i,0]*K^T[0,0] + Q[i,1]*K^T[1,0] + Q[i,2]*K^T[2,0]

Result: 1 output element, 3/6 = 50% column utilization
```

Then repeat for j=1 in a **second pass** → 2 passes for 2 elements.

---

## With Element Packing (K_id = 2, 100% utilization)

We compute **two elements in one pass** by packing different j-indices into the unused rows:

```
Computing score[i, j=0] AND score[i, j=1] simultaneously:

  a[k] = log(Q[i,k]) + log(K^T[k, j=0])    for k = 0,1,2
  b[k] = log(Q[i,k]) + log(K^T[k, j=1])    for k = 0,1,2

Identity crossbar (6×6):          Input vector (PACKED, 6 rows):

col:  0  1  2  3  4  5
     [1  0  0  0  0  0]  ← row 0   a[0]    ← for j=0
     [0  1  0  0  0  0]  ← row 1   a[1]    ← for j=0
     [0  0  1  0  0  0]  ← row 2   a[2]    ← for j=0
     [0  0  0  1  0  0]  ← row 3   b[0]    ← for j=1
     [0  0  0  0  1  0]  ← row 4   b[1]    ← for j=1
     [0  0  0  0  0  1]  ← row 5   b[2]    ← for j=1

ACAM output per column:
  col 0: exp(a[0]) = Q[i,0] * K^T[0,0]   ✓ for j=0
  col 1: exp(a[1]) = Q[i,1] * K^T[1,0]   ✓ for j=0
  col 2: exp(a[2]) = Q[i,2] * K^T[2,0]   ✓ for j=0
  col 3: exp(b[0]) = Q[i,0] * K^T[0,1]   ✓ for j=1
  col 4: exp(b[1]) = Q[i,1] * K^T[1,1]   ✓ for j=1
  col 5: exp(b[2]) = Q[i,2] * K^T[2,1]   ✓ for j=1

CLB reduce (cols 0-2): score[i,0] = Σ_k Q[i,k] * K^T[k,0]
CLB reduce (cols 3-5): score[i,1] = Σ_k Q[i,k] * K^T[k,1]

Result: 2 output elements in 1 pass, 6/6 = 100% column utilization
```

**Key insight**: the identity matrix routes each row to its matching column. Rows 0-2 naturally map to cols 0-2 (element j=0), rows 3-5 map to cols 3-5 (element j=1). The CLB reduction sums each group independently.

---

## Packing Factor (K_id)

```
K_id = floor(C / d_head)
```

| Crossbar | d_head | K_id | Elements/pass | Column utilization |
|----------|--------|------|---------------|-------------------|
| 1024×128 (Proposed) | 64 | 2 | 2 | 100% |
| 1024×256 (AL-like) | 64 | 4 | 4 | 100% |
| 1024×128 | 128 | 1 | 1 | 100% |
| 1024×128 | 100 | 1 | 1 | 78% (28 cols idle) |

---

## What About Score×V?

Score×V has inner dimension = **seq_len** (not d_head):

```
GEMM:  Score(S,S) × V(S,d) → Output(S,d)

output[i,m] = Σ_j  Score[i,j] * V[j,m]

  K dimension (reduction) = seq_len = S
```

| Crossbar | seq_len | K_id | Passes/element |
|----------|---------|------|---------------|
| 1024×128 | 64 | 2 | 1 |
| 1024×128 | 128 | 1 | 1 |
| 1024×128 | 256 | 0 → 1 | ceil(256/128) = 2 |
| 1024×128 | 1024 | 0 → 1 | ceil(1024/128) = 8 |

When seq_len > C, the inner dimension exceeds the crossbar width → multiple passes needed, no packing possible (K_id = 1).

---

## Summary

- The DPE identity crossbar is a **parallel exp() unit** — all C columns fire every pass
- Without packing: only `d_head` columns carry useful data, rest are wasted
- With packing: concatenate `K_id` independent elements' inputs into one C-wide vector
- Packing is free — same crossbar energy, more useful work per pass
- BERT-Tiny (d_head=64): Proposed gets 2×, AL-like gets 4× throughput from packing
