# `gemm_top` — GEMM (v3 module spec)

Status: 2026-10-22, written for the bit-plane pass (`ip/xbar/spec.md`, `docs/architecture.md` §3.1). Until now
`gemm_top` was v2's certified module, unchanged (charter `git show v2-final:v2/sim/spec/gemm.md` v0.3: 40-bit ports,
a byte tree, a lane serializer). The behavior model `simulator/models/gemm.py` implements this spec. Both RTL twins
(`rtl/{sim,vtr}/gemm_top.v`) still have v2's interface and change in the RTL pass.

## 1 Function

$Y = \mathrm{trunc8}(X W)$, REGULAR, for $X$ $M \times K$ and $W$ $K \times N$ (int8):
- $V = \lceil K/R \rceil$ row tiles × $H = \lceil N/C \rceil$ column tiles of `dpe`, in lock-step;
- the $V$ tiles of a column are summed mod 256 (a tree), and the low byte is the output;
- P29 extent: `ROWS_USED` = $K$ when $V = 1$ (else $R$); `COLS_USED` = $N$ when $H = 1$ (else $C$).

Values are v2's: the wide sum of the sign-extended tile bytes, then its low byte (the probe `S_col` keeps the wide sum).

## 2 Parameters

`K, N, R, C, P, ROWS_USED, COLS_USED`. `BUF` is removed (the data ports are planes).

Derived: $V$, $H$; `TREE_PIPE` = $\lceil \log_2 V \rceil$ registered tree levels ($V > 1$), else 0;
`NTILES` = $V H$; `L_W` = `TREE_PIPE` + `SER_LAT`.

Guards (v2's): $K, N, R, C, P \ge 1$; the extent guard above (`g_bad_extent`).

## 3 Ports

| port | width | dir | discipline | unit |
|---|---|---|---|---|
| `act_in`, `act_en` | $V R$, 1 | in | burst on ready | a row: $V$ lanes, lane $v$ = rows $vR \ldots vR + $ `ROWS_USED` $- 1$ of the row, plane framing (8 words, LSB first; bits past `ROWS_USED` are 0) |
| `MSB_SA_Ready` | 1 | out | | a burst may start (the AND of the tiles' ready) |
| `data_out`, `out_valid` | $H C$, 1 | out | push | a row: $H$ lanes of `COLS_USED` results, 8 plane words |
| `dpe_done` | 1 | out | | pulse one cycle after the row's last output word |
| `reg_full` | 1 | out | | a busy flag (not a word valid) |
| `weight_in`, `weight_en` | 8, 1 | in | | simulation twin only (the testbench preloads, R8) |

## 4 Structure

```text
 act_in lane v (R bits per plane) -> the H tiles (v, h): dpe REGULAR, rows ROWS_USED
 per column h: the V tiles' drain planes -> bit-serial mod-256 tree (TREE_PIPE levels) -> output register -> data_out lane h
```

- **The tree is bit-serial.** Each level adds two planes column by column, with a full adder and a carry flip-flop
  per column. The carry is cleared at plane 0, and the carry out of plane 7 is dropped (mod 256). Every level is
  registered.
- **$V = 1$:** no tree; the drain plane goes through the output register.

## 5 Timing rules

| name | value | rule |
|---|---|---|
| — | the pass rule | a row's burst at $t_b$ (in a cycle with `MSB_SA_Ready` high) is the tiles' `load_start` (`ip/xbar/spec.md` §4) |
| `TREE_PIPE` | $\lceil \log_2 V \rceil$ | registered bit-serial tree levels ($V > 1$) |
| `SER_LAT` | 1 | the output register |
| `L_W` | `TREE_PIPE` + `SER_LAT` | `drain_start` → the row's first output word |

- Output word $b$ of the row, every lane: at `drain_start` + `L_W` + $b$ ($b = 0 \ldots 7$).
- `dpe_done` one cycle after word 7.
- `MSB_SA_Ready` rises at `acam` + 1, which is the burst port's credit.

Back to back from cycle 0, row $m$'s burst is at $10m$ and its last output word at $10m + 17 + L_W$. So $M$ rows
take $10M + 8 + L_W$ cycles. Under v2's 40-bit pass the row period was $\max(\lceil 8\,\text{ROWS\_USED}/40 \rceil +
8,\ \lceil 8C/40 \rceil + 1)$.

## 6 Counts and resources

Unchanged from v2. Per row: $V H$ passes (REGULAR, `ROWS_USED`, `COLS_USED`); $N(V - 1)$ tree adds; $N$ output
bytes. Crossbars: $V H$.

## 7 Gates

- **The model** (`simulator/tests/testGemm.py`):
  - the v2 GEMM corpora's values (output bytes and the `S_col` probe);
  - every row's events equal the closed form of §5, written in the test;
  - late offers;
  - the structure (parameters and signals against the RTL; `BUF` leaves it in the RTL pass);
  - planted faults (a latency, a wrong extent seen in the counts, a dropped ready credit).
- **GATE 2:** covered by the head's 64×64 runs (bring-up and gate H; `top/spec.md`).
