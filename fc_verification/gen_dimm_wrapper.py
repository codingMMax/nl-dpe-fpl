#!/usr/bin/env python3
"""
Generate standalone DIMM pipeline RTL for verification.

Uses the Proposed NL-DPE config (R=1024, C=128, dpe_buf_width=40)
matching the IMC simulator's gemm_log() and dimm_nonlinear() model.

Usage:
    python3 fc_verification/gen_dimm_wrapper.py
"""
import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "nl_dpe"))

from gen_attention_wrapper import _gen_dimm_score_matrix
from gen_gemv_wrappers import _get_supporting_modules

# ── Proposed NL-DPE config (matching run_seqlen_imc.py) ──────────────
R = 1024        # crossbar rows
C = 128         # crossbar cols
DPE_BW = 40     # dpe_buf_width (bits)
EPW = DPE_BW // 8  # elements per packed word = 5
DW = DPE_BW     # DATA_WIDTH = dpe_buf_width

# ── BERT-Tiny attention head dims ────────────────────────────────────
D_HEAD = 64     # head dimension
N_SEQ = 128     # sequence length (BERT-Tiny actual)

# ── Derived parameters ───────────────────────────────────────────────
K_id = C // D_HEAD                    # identity packing factor = 2
DUAL_IDENTITY = K_id >= 2
PACKED_WIDTH = K_id * D_HEAD          # = C = 128 (full column utilization)
PACKED_D = math.ceil(D_HEAD / EPW)    # packed words per key vector = 13

# SRAM depths (padded per-key storage)
DEPTH_Q = PACKED_D + 1                # one query: 13+1 = 14
DEPTH_K = N_SEQ * PACKED_D + 1        # all keys: 4*13+1 = 53
DEPTH_SCORE = N_SEQ + 1               # one score per element: 4+1 = 5

H_DIMM = math.ceil(max(D_HEAD, N_SEQ) / C)  # parallel DPE instances = 1

# Simulator cycle counts (ground truth)
READ_CYCLES = math.ceil(C * 8 / DPE_BW)           # = 26
OUTPUT_CYCLES = math.ceil(C * 8 / DPE_BW)         # = 26
COMPUTE_CYCLES = 3                                  # ACAM
PER_PASS_CYCLES = READ_CYCLES + COMPUTE_CYCLES + OUTPUT_CYCLES  # = 55
DPE_PASSES = math.ceil(D_HEAD / C)                 # = 1
CLB_REDUCE = math.ceil(math.log2(max(2, D_HEAD)))  # = 6
CYCLES_PER_ROW = 1 + DPE_PASSES * PER_PASS_CYCLES + CLB_REDUCE  # = 62

OUT_DIR = PROJECT_ROOT / "fc_verification" / "rtl"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "dimm_pipeline_d64_c128.v"

# ── Generate RTL ─────────────────────────────────────────────────────
print("=" * 60)
print("  DIMM Pipeline RTL Generator")
print("=" * 60)
print()
print("Configuration (Proposed NL-DPE, BERT-Tiny):")
print(f"  Crossbar:        R={R}, C={C}")
print(f"  dpe_buf_width:   {DPE_BW} bits (epw={EPW})")
print(f"  d_head:          {D_HEAD}")
print(f"  seq_len (test):  {N_SEQ}")
print(f"  K_id:            {K_id} ({'dual' if DUAL_IDENTITY else 'single'}-identity)")
print(f"  packed_width:    {PACKED_WIDTH} (= C, full utilization)")
print(f"  packed_d:        {PACKED_D} words per key vector")
print(f"  h_dimm:          {H_DIMM} parallel DPE(s)")
print()
print("SRAM Depths:")
print(f"  Q SRAM:     {DEPTH_Q} words (1 query × {PACKED_D} + 1)")
print(f"  K SRAM:     {DEPTH_K} words ({N_SEQ} keys × {PACKED_D} + 1)")
print(f"  Score SRAM: {DEPTH_SCORE} words ({N_SEQ} scores + 1)")
print()
print("IMC Simulator Cycle Counts (ground truth):")
print(f"  DPE read:          {READ_CYCLES} cycles (ceil({C}×8/{DPE_BW}))")
print(f"  DPE compute:       {COMPUTE_CYCLES} cycles (ACAM)")
print(f"  DPE output:        {OUTPUT_CYCLES} cycles (ceil({C}×8/{DPE_BW}))")
print(f"  Per DPE pass:      {PER_PASS_CYCLES} cycles")
print(f"  DPE passes/elem:   {DPE_PASSES}")
print(f"  CLB reduce:        {CLB_REDUCE} cycles (ceil(log2({D_HEAD})))")
print(f"  Cycles per row:    {CYCLES_PER_ROW} (1 + {DPE_PASSES}×{PER_PASS_CYCLES} + {CLB_REDUCE})")
print()
print("RTL DPE Parameters:")
print(f"  KERNEL_WIDTH:      {PACKED_WIDTH} (= K_id × d_head = C)")
print(f"  NUM_COLS:          {C}")
print(f"  DPE_BUF_WIDTH:     {DPE_BW}")
print(f"  COMPUTE_CYCLES:    {COMPUTE_CYCLES}")
print(f"  ACAM_MODE:         1 (exp)")
print(f"  LOAD_STROBES:      {math.ceil(PACKED_WIDTH / EPW)} (ceil({PACKED_WIDTH}/{EPW}))")
print(f"  OUTPUT_CYCLES:     {math.ceil(C / EPW)} (ceil({C}/{EPW}))")
print()

# Generate the dimm_score_matrix module
rtl = _gen_dimm_score_matrix(
    n_seq=N_SEQ,
    d_head=D_HEAD,
    h_dimm=H_DIMM,
    depth_q=DEPTH_Q,
    depth_k=DEPTH_K,
    depth_score=DEPTH_SCORE,
    data_width=DW,
    dual_identity=DUAL_IDENTITY,
    uid=0,
)

# Supporting modules (sram, controller, etc.)
supporting = _get_supporting_modules()

with open(OUT_FILE, "w") as f:
    f.write(f"// DIMM Pipeline RTL — Proposed NL-DPE, BERT-Tiny dims\n")
    f.write(f"// R={R}, C={C}, d_head={D_HEAD}, N={N_SEQ}, K_id={K_id}\n")
    f.write(f"// dpe_buf_width={DPE_BW}, dual_identity={DUAL_IDENTITY}\n")
    f.write(f"// Simulator: per_pass={PER_PASS_CYCLES}cyc, per_row={CYCLES_PER_ROW}cyc\n\n")
    f.write(rtl)
    f.write("\n\n")
    f.write(supporting)

print(f"Generated: {OUT_FILE}")
print(f"  Lines: {len(rtl.splitlines())} (dimm_score_matrix) + {len(supporting.splitlines())} (supporting)")
print()

# ── Verify RTL parameters ────────────────────────────────────────────
print("RTL Parameter Verification:")
with open(OUT_FILE) as f:
    content = f.read()

checks = [
    ("dual_identity", "S_WRITE_B" in content, True),
    ("K_id=2 (dual)", "feed_half" in content, True),
    ("byte-wise adder", "add_a" in content, True),
    ("DPE direct (no conv_layer)", "dpe dimm_exp" in content, True),
    ("masked accumulator", "masked_sum_a" in content or "masked_byte_sum" in content, True),
    ("MSB_SA_Ready wait", "MSB_SA_Ready" in content, True),
    ("score SRAM", f"DEPTH({DEPTH_SCORE})" in content, True),
]

all_pass = True
for name, check, expected in checks:
    status = "✓" if check == expected else "✗"
    if check != expected:
        all_pass = False
    print(f"  [{status}] {name}")

if all_pass:
    print("\nAll RTL checks passed.")
else:
    print("\nWARNING: Some RTL checks failed!")
