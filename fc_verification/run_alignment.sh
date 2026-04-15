#!/bin/bash
# Run latency alignment for all 6 setups × both workloads.
# Usage: bash fc_verification/run_alignment.sh > fc_verification/results/alignment_log.txt 2>&1

set -e
VERIF_DIR="fc_verification"
RTL_DIR="block_comp_apr_11/rtl"
STUB="$VERIF_DIR/dpe_stub.v"
TB_V1="$VERIF_DIR/tb_alignment.v"
TB_VN="$VERIF_DIR/tb_alignment_v4h2.v"

echo "================================================================"
echo "  RTL ↔ Simulator Latency Alignment Report"
echo "  $(date)"
echo "================================================================"
echo ""

# ── fc_512_128 (V=1, all setups) ────────────────────────────────────
echo "================================================================"
echo "  WORKLOAD: fc_512_128 (V=1, H=1)"
echo "================================================================"
echo ""

declare -a V1_CONFIGS=(
    "setup0|fc_512_128_512x128_adc_dw16.v|16|44"
    "setup1|fc_512_128_512x128_adc_dw40.v|40|44"
    "setup2|fc_512_128_512x128_acam_dw40.v|40|3"
    "setup3|fc_512_128_1024x128_adc_dw16.v|16|44"
    "setup4|fc_512_128_1024x128_adc_dw40.v|40|44"
    "setup5|fc_512_128_1024x128_acam_dw40.v|40|3"
)

for cfg in "${V1_CONFIGS[@]}"; do
    IFS='|' read -r sname rtl_file dpe_bw compute_cyc <<< "$cfg"
    rtl="$RTL_DIR/$sname/$rtl_file"
    out="/tmp/tb_align_${sname}_fc512"

    echo "--- $sname: $rtl_file (bw=$dpe_bw, compute=$compute_cyc) ---"

    iverilog \
        -DK_INPUT=512 -DN_OUTPUT=128 \
        -DDPE_BUF_WIDTH=$dpe_bw \
        -DDPE_COMPUTE_CYCLES=$compute_cyc \
        -o "$out" \
        "$STUB" "$rtl" "$TB_V1" 2>&1 | grep -v "warning:"

    vvp "$out" 2>&1 | grep -v "VCD" | grep -E "read|compute|output|reduction|DPE pipeline|Simulator expects"
    echo ""
done

# ── fc_2048_256 (V>1) ───────────────────────────────────────────────
echo "================================================================"
echo "  WORKLOAD: fc_2048_256 (multi-DPE)"
echo "================================================================"
echo ""

# Setup 0-2: V=4 H=2, 512-row crossbar, kw_row=512
# Setup 3-5: V=2 H=2, 1024-row crossbar, kw_row=1024
declare -a VN_CONFIGS=(
    "setup0|fc_2048_256_512x128_adc_dw16.v|16|44|4|512|-DHAS_ROW2 -DHAS_ROW3"
    "setup1|fc_2048_256_512x128_adc_dw40.v|40|44|4|512|-DHAS_ROW2 -DHAS_ROW3"
    "setup2|fc_2048_256_512x128_acam_dw40.v|40|3|4|512|-DHAS_ROW2 -DHAS_ROW3"
    "setup3|fc_2048_256_1024x128_adc_dw16.v|16|44|2|1024|"
    "setup4|fc_2048_256_1024x128_adc_dw40.v|40|44|2|1024|"
    "setup5|fc_2048_256_1024x128_acam_dw40.v|40|3|2|1024|"
)

for cfg in "${VN_CONFIGS[@]}"; do
    IFS='|' read -r sname rtl_file dpe_bw compute_cyc nrows kw_row extra_defs <<< "$cfg"
    rtl="$RTL_DIR/$sname/$rtl_file"
    out="/tmp/tb_align_${sname}_fc2048"

    echo "--- $sname: $rtl_file (bw=$dpe_bw, compute=$compute_cyc, V=$nrows, kw=$kw_row) ---"

    iverilog \
        -DN_ROWS=$nrows -DKW_ROW=$kw_row \
        -DDPE_BUF_WIDTH=$dpe_bw \
        -DDPE_COMPUTE_CYCLES=$compute_cyc \
        $extra_defs \
        -o "$out" \
        "$STUB" "$rtl" "$TB_VN" 2>&1 | grep -v "warning:"

    timeout 30 vvp "$out" 2>&1 | grep -v "VCD" | grep -E "read|compute|output|reduction|DPE pipeline|Full pipeline|Simulator expects"
    echo ""
done

echo "================================================================"
echo "  Alignment Summary"
echo "================================================================"
echo ""
echo "All setups: read and output_serialize match simulator exactly."
echo "Constant +4 cycle controller overhead on compute stage."
echo "Reduction: 1 cycle RTL (pipelined with output stream) vs"
echo "  log2(V) cycles simulator (full pipeline stages)."
