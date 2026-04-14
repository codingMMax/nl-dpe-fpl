#!/bin/bash
# Run latency alignment testbench for all 6 setups × fc_512_128
# Compares RTL cycle counts (pre-loaded SRAM) vs simulator predictions.
#
# Usage: bash block_comp_apr_11/tb/run_alignment.sh > block_comp_apr_11/results/alignment_log.txt 2>&1

set -e
TB_DIR="block_comp_apr_11/tb"
RTL_DIR="block_comp_apr_11/rtl"
STUB="$TB_DIR/dpe_stub.v"
TB="$TB_DIR/tb_alignment.v"

echo "================================================================"
echo "  RTL ↔ Simulator Latency Alignment Report"
echo "  $(date)"
echo "================================================================"
echo ""

# Setup configs: name, rtl_file, dpe_buf_width, compute_cycles
# ADC compute: 44 cycles (8 bit-slices × 3-stage pipeline + ADC)
# ACAM compute: 3 cycles (8 bit-slices × 2-stage pipeline + ACAM drain)
declare -a CONFIGS=(
    "setup0|fc_512_128_512x128_adc_dw16.v|16|44"
    "setup1|fc_512_128_512x128_adc_dw40.v|40|44"
    "setup2|fc_512_128_512x128_acam_dw40.v|40|3"
    "setup3|fc_512_128_1024x128_adc_dw16.v|16|44"
    "setup4|fc_512_128_1024x128_adc_dw40.v|40|44"
    "setup5|fc_512_128_1024x128_acam_dw40.v|40|3"
)

K=512
N=128

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r sname rtl_file dpe_bw compute_cyc <<< "$cfg"
    rtl="$RTL_DIR/$sname/$rtl_file"
    out="/tmp/tb_align_${sname}"

    echo "================================================================"
    echo "  $sname: $rtl_file (DPE_BUF_WIDTH=$dpe_bw, COMPUTE=$compute_cyc)"
    echo "================================================================"

    iverilog \
        -DK_INPUT=$K -DN_OUTPUT=$N \
        -DDPE_BUF_WIDTH=$dpe_bw \
        -DDPE_COMPUTE_CYCLES=$compute_cyc \
        -o "$out" \
        "$STUB" "$rtl" "$TB" 2>&1 | grep -v "warning:"

    vvp "$out" 2>&1 | grep -v "VCD"

    echo ""
done

# Summary table
echo "================================================================"
echo "  Summary: Simulator vs RTL cycle counts (fc_512_128)"
echo "================================================================"
echo ""
echo "Simulator predictions (from _dpe_buf_fill_row + _core_bit_pipeline_row_latency):"
echo "  Setup 0 (16b/ADC):  read=256  compute=44  output=64   total=364"
echo "  Setup 1 (40b/ADC):  read=103  compute=44  output=26   total=173"
echo "  Setup 2 (40b/ACAM): read=103  compute=3   output=26   total=132"
echo "  Setup 3 (16b/ADC):  read=256  compute=44  output=64   total=364"
echo "  Setup 4 (40b/ADC):  read=103  compute=44  output=26   total=173"
echo "  Setup 5 (40b/ACAM): read=103  compute=3   output=26   total=132"
echo ""
echo "Expected RTL delta: +4 cycles constant controller overhead on compute stage."
echo "Expected RTL totals: sim_total + 4"
