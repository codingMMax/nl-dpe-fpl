#!/usr/bin/env python3
"""
NL-DPE Crossbar Size DSE — Detailed Implementation Plan
========================================================

This file is a specification document (not executable code).
It describes the interface, logic, and dependencies for each TODO module
listed in dse_experiment_plan.md §9.

Modules:
  1. azurelily/models/fc.py          — FC+activation model for IMC simulator
  2. azurelily/IMC/test.py           — register fc_model (1-line change)
  3. nl_dpe/gen_arch_xml.py           — generate per-(R,C) arch XMLs
  4. nl_dpe/gen_gemv_wrappers.py      — extend for FC+activation RTL
  5. gemv_dse.py                      — DSE orchestrator (Round 1 + Round 2)
"""

# =============================================================================
# MODULE 1: azurelily/models/fc.py
# =============================================================================
#
# PURPOSE:
#   Single FC layer with activation for the IMC simulator.
#   Clone of gemv_model (azurelily/models/gemv.py) but with has_act=True.
#   This is what the DSE uses to evaluate energy and latency per (R,C,K,N).
#
# INTERFACE:
#   fc_model(num_computes, num_inputs, seq_length, head_dim,
#            debug, energy_stats) -> (all_layers, num_finishes)
#
#   CLI mapping: --seq_length=K  --head_dim=N  (reuses gemv_model convention)
#
# IMPLEMENTATION:
#
#   import nn
#   import nn.constant as C
#
#   FC_SHAPES = {
#       "64_64":     (64,   64),
#       "512_128":   (512,  128),
#       "2048_256":  (2048, 256),
#       "256_512":   (256,  512),
#       "512_512":   (512,  512),
#   }
#
#   def fc_model(num_computes, num_inputs, seq_length, head_dim,
#                debug, energy_stats, K=64, N=64):
#       K = seq_length
#       N = head_dim
#       layer = nn.Layer(
#           in_channels=K,
#           out_channels=N,
#           kernel_size=1,
#           stride=1,
#           padding=0,
#           name=f"fc_{K}_{N}",
#           type="linear",
#           has_act=True,           # <-- key difference from gemv_model
#           num_computes=num_computes,
#           num_inputs=num_inputs,
#           debug=debug,
#           energy_stats=energy_stats,
#       )
#       layer.set_input(1, 1, K, is_first=True)
#       all_layers = [layer]
#       all_layers[0].add_event(C.EVENT_NEW_DATA, 0)
#       return all_layers, num_inputs * len(all_layers)
#
# WHAT THE SIMULATOR DOES WITH has_act=True (scheduler.py:_run_linear):
#   1. Calls imc_core.run_gemm(M=1, K, N) → VMM + ACAM + reduction + BRAM
#   2. Checks analoge_nonlinear_check(M, K, N):
#      - If V=1 (K ≤ rows): returns True → activation handled by ACAM, no CLB cost
#      - If V>1 (K > rows): returns False → calls fpga.activation(M, N)
#        which adds: energy = M × N × act_energy_pj_per_op (0.45 pJ)
#        and merges activation latency into pipeline
#
# DEPENDENCIES: nn.Layer, nn.constant (both exist)
# TESTING: python azurelily/IMC/test.py --model fc --imc_file IMC/configs/nl_dpe.json
#          --seq_length 512 --head_dim 128
# EXPECTED: same energy as gemv for V=1 cases; higher energy for V>1 (activation added)
#
# EFFORT: ~30 lines, copy-paste from gemv.py + change has_act


# =============================================================================
# MODULE 2: azurelily/IMC/test.py — register fc_model
# =============================================================================
#
# CHANGE: Add 1 import + 1 dict entry
#
#   from models.fc import fc_model     # <-- add import
#
#   model_list = {
#       "lenet":    lenet_model,
#       ...
#       "gemv":     gemv_model,
#       "fc":       fc_model,           # <-- add entry
#   }
#
# EFFORT: 2 lines


# =============================================================================
# MODULE 3: nl_dpe/gen_arch_xml.py
# =============================================================================
#
# PURPOSE:
#   Generate per-(R,C) architecture XMLs for VTR.
#   Round 1: update DPE tile dimensions in auto_layout XML.
#   Round 2: additionally modify layout for CLB replacement / DSP+BRAM equiv.
#
# INTERFACE:
#   gen_arch_xml(
#       rows: int, cols: int,
#       template_xml: str = "nl_dpe_22nm_auto.xml",
#       output_dir: str = ".",
#       mode: str = "auto",             # "auto" | "fixed_clb_replace" | "fixed_dsp_bram"
#       # Round 2 only:
#       fixed_grid_w: int = None,
#       fixed_grid_h: int = None,
#       clb_replace_ratio: float = None, # e.g. 0.05, 0.08, 0.12, 0.15
#       extra_dsps: int = None,          # for DSP+BRAM equivalence
#       extra_brams: int = None,
#   ) -> Path   # returns path to generated XML
#
# CLI:
#   python gen_arch_xml.py --rows 256 --cols 128
#   python gen_arch_xml.py --rows 256 --cols 128 --mode fixed_clb_replace \
#       --fixed-grid 80x80 --clb-replace-ratio 0.10
#
# IMPLEMENTATION PLAN:
#
#   1. Read template XML as text (xml.etree is fragile with VTR's DTD-heavy XML)
#      Use regex-based patching — same approach as run_imc_with_vtr_freq.py
#
#   2. Import dpe_specs from area_power.py:
#      sys.path.insert(0, SCRIPT_DIR)
#      from area_power import dpe_specs
#      specs = dpe_specs(rows, cols)
#
#   3. PATCH DPE TILE — find and replace the <tile name="wc" ...> line:
#      Original (line 658 of nl_dpe_22nm_auto.xml):
#        <tile name="wc" height="5" width="6" area="1595429">
#      Replace with:
#        <tile name="wc" height="{specs['tile_height']}" width="{specs['tile_width']}"
#              area="{int(specs['area_tag_mwta'])}">
#
#      Regex pattern:
#        r'<tile name="wc" height="\d+" width="\d+" area="\d+">'
#      Replacement:
#        f'<tile name="wc" height="{h}" width="{w}" area="{area}">'
#
#   4. MODE = "auto" (Round 1):
#      - Keep <auto_layout> as is — no layout changes needed
#      - Just write patched XML to output_dir/nl_dpe_{R}x{C}_auto.xml
#
#   5. MODE = "fixed_clb_replace" (Round 2, Part 1):
#      - Replace <auto_layout> block with <fixed_layout name="dse" width="W" height="H">
#      - Remove some CLB columns, add DPE columns
#      - n_dpes = floor(clb_replace_ratio * total_clb_area / specs['area_tag_mwta'])
#      - Each DPE column occupies tile_width grid columns and tile_height grid rows
#      - Layout generation:
#        a. Start with baseline fixed_layout column assignments
#        b. Remove ceil(n_dpes * tile_width) CLB columns from the middle
#        c. Insert n_dpes DPE column entries with correct startx
#      - Output: nl_dpe_{R}x{C}_clb{ratio}_fixed.xml
#
#   6. MODE = "fixed_dsp_bram" (Round 2, Part 2):
#      - Remove DPE tile from XML entirely
#      - Add extra_dsps DSP columns + extra_brams BRAM columns
#        within the same fixed_layout grid
#      - Output: nl_dpe_{R}x{C}_dsp{X}_bram{Y}_fixed.xml
#
# KEY DESIGN DECISION:
#   Use regex-based XML patching (not xml.etree) because:
#   - VTR arch XML uses custom DTD entities that break standard XML parsers
#   - The template file is stable — only 3 locations need patching
#   - run_imc_with_vtr_freq.py already uses this approach successfully
#
# REFERENCE DATA FROM TEMPLATE (nl_dpe_22nm_auto.xml):
#   Line 635: <tile name="clb" area="27905">                     CLB_AREA_MWTA
#   Line 658: <tile name="wc" height="5" width="6" area="1595429">  DPE tile
#   Line 672: <tile name="dsp_top" height="4" width="1" area="253779">
#   Line 708: <tile name="memory" height="2" width="1" area="137668">
#   Line 730: <auto_layout aspect_ratio="1.0">
#   Line 734:   <col type="dsp_top" startx="6" starty="1" repeatx="16" priority="20"/>
#   Line 735:   <col type="wc" startx="6" starty="1" repeatx="16" priority="22"/>
#   Line 736:   <col type="memory" startx="2" starty="1" repeatx="16" priority="20"/>
#   Line 737: </auto_layout>
#
# DEPENDENCIES: area_power.py (dpe_specs), template XML
# TESTING: generate XML, diff against template, run VTR sanity check
# EFFORT: ~120 lines


# =============================================================================
# MODULE 4: nl_dpe/gen_gemv_wrappers.py — extend for FC+activation RTL
# =============================================================================
#
# PURPOSE:
#   Current gen_gemv_wrappers.py generates bare GEMV wrappers (no activation).
#   Extend to also generate FC+activation wrappers for the DSE.
#   The FC wrapper is identical to GEMV except:
#     - When V>1: add a tanh/sigmoid CLB LUT module after the reduction adder tree
#     - When V=1: no activation module (ACAM handles it)
#
# CURRENT BEHAVIOR:
#   gen_wrapper(name, k, n, rows, cols) reads gemv_1_channel.v as template,
#   substitutes K/N/ROWS/COLS parameters, writes gemv_{name}.v
#
# CHANGES NEEDED:
#
#   1. Add --fc flag to CLI:
#      parser.add_argument("--fc", action="store_true",
#                          help="Generate FC+activation wrappers instead of bare GEMV")
#
#   2. Create fc_1_channel.v template (new file):
#      Clone gemv_1_channel.v with these changes:
#      a. Module name: fc (not gemv)
#      b. For V>1 stacking modules (V2_H1, V4_H1, V8_H1, V2_H2):
#         Add activation LUT after the final adder output.
#         The activation is a simple N-wide CLB lookup table:
#
#           // tanh/sigmoid approximation via piecewise-linear LUT
#           // 16-bit input → 16-bit output, uses ~4 LUTs per element
#           module activation_lut #(
#               parameter DATA_WIDTH = 16,
#               parameter N = 64
#           )(
#               input wire clk,
#               input wire [DATA_WIDTH*N-1:0] data_in,
#               input wire valid_in,
#               output reg [DATA_WIDTH*N-1:0] data_out,
#               output reg valid_out
#           );
#               integer i;
#               always @(posedge clk) begin
#                   valid_out <= valid_in;
#                   for (i = 0; i < N; i = i + 1) begin
#                       // Piecewise-linear tanh approximation
#                       // Uses comparators + shifts → synthesizes to CLB LUTs
#                       if ($signed(data_in[i*DATA_WIDTH +: DATA_WIDTH]) > 16'sd8192)
#                           data_out[i*DATA_WIDTH +: DATA_WIDTH] <= 16'sd16384;
#                       else if ($signed(data_in[i*DATA_WIDTH +: DATA_WIDTH]) < -16'sd8192)
#                           data_out[i*DATA_WIDTH +: DATA_WIDTH] <= -16'sd16384;
#                       else
#                           data_out[i*DATA_WIDTH +: DATA_WIDTH] <= data_in[i*DATA_WIDTH +: DATA_WIDTH];
#                   end
#               end
#           endmodule
#
#      c. For V=1 paths (single_dpe, v1_h2): NO activation module
#         (ACAM handles activation in hardware — the VTR design shouldn't
#          include CLB activation logic for V=1)
#
#      The key insight: for the DSE, we need the RTL to correctly reflect
#      CLB usage. V>1 designs instantiate activation_lut → VTR reports
#      higher CLB count. V=1 designs skip it → lower CLB count.
#
#   3. gen_fc_wrapper(name, k, n, rows, cols) -> Path:
#      Same as gen_wrapper but reads fc_1_channel.v template.
#      Output: fc_{name}_{rows}x{cols}.v
#      e.g.: fc_512_128_256x128.v
#
#   4. When --fc is set, generate fc_* wrappers for all 5 shapes.
#      When not set, generate gemv_* wrappers (backward compatible).
#
# EXISTING STACKING MODULES IN gemv_1_channel.v:
#   - conv_layer_single_dpe          (V=1, H=1)
#   - conv_layer_stacked_dpes_V2_H1  (V=2, H=1)
#   - conv_layer_stacked_dpes_V4_H1  (V=4, H=1)
#   - conv_layer_stacked_dpes_V8_H1  (V=8, H=1)
#   - conv_layer_stacked_dpes_V1_H2  (V=1, H=2)
#   - conv_layer_stacked_dpes_V2_H2  (V=2, H=2)
#
# NEW STACKING MODULES NEEDED FOR DSE CONFIGS:
#   When (R,C) varies across {128,256,512}×{64,128,256}, V and H can be:
#   - V up to 16 (K=2048, R=128 → V=16), H up to 8 (N=512, C=64 → H=8)
#   - We need: V16_H4, V16_H1, V8_H2, V4_H2, V4_H4, etc.
#
#   Rather than hardcoding every (V,H) combination, use GENERATE blocks:
#   Create a new template fc_1_channel.v with a fully generic stacking module:
#
#     module conv_layer_stacked_dpes_generic #(
#         parameter N_DPE_V = 1,
#         parameter N_DPE_H = 1,
#         ... // same params as existing stacking modules
#     )( ... );
#         // Generate V×H DPE instances
#         genvar gi, gj;
#         generate
#             for (gi = 0; gi < N_DPE_V; gi = gi + 1) begin : dpe_row
#                 for (gj = 0; gj < N_DPE_H; gj = gj + 1) begin : dpe_col
#                     dpe dpe_inst ( ... );
#                 end
#             end
#         endgenerate
#
#         // Reduction adder tree (if V > 1):
#         // For each horizontal column, reduce V partial sums
#         // ceil(log2(V)) levels of adder tree
#
#         // Horizontal concatenation (if H > 1):
#         // Concatenate H output columns into single N-wide output
#
#         // Activation LUT (if V > 1):
#         // activation_lut on the final output
#     endmodule
#
#   This single generic module handles ALL (V,H) combinations via Verilog
#   generate. Much cleaner than hardcoding V4_H1, V8_H1, etc.
#
# ALTERNATIVE (simpler, may be sufficient):
#   Since gemv_1_channel.v already has V1_H1, V2_H1, V4_H1, V8_H1, V1_H2, V2_H2,
#   we could add only the missing combinations needed for the DSE tiling table:
#
#   From the tiling table (dse_experiment_plan.md §4):
#     V values: 1, 2, 4, 8, 16, 32, 64
#     H values: 1, 2, 4
#     Missing modules: V16_H1, V32_H1, V16_H4, V16_H2, V32_H2, V8_H2
#
#   This is a lot of hardcoded modules. The generic approach is better.
#
# DEPENDENCIES: gemv_1_channel.v (template), area_power.py (for V/H calculation)
# TESTING: VTR synthesis of generated fc_*.v files
# EFFORT: ~200 lines (generic stacking module + activation_lut + wrapper gen)


# =============================================================================
# MODULE 5: gemv_dse.py — DSE orchestrator
# =============================================================================
#
# PURPOSE:
#   Orchestrate the full DSE: generate configs → run VTR → run IMC sim → collect metrics.
#   Supports Round 1 (auto_layout, 54 runs) and Round 2 (fixed_layout, 144 runs).
#
# INTERFACE:
#   python gemv_dse.py --round 1                     # Run Round 1 (54 VTR runs)
#   python gemv_dse.py --round 2 --top-k 3           # Run Round 2 (144 VTR runs)
#   python gemv_dse.py --round 1 --dry-run            # Print what would be run
#   python gemv_dse.py --round 1 --config 256x128     # Single config
#   python gemv_dse.py --round 1 --jobs 8             # Parallel VTR jobs
#
# CLI ARGUMENTS:
#   --round {1,2}           Which round to run
#   --configs               Comma-separated configs (default: all 9)
#   --workloads             Comma-separated workloads (default: all 6)
#   --jobs                  Max parallel VTR jobs (default: cpu_count)
#   --output-dir            Output directory (default: dse_results/)
#   --dry-run               Print plan without running
#   --top-k                 Round 2 only: how many top configs from Round 1
#   --r1-results            Round 2 only: path to Round 1 CSV
#   --template-scale        Round 2 only: scale factor for template grid (default: 1.2)
#
# IMPLEMENTATION PLAN:
#
#   ─── Configuration ───────────────────────────────────────────────────
#
#   DPE_CONFIGS = [(r, c) for r in [128, 256, 512] for c in [64, 128, 256]]
#
#   FC_WORKLOADS = {
#       "fc_64_64":     (64,   64),
#       "fc_512_128":   (512,  128),
#       "fc_2048_256":  (2048, 256),
#       "fc_256_512":   (256,  512),
#       "fc_512_512":   (512,  512),
#   }
#   # Attention workload handled separately (existing RTL, not parameterized)
#
#   ─── Round 1 Pipeline ───────────────────────────────────────────────
#
#   def run_round1(configs, workloads, output_dir, jobs):
#       results = []
#
#       # Step 1: Generate arch XMLs (one per config)
#       for (R, C) in configs:
#           gen_arch_xml(R, C, mode="auto")  # → nl_dpe_{R}x{C}_auto.xml
#
#       # Step 2: Generate RTL (one per config × workload)
#       for (R, C) in configs:
#           for wl_name, (K, N) in workloads.items():
#               gen_fc_wrapper(wl_name, K, N, R, C)  # → fc_{K}_{N}_{R}x{C}.v
#
#       # Step 3: Run VTR (parallelized, 54 total)
#       vtr_jobs = []
#       for (R, C) in configs:
#           arch = f"nl_dpe_{R}x{C}_auto.xml"
#           for wl_name, (K, N) in workloads.items():
#               rtl = f"fc_{K}_{N}_{R}x{C}.v"
#               vtr_jobs.append((R, C, K, N, wl_name, arch, rtl))
#
#       vtr_results = run_vtr_batch(vtr_jobs, jobs=jobs)
#       # Each result: {fmax_mhz, grid_w, grid_h, resources, run_dir}
#
#       # Step 4: Run IMC simulator (per VTR result)
#       for job, vtr_res in zip(vtr_jobs, vtr_results):
#           R, C, K, N, wl_name, _, _ = job
#
#           # Patch IMC config
#           specs = dpe_specs(R, C)
#           config = patch_imc_config(
#               base_config="azurelily/IMC/configs/nl_dpe.json",
#               rows=R, cols=C,
#               e_analoge_pj=specs['e_analogue_pj'],
#               e_digital_pj=specs['e_digital_pj'],
#               freq_mhz=vtr_res['fmax_mhz'],
#           )
#
#           # Run IMC
#           energy_pj, latency_ns, breakdown = run_imc_fc(config, K, N)
#
#           # Compute metrics
#           grid_area_mm2 = vtr_res['grid_w'] * vtr_res['grid_h'] * 2239 / 1e6
#           throughput = 1e9 / latency_ns
#           V = math.ceil(K / R)
#           H = math.ceil(N / C)
#
#           results.append({
#               'config': f"{R}x{C}",
#               'rows': R, 'cols': C,
#               'workload': wl_name, 'K': K, 'N': N,
#               'V': V, 'H': H, 'dpe_count': V * H,
#               'acam_eligible': V == 1,
#               'fmax_mhz': vtr_res['fmax_mhz'],
#               'grid_w': vtr_res['grid_w'],
#               'grid_h': vtr_res['grid_h'],
#               'fpga_area_mm2': grid_area_mm2,
#               'latency_ns': latency_ns,
#               'energy_pj': energy_pj,
#               'throughput_per_mm2': throughput / grid_area_mm2,
#               'throughput_per_J': 1e12 / energy_pj,
#               # breakdown
#               'e_imc_vmm': breakdown.get('imc_vmm', 0),
#               'e_imc_digital': breakdown.get('imc_digital_post', 0),
#               'e_clb_reduction': breakdown.get('clb_reduction', 0),
#               'e_clb_activation': breakdown.get('fpga_activation', 0),
#               'e_sram': breakdown.get('sram_read', 0) + breakdown.get('sram_write', 0),
#               # VTR resources
#               'clb_count': vtr_res['resources'].get('clb', 0),
#               'dsp_count': vtr_res['resources'].get('dsp_top', 0),
#               'mem_count': vtr_res['resources'].get('memory', 0),
#               'wc_count': vtr_res['resources'].get('wc', 0),
#           })
#
#       # Step 5: Write CSV
#       write_csv(results, output_dir / "round1_results.csv")
#
#       # Step 6: Select top-K configs
#       # Rank by average throughput_per_mm2 across workloads
#       top_k = select_top_configs(results, k=3, metric='throughput_per_mm2')
#       print(f"Top-{len(top_k)} configs: {top_k}")
#
#       return results, top_k
#
#   ─── VTR Grid Size Parsing ──────────────────────────────────────────
#
#   def parse_grid_size(vpr_log_path):
#       """Parse FPGA grid dimensions from VTR output.
#
#       VTR prints: 'FPGA sized to 40 x 40: 1600 grid tiles (auto)'
#       Returns: (grid_w, grid_h)
#       """
#       import re
#       pattern = r"FPGA sized to (\d+) x (\d+)"
#       content = Path(vpr_log_path).read_text()
#       match = re.search(pattern, content)
#       if match:
#           return int(match.group(1)), int(match.group(2))
#       raise ValueError(f"Could not parse grid size from {vpr_log_path}")
#
#   ─── IMC Config Patching ────────────────────────────────────────────
#
#   def patch_imc_config(base_config, rows, cols, e_analoge_pj, e_digital_pj,
#                        freq_mhz, output_path=None):
#       """Create a patched nl_dpe.json for a specific (R, C, freq) config.
#
#       Patches:
#         geometry.array_rows = rows
#         geometry.array_cols = cols
#         params.e_analoge_pj = e_analoge_pj
#         params.e_digital_pj = e_digital_pj
#         fpga_specs.freq = freq_mhz
#       """
#       import json
#       with open(base_config) as f:
#           cfg = json.load(f)
#       cfg['geometry']['array_rows'] = rows
#       cfg['geometry']['array_cols'] = cols
#       cfg['params']['e_analoge_pj'] = e_analoge_pj
#       cfg['params']['e_digital_pj'] = e_digital_pj
#       cfg['fpga_specs']['freq'] = freq_mhz
#       if output_path is None:
#           output_path = f"/tmp/nl_dpe_{rows}x{cols}.json"
#       with open(output_path, 'w') as f:
#           json.dump(cfg, f, indent=4)
#       return output_path
#
#   ─── IMC Simulator Invocation ───────────────────────────────────────
#
#   def run_imc_fc(config_path, K, N):
#       """Run IMC simulator for a single FC+activation layer.
#
#       Option A (subprocess — simpler, uses existing test.py CLI):
#           cmd = f"python azurelily/IMC/test.py --model fc --imc_file {config_path}"
#                 f" --seq_length {K} --head_dim {N}"
#           # Parse stdout for energy/latency (regex from run_imc_with_vtr_freq.py)
#
#       Option B (in-process — faster, no subprocess overhead):
#           sys.path.insert(0, 'azurelily')
#           from IMC.simulator import IMC
#           from models.fc import fc_model
#           imc = IMC(config_path)
#           all_layers, _ = fc_model(1, 1, K, N, False, energy_stats_template())
#           for layer in all_layers:
#               imc.run_layer(layer)
#           imc.finalize_latency_stats()
#           energy_pj = sum(imc.energy_breakdown.values())
#           latency_ns = sum(imc.latency_stats.values())
#           return energy_pj, latency_ns, dict(imc.energy_breakdown)
#
#       Recommendation: Option B for Round 1 (54 calls, in-process is faster).
#                       Option A as fallback / for debugging.
#       """
#       pass
#
#   ─── Round 2 Pipeline ───────────────────────────────────────────────
#
#   def run_round2(top_configs, r1_results, output_dir, jobs, template_scale=1.2):
#       """
#       Round 2 has two parts:
#         Part 1: CLB replacement sweep (72 runs)
#         Part 2: DSP+BRAM equivalence (72 runs)
#
#       Template generation:
#         1. From Round 1, find (128×64, fc_2048_256) auto_layout result
#         2. grid_template = (grid_w * template_scale, grid_h * template_scale)
#         3. Use grid_template as fixed_layout for all Round 2 runs
#       """
#
#       # ── Template ──
#       # Find the Round 1 result for smallest DPE + largest workload
#       template_result = find_result(r1_results, config="128x64", workload="fc_2048_256")
#       grid_w = int(template_result['grid_w'] * template_scale)
#       grid_h = int(template_result['grid_h'] * template_scale)
#       # Round up to even numbers for VTR compatibility
#       grid_w += grid_w % 2
#       grid_h += grid_h % 2
#
#       # ── Part 1: CLB replacement ──
#       CLB_RATIOS = [0.05, 0.08, 0.12, 0.15]
#       part1_results = []
#       for (R, C) in top_configs:
#           specs = dpe_specs(R, C)
#           for ratio in CLB_RATIOS:
#               # Compute number of DPEs that fit in ratio% of CLB area
#               # total_clb_cells ≈ grid_w * grid_h * fraction_clb
#               # CLB area in MWTA per cell = 27905
#               # n_dpes = floor(ratio * total_clb_cells * 27905 / area_tag_mwta)
#               #
#               # Generate fixed_layout arch XML with CLB columns replaced by DPE columns
#               gen_arch_xml(R, C, mode="fixed_clb_replace",
#                           fixed_grid_w=grid_w, fixed_grid_h=grid_h,
#                           clb_replace_ratio=ratio)
#               # Run VTR + IMC for 6 workloads (same as Round 1 inner loop)
#               # Collect: throughput/mm², throughput/J, DPE utilization, CLB count
#
#       # ── Part 2: DSP+BRAM equivalence ──
#       DSP_BRAM_PAIRS = [
#           ("all_dsp",       lambda area: (int(area / 253779), 0)),
#           ("balanced",      lambda area: solve_balanced(area)),     # X'/Y' = X/Y
#           ("equal_area",    lambda area: (int(area/2 / 253779), int(area/2 / 137668))),
#           ("storage_first", lambda area: (1, int((area - 253779) / 137668))),
#       ]
#       part2_results = []
#       for (R, C) in top_configs:
#           specs = dpe_specs(R, C)
#           # Get n_dpes from Part 1 (or Round 1) for this config
#           n_dpes = get_n_dpes_for_config(R, C, r1_results)
#           total_dpe_area = n_dpes * specs['area_tag_mwta']
#           for pair_name, pair_fn in DSP_BRAM_PAIRS:
#               x_prime, y_prime = pair_fn(total_dpe_area)
#               gen_arch_xml(R, C, mode="fixed_dsp_bram",
#                           fixed_grid_w=grid_w, fixed_grid_h=grid_h,
#                           extra_dsps=x_prime, extra_brams=y_prime)
#               # Run VTR + IMC for 6 workloads
#               # But RTL uses DSPs instead of DPE black boxes (different RTL!)
#               # This requires a DSP-based GEMM wrapper — separate from DPE RTL
#
#       # Write CSV
#       write_csv(part1_results, output_dir / "round2_part1_results.csv")
#       write_csv(part2_results, output_dir / "round2_part2_results.csv")
#
#   ─── Output CSV Schema ──────────────────────────────────────────────
#
#   # Round 1 CSV columns:
#   ROUND1_COLUMNS = [
#       'config', 'rows', 'cols',
#       'workload', 'K', 'N',
#       'V', 'H', 'dpe_count', 'acam_eligible',
#       'fmax_mhz', 'grid_w', 'grid_h', 'fpga_area_mm2',
#       'latency_ns', 'energy_pj',
#       'throughput_per_mm2', 'throughput_per_J',
#       'e_imc_vmm', 'e_imc_digital', 'e_clb_reduction',
#       'e_clb_activation', 'e_sram',
#       'clb_count', 'dsp_count', 'mem_count', 'wc_count',
#   ]
#
#   # Round 2 Part 1 adds:
#   ROUND2_P1_EXTRA = ['clb_replace_ratio', 'n_dpes_placed', 'dpe_utilization']
#
#   # Round 2 Part 2 adds:
#   ROUND2_P2_EXTRA = ['pair_name', 'x_prime_dsps', 'y_prime_brams']
#
#   ─── Top-K Selection ────────────────────────────────────────────────
#
#   def select_top_configs(results, k=3, metric='throughput_per_mm2'):
#       """Select top-K configs by average metric across workloads.
#
#       For each config, compute mean(metric) over all workloads.
#       Return top-K config tuples sorted descending.
#       """
#       from collections import defaultdict
#       config_scores = defaultdict(list)
#       for r in results:
#           config_scores[r['config']].append(r[metric])
#       avg_scores = {cfg: sum(v)/len(v) for cfg, v in config_scores.items()}
#       sorted_configs = sorted(avg_scores.items(), key=lambda x: -x[1])
#       return [cfg for cfg, _ in sorted_configs[:k]]
#
#
# DEPENDENCIES:
#   - nl_dpe/area_power.py (dpe_specs)
#   - nl_dpe/gen_arch_xml.py (gen_arch_xml)
#   - nl_dpe/gen_gemv_wrappers.py (gen_fc_wrapper)
#   - nl_dpe/run_vtr.py (VTR invocation — reuse parse_metrics, parse_resources)
#   - azurelily/IMC/simulator.py (IMC class)
#   - azurelily/models/fc.py (fc_model)
#
# EFFORT: ~350 lines
#
# EXECUTION ORDER:
#   1. Implement fc.py + register in test.py (5 min)
#   2. Implement gen_arch_xml.py (30 min)
#   3. Extend gen_gemv_wrappers.py with generic stacking + activation (45 min)
#   4. Implement gemv_dse.py Round 1 pipeline (30 min)
#   5. Test Round 1 end-to-end on 1 config × 1 workload
#   6. Run full Round 1 (54 VTR runs)
#   7. Analyze results, select top-3
#   8. Implement gemv_dse.py Round 2 pipeline (30 min)
#   9. Run Round 2 (144 VTR runs)


# =============================================================================
# DEPENDENCY GRAPH
# =============================================================================
#
#   area_power.py (DONE)
#        │
#        ├──→ gen_arch_xml.py ──→ per-(R,C) arch XML files
#        │                              │
#        └──→ gen_gemv_wrappers.py ──→ per-(R,C,K,N) RTL files
#                                       │
#                                       ▼
#                               run_vtr.py (existing)
#                                       │
#                                       ▼
#                              VTR output: Fmax, grid_W×H, resources
#                                       │
#        area_power.py ─────────────────┤
#        (e_analogue, e_digital)        │
#                                       ▼
#                              patch_imc_config() → patched nl_dpe.json
#                                       │
#        fc.py (new) ───────────────────┤
#                                       ▼
#                              IMC simulator → energy_pj, latency_ns
#                                       │
#                                       ▼
#                              gemv_dse.py → CSV + plots
#
# =============================================================================
